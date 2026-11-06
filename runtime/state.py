"""Rebuild a session's state by replaying its events.

The event log is the only stored state. `project()` folds a session's events, in order, into
a SessionState: the message list to send to the model, and the status of every tool call the
model has made. Running it on the same events always gives the same state, which is how a
restarted process recovers exactly where the previous one stopped.

Event types and payloads:

    user_message        {text}
    assistant_message   {text, reasoning?, tool_calls: [{id, name, arguments, arguments_error?}]}
    approval_requested  {call_id}
    approval_decided    {call_id, decision: approve | approve_for_session | deny, feedback, source}
    tool_started        {call_id}
    tool_restarted      {call_id}               re-run after a crash interrupted it
    tool_finished       {call_id, output, is_error, interrupted?}
    tool_skipped        {call_id, reason}
    turn_stopped        {reason}                the turn ended without a final answer

A tool call's status moves through these transitions, one event each:

    requested --approval_requested--> awaiting_approval --approval_decided(approve)--> approved
    requested | approved --tool_started--> started --tool_finished--> finished
    awaiting_approval --approval_decided(deny)--> finished   (never started)
    requested --tool_skipped--> finished                     (never started)

Because each step is its own event, a crash leaves the call in the status it had reached:
`awaiting_approval` (nobody answered yet) or `started` (the tool may or may not have run).
`approve_for_session` decisions are collected in `approved_tools`, so they survive a restart.

`turn_open` is True from a user message until the turn ends, either with an assistant reply
that has no tool calls or with `turn_stopped`. If it is still True after replay, the process
died mid-turn and the turn can be resumed.
Events that break this order (finishing a call that never started, an unknown type, ...) mean
the log is corrupt, and replay stops with InvalidEventSequence rather than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from runtime.events import Event
from runtime.messages import Message, ToolCall


class ToolCallStatus(Enum):
    REQUESTED = "requested"  # the model asked for it; nothing has happened yet
    AWAITING_APPROVAL = "awaiting_approval"  # approval asked for; no answer yet
    APPROVED = "approved"  # approved; not started yet
    STARTED = "started"  # tool_started written; no result yet
    FINISHED = "finished"  # has a result in `messages` (ran, denied or skipped)


@dataclass(slots=True)
class ToolCallState:
    call: ToolCall
    status: ToolCallStatus = ToolCallStatus.REQUESTED


@dataclass(slots=True)
class SessionState:
    messages: list[Message] = field(default_factory=list)
    tool_calls: dict[str, ToolCallState] = field(default_factory=dict)  # every call, by id
    open_calls: dict[str, ToolCallState] = field(default_factory=dict)  # not finished, in call order
    approved_tools: set[str] = field(default_factory=set)  # approved for the rest of the session
    turn_open: bool = False
    last_seq: int = 0

    def unfinished_calls(self) -> list[ToolCallState]:
        """Tool calls that have no result yet, in the order the model made them."""
        return list(self.open_calls.values())


class InvalidEventSequence(ValueError):
    """The events cannot have been produced by a correct runtime."""


def project(events: Iterable[Event]) -> SessionState:
    """Replay events in order into a new SessionState (used once, when a session is opened)."""
    state = SessionState()
    for event in events:
        apply(state, event)
    return state


def apply(state: SessionState, event: Event) -> None:
    """Fold one event into `state` in place.

    A running session calls this with each event it has just appended, so its in-memory state
    stays equal to a full replay without reading the log back.
    """
    handler = _HANDLERS.get(event.type)
    if handler is None:
        raise InvalidEventSequence(f"seq {event.seq}: unknown event type {event.type!r}")
    handler(state, event)
    state.last_seq = event.seq


def _on_user_message(state: SessionState, event: Event) -> None:
    """Start a turn. Only valid once the previous turn has ended."""
    if state.turn_open:
        raise InvalidEventSequence(f"seq {event.seq}: user_message while a turn is still open")
    state.messages.append(Message(role="user", text=event.payload["text"]))
    state.turn_open = True


def _on_assistant_message(state: SessionState, event: Event) -> None:
    """Add the model's reply and register each tool call it made as REQUESTED."""
    _require_no_open_calls(state, event)
    calls = tuple(
        ToolCall(
            id=c["id"],
            name=c["name"],
            arguments=c["arguments"],
            arguments_error=c.get("arguments_error"),
        )
        for c in event.payload.get("tool_calls", [])
    )
    for call in calls:
        if call.id in state.tool_calls:
            raise InvalidEventSequence(f"seq {event.seq}: duplicate tool call id {call.id!r}")
        tool = ToolCallState(call)
        state.tool_calls[call.id] = tool
        state.open_calls[call.id] = tool
    state.messages.append(
        Message(
            role="assistant",
            text=event.payload.get("text", ""),
            reasoning=event.payload.get("reasoning", ""),
            tool_calls=calls,
        )
    )
    if not calls:
        state.turn_open = False  # a reply without tool calls is the final answer


def _on_approval_requested(state: SessionState, event: Event) -> None:
    """REQUESTED -> AWAITING_APPROVAL."""
    tool = _require_call(state, event, expected={ToolCallStatus.REQUESTED})
    tool.status = ToolCallStatus.AWAITING_APPROVAL


def _on_approval_decided(state: SessionState, event: Event) -> None:
    """AWAITING_APPROVAL -> APPROVED, or -> FINISHED with a "denied" result for the model."""
    tool = _require_call(state, event, expected={ToolCallStatus.AWAITING_APPROVAL})
    decision = event.payload["decision"]
    if decision == "deny":
        _finish(state, tool, denial_message(event.payload.get("feedback", "")), is_error=True)
        return
    if decision == "approve_for_session":
        state.approved_tools.add(tool.call.name)
    elif decision != "approve":
        raise InvalidEventSequence(f"seq {event.seq}: unknown decision {decision!r}")
    tool.status = ToolCallStatus.APPROVED


def denial_message(feedback: str) -> str:
    """The tool result the model sees for a denied call."""
    if feedback:
        return f"The user denied this tool call. Their feedback: {feedback}"
    return "The user denied this tool call."


def _on_tool_started(state: SessionState, event: Event) -> None:
    """REQUESTED or APPROVED -> STARTED."""
    tool = _require_call(
        state, event, expected={ToolCallStatus.REQUESTED, ToolCallStatus.APPROVED}
    )
    tool.status = ToolCallStatus.STARTED


def _on_tool_restarted(state: SessionState, event: Event) -> None:
    """STARTED -> STARTED. Records that a call interrupted by a crash is being run again."""
    _require_call(state, event, expected={ToolCallStatus.STARTED})


def _on_tool_finished(state: SessionState, event: Event) -> None:
    """STARTED -> FINISHED, with the tool's output as the result."""
    tool = _require_call(state, event, expected={ToolCallStatus.STARTED})
    _finish(state, tool, event.payload["output"], event.payload.get("is_error", False))


def _on_tool_skipped(state: SessionState, event: Event) -> None:
    """REQUESTED -> FINISHED without running, with the reason as an error result."""
    tool = _require_call(state, event, expected={ToolCallStatus.REQUESTED})
    _finish(state, tool, event.payload["reason"], is_error=True)


def _on_turn_stopped(state: SessionState, event: Event) -> None:
    """End the turn without a final answer (denied, step limit, truncated reply)."""
    _require_no_open_calls(state, event)
    if not state.turn_open:
        raise InvalidEventSequence(f"seq {event.seq}: turn_stopped with no open turn")
    state.turn_open = False


def _finish(state: SessionState, tool: ToolCallState, text: str, is_error: bool) -> None:
    """Close a call and add its result to the conversation as a tool message."""
    tool.status = ToolCallStatus.FINISHED
    del state.open_calls[tool.call.id]
    state.messages.append(
        Message(role="tool", text=text, tool_call_id=tool.call.id, is_error=is_error)
    )


def _require_no_open_calls(state: SessionState, event: Event) -> None:
    """Every tool call must have a result before the conversation moves on.

    Model APIs reject a history where an assistant tool call has no matching tool result,
    so a log in that shape could never be sent to the model.
    """
    if state.open_calls:
        open_ids = list(state.open_calls)
        raise InvalidEventSequence(
            f"seq {event.seq}: {event.type} while tool calls {open_ids} have no result"
        )


def _require_call(
    state: SessionState, event: Event, expected: set[ToolCallStatus]
) -> ToolCallState:
    """Look up the call an event refers to and check it is in a status the event moves it from."""
    call_id = event.payload["call_id"]
    tool = state.tool_calls.get(call_id)
    if tool is None:
        raise InvalidEventSequence(f"seq {event.seq}: {event.type} for unknown call {call_id!r}")
    if tool.status not in expected:
        allowed = " or ".join(sorted(s.value for s in expected))
        raise InvalidEventSequence(
            f"seq {event.seq}: {event.type} for call {call_id!r} in status {tool.status.value}, "
            f"expected {allowed}"
        )
    return tool


_HANDLERS = {
    "user_message": _on_user_message,
    "assistant_message": _on_assistant_message,
    "approval_requested": _on_approval_requested,
    "approval_decided": _on_approval_decided,
    "tool_started": _on_tool_started,
    "tool_restarted": _on_tool_restarted,
    "tool_finished": _on_tool_finished,
    "tool_skipped": _on_tool_skipped,
    "turn_stopped": _on_turn_stopped,
}
