"""Rebuild a session's state by replaying its events.

The event log is the only stored state. `project()` folds a session's events, in order, into
a SessionState: the message list to send to the model, and the status of every tool call the
model has made. Running it on the same events always gives the same state, which is how a
restarted process recovers exactly where the previous one stopped.

Event types and payloads:

    user_message       {text}
    assistant_message  {text, tool_calls: [{id, name, arguments}]}
    tool_started       {call_id}
    tool_finished      {call_id, output, is_error}

A tool call moves requested -> started -> finished. Each step is a separate event, so a crash
between `tool_started` and `tool_finished` is visible after replay as a call left in `started`.
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
    REQUESTED = "requested"  # the model asked for it; it has not started
    STARTED = "started"  # tool_started written; no result yet
    FINISHED = "finished"  # tool_finished written; the result is in `messages`


@dataclass(slots=True)
class ToolCallState:
    call: ToolCall
    status: ToolCallStatus = ToolCallStatus.REQUESTED


@dataclass(slots=True)
class SessionState:
    messages: list[Message] = field(default_factory=list)
    tool_calls: dict[str, ToolCallState] = field(default_factory=dict)  # every call, by id
    open_calls: dict[str, ToolCallState] = field(default_factory=dict)  # not finished, in call order
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
    """A user message is only valid when no tool call from the previous reply is still open."""
    _require_no_open_calls(state, event)
    state.messages.append(Message(role="user", text=event.payload["text"]))


def _on_assistant_message(state: SessionState, event: Event) -> None:
    """Add the model's reply and register each tool call it made as REQUESTED."""
    _require_no_open_calls(state, event)
    calls = tuple(
        ToolCall(id=c["id"], name=c["name"], arguments=c["arguments"])
        for c in event.payload.get("tool_calls", [])
    )
    for call in calls:
        if call.id in state.tool_calls:
            raise InvalidEventSequence(f"seq {event.seq}: duplicate tool call id {call.id!r}")
        tool = ToolCallState(call)
        state.tool_calls[call.id] = tool
        state.open_calls[call.id] = tool
    state.messages.append(
        Message(role="assistant", text=event.payload.get("text", ""), tool_calls=calls)
    )


def _on_tool_started(state: SessionState, event: Event) -> None:
    """REQUESTED -> STARTED."""
    tool = _require_call(state, event, expected=ToolCallStatus.REQUESTED)
    tool.status = ToolCallStatus.STARTED


def _on_tool_finished(state: SessionState, event: Event) -> None:
    """STARTED -> FINISHED, and add the result to the conversation as a tool message."""
    tool = _require_call(state, event, expected=ToolCallStatus.STARTED)
    tool.status = ToolCallStatus.FINISHED
    del state.open_calls[tool.call.id]
    state.messages.append(
        Message(
            role="tool",
            text=event.payload["output"],
            tool_call_id=tool.call.id,
            is_error=event.payload.get("is_error", False),
        )
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


def _require_call(state: SessionState, event: Event, expected: ToolCallStatus) -> ToolCallState:
    """Look up the call an event refers to and check it is in the status the event moves it from."""
    call_id = event.payload["call_id"]
    tool = state.tool_calls.get(call_id)
    if tool is None:
        raise InvalidEventSequence(f"seq {event.seq}: {event.type} for unknown call {call_id!r}")
    if tool.status is not expected:
        raise InvalidEventSequence(
            f"seq {event.seq}: {event.type} for call {call_id!r} in status {tool.status.value}, "
            f"expected {expected.value}"
        )
    return tool


_HANDLERS = {
    "user_message": _on_user_message,
    "assistant_message": _on_assistant_message,
    "tool_started": _on_tool_started,
    "tool_finished": _on_tool_finished,
}
