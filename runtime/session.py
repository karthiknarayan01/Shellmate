"""The agent loop for one session.

One turn = the user's message, then model steps until the model replies without tool calls:

    record user_message
    repeat (at most max_steps):
        reply = model(system, messages, tools)
        cut off at the output token cap?  -> raise ReplyTruncated, record nothing
        record assistant_message
        no tool calls?  -> the turn is over, return the reply text
        for each call, in order:
            malformed (unknown tool, bad arguments)  -> record tool_skipped(error); next call
            needs approval                           -> record approval_requested, ask,
                                                        record approval_decided
                denied with feedback  -> the model sees the denial; next call
                denied, no feedback   -> skip the remaining calls; the turn stops
            record tool_started -> run -> record tool_finished

Every change goes through `_record()`, which appends the event to the log first and then
applies it to the in-memory state. The state is therefore always what a replay of the log
would produce, and a process that dies at any point leaves a log that `Session.open()` can
rebuild. A denied tool never runs: `tool_started` is only written after an approval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from runtime.approval import Approver, Decision, DenyAll
from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.model import ModelClient
from runtime.state import SessionState, apply, project
from runtime.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools to look things up instead of guessing. "
    "When you have the answer, reply without calling a tool."
)


@dataclass(frozen=True, slots=True)
class TurnResult:
    """How a turn ended. `denied_call` is set when the user denied a call and the turn stopped."""

    text: str
    denied_call: ToolCall | None = None


class MaxStepsReached(RuntimeError):
    """The model kept calling tools for max_steps steps without giving a final answer."""


class ReplyTruncated(RuntimeError):
    """The model hit its output token cap, so its reply may be incomplete. It was not recorded."""


class OpenToolCalls(RuntimeError):
    """The session's last turn stopped with tool calls that have no result."""


class Session:
    def __init__(
        self,
        log: EventLog,
        session_id: str,
        state: SessionState,
        model: ModelClient,
        tools: ToolRegistry,
        approver: Approver | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 20,
    ):
        """Use Session.create() or Session.open() rather than calling this directly.

        Without an approver, every tool that requires approval is denied.
        """
        self.log = log
        self.session_id = session_id
        self.state = state
        self.model = model
        self.tools = tools
        self.approver = approver or DenyAll()
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    @classmethod
    def create(cls, log: EventLog, model: ModelClient, tools: ToolRegistry, **options: Any) -> Session:
        """Start a new, empty session."""
        return cls(log, log.new_session(), SessionState(), model, tools, **options)

    @classmethod
    def open(
        cls, log: EventLog, session_id: str, model: ModelClient, tools: ToolRegistry, **options: Any
    ) -> Session:
        """Continue an existing session: its state is rebuilt by replaying its events."""
        return cls(log, session_id, project(log.load(session_id)), model, tools, **options)

    def run_turn(self, user_text: str) -> TurnResult:
        """Run one turn: until the model answers without tools, or the user stops it."""
        if self.state.open_calls:
            raise OpenToolCalls(f"tool calls without a result: {list(self.state.open_calls)}")
        self._record("user_message", {"text": user_text})
        for _ in range(self.max_steps):
            result = self._step()
            if result is not None:
                return result
        raise MaxStepsReached(f"no final answer after {self.max_steps} steps")

    def _step(self) -> TurnResult | None:
        """One model call, then its tool calls in order. Returns None if the turn continues.

        A truncated reply is rejected before anything is recorded: its last tool call may have
        incomplete arguments, and its text may be cut mid-sentence.
        """
        reply = self.model.complete(self.system_prompt, self.state.messages, self.tools.specs())
        if reply.truncated:
            raise ReplyTruncated(
                f"reply stopped at the output token cap after {reply.usage.output_tokens} tokens"
            )
        self._record(
            "assistant_message",
            {
                "text": reply.text,
                "reasoning": reply.reasoning,
                "tool_calls": [asdict(c) for c in reply.tool_calls],
                "usage": asdict(reply.usage),
            },
        )
        if not reply.tool_calls:
            return TurnResult(reply.text)

        for index, call in enumerate(reply.tool_calls):
            decision = self._handle_call(call)
            if decision is not None and not decision.approved and not decision.feedback:
                remaining = reply.tool_calls[index + 1 :]
                self._skip(remaining, reason=f"skipped: the user denied {call.name}")
                return TurnResult(reply.text, denied_call=call)
        return None

    def _handle_call(self, call: ToolCall) -> Decision | None:
        """Run one call, asking for approval first if needed. Returns the decision, if one was made."""
        error = self.tools.check(call)
        if error is not None:
            self._skip((call,), reason=error.output)
            return None
        decision = None
        if self.tools.requires_approval(call.name):
            decision = self._decide(call)
            if not decision.approved:
                return decision
        self._run(call)
        return decision

    def _decide(self, call: ToolCall) -> Decision:
        """Record the request, get a decision, record it.

        A tool approved for the session is approved without asking. Otherwise the approver is
        asked; this is where an interactive approver blocks until a person answers.
        """
        self._record("approval_requested", {"call_id": call.id})
        if call.name in self.state.approved_tools:
            decision, source = Decision("approve"), "session"
        else:
            decision, source = self.approver.ask(call), "approver"
        self._record(
            "approval_decided",
            {
                "call_id": call.id,
                "decision": decision.kind,
                "feedback": decision.feedback,
                "source": source,
            },
        )
        return decision

    def _run(self, call: ToolCall) -> None:
        """Record tool_started, run the tool, record tool_finished."""
        self._record("tool_started", {"call_id": call.id})
        result = self.tools.execute(call)
        self._record(
            "tool_finished",
            {"call_id": call.id, "output": result.output, "is_error": result.is_error},
        )

    def _skip(self, calls: tuple[ToolCall, ...], reason: str) -> None:
        """Give each call an error result without running it; every call needs a result."""
        for call in calls:
            self._record("tool_skipped", {"call_id": call.id, "reason": reason})

    def _record(self, type: str, payload: dict[str, Any]) -> None:
        """Append the event (durable once this returns), then fold it into the state."""
        apply(self.state, self.log.append(self.session_id, type, payload))
