"""The agent loop for one session, including resuming a turn after a crash.

One turn = the user's message, then model steps until the model replies without tool calls:

    record user_message
    repeat (at most max_steps):
        reply = model(system, messages, tools)
        cut off at the output token cap?  -> record turn_stopped, raise ReplyTruncated
        record assistant_message
        no tool calls?  -> the turn is over, return the reply text
        resolve each tool call, in order (see _resolve)
    step limit reached  -> record turn_stopped, raise MaxStepsReached

Every change goes through `_record()`, which appends the event to the log first and then
applies it to the in-memory state. If the process dies, the log holds everything up to the
last completed step, and `Session.open()` + `resume()` carry the turn on from there:

    call status after replay   what resume does
    requested                  handle it as if new (check, approval, run)
    awaiting_approval          ask the approver again
    approved                   run it
    started                    idempotent tool  -> run it again (tool_restarted)
                               other tool       -> don't run it; the model is told it was
                                                   interrupted and may or may not have happened
    (no open calls)            continue with the next model step

A started, non-idempotent tool is never run a second time, because the log can't tell whether
its side effect happened before the crash.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from runtime.approval import Approver, Decision, DenyAll
from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.model import ModelClient
from runtime.state import SessionState, ToolCallState, ToolCallStatus, apply, project
from runtime.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools to look things up instead of guessing. "
    "When you have the answer, reply without calling a tool."
)

INTERRUPTED_RESULT = (
    "Interrupted: the process stopped while this tool was running, so it may or may not have "
    "completed. Check its effect before running it again."
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


class TurnInProgress(RuntimeError):
    """A turn was interrupted by a crash and hasn't been resumed yet."""


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
        """Continue an existing session: its state is rebuilt by replaying its events.

        If `needs_resume` is True afterwards, call resume() before the next run_turn().
        """
        return cls(log, session_id, project(log.load(session_id)), model, tools, **options)

    @property
    def needs_resume(self) -> bool:
        """True when the last turn was interrupted and never ended."""
        return self.state.turn_open

    def run_turn(self, user_text: str) -> TurnResult:
        """Run one turn: until the model answers without tools, or the user stops it."""
        if self.state.turn_open:
            raise TurnInProgress("the previous turn was interrupted; call resume() first")
        self._record("user_message", {"text": user_text})
        return self._continue_turn()

    def resume(self) -> TurnResult | None:
        """Finish a turn interrupted by a crash. Returns None if there was nothing to resume."""
        if not self.state.turn_open:
            return None
        denied = self._resolve_open_calls()
        if denied is not None:
            return self._stop_after_denial(denied)
        return self._continue_turn()

    def _continue_turn(self) -> TurnResult:
        """Model steps until a final answer, a denial that stops the turn, or the step limit."""
        for _ in range(self.max_steps):
            result = self._step()
            if result is not None:
                return result
        self._record("turn_stopped", {"reason": "max_steps"})
        raise MaxStepsReached(f"no final answer after {self.max_steps} steps")

    def _step(self) -> TurnResult | None:
        """One model call, then its tool calls in order. Returns None if the turn continues.

        A truncated reply is not recorded: its last tool call may have incomplete arguments,
        and its text may be cut mid-sentence. The turn is stopped instead.
        """
        reply = self.model.complete(self.system_prompt, self.state.messages, self.tools.specs())
        if reply.truncated:
            self._record("turn_stopped", {"reason": "truncated"})
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
        denied = self._resolve_open_calls()
        if denied is not None:
            return self._stop_after_denial(denied)
        return None

    def _resolve_open_calls(self) -> ToolCall | None:
        """Give every open call a result, in call order.

        Returns the call whose denial (without feedback) stopped the turn, after skipping the
        calls that came after it; otherwise None.
        """
        for tool in self.state.unfinished_calls():
            decision = self._resolve(tool)
            if decision is not None and not decision.approved and not decision.feedback:
                self._skip_open_calls(reason=f"skipped: the user denied {tool.call.name}")
                return tool.call
        return None

    def _resolve(self, tool: ToolCallState) -> Decision | None:
        """Move one call to FINISHED from whatever status it is in.

        Returns the approval decision, if one was made.
        """
        call = tool.call
        if tool.status is ToolCallStatus.STARTED:
            self._recover_started(call)
            return None
        if tool.status is ToolCallStatus.REQUESTED:
            error = self.tools.check(call)
            if error is not None:
                self._record("tool_skipped", {"call_id": call.id, "reason": error.output})
                return None
            if not self.tools.requires_approval(call.name):
                self._run(call)
                return None
            self._record("approval_requested", {"call_id": call.id})
        decision = None
        if tool.status is ToolCallStatus.AWAITING_APPROVAL:
            decision = self._decide(call)
            if not decision.approved:
                return decision
        self._run(call)  # status is APPROVED here
        return decision

    def _recover_started(self, call: ToolCall) -> None:
        """A call that was running when the process died: re-run it only if that's safe."""
        if self.tools.is_idempotent(call.name):
            self._record("tool_restarted", {"call_id": call.id})
            result = self.tools.execute(call)
            self._finish(call, result.output, result.is_error)
        else:
            self._finish(call, INTERRUPTED_RESULT, is_error=True, interrupted=True)

    def _decide(self, call: ToolCall) -> Decision:
        """Get and record a decision for a call awaiting approval.

        A tool approved for the session is approved without asking. Otherwise the approver is
        asked; this is where an interactive approver blocks until a person answers.
        """
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
        self._finish(call, result.output, result.is_error)

    def _finish(self, call: ToolCall, output: str, is_error: bool, interrupted: bool = False) -> None:
        payload = {"call_id": call.id, "output": output, "is_error": is_error}
        if interrupted:
            payload["interrupted"] = True
        self._record("tool_finished", payload)

    def _skip_open_calls(self, reason: str) -> None:
        """Give every call still open an error result without running it."""
        for tool in self.state.unfinished_calls():
            self._record("tool_skipped", {"call_id": tool.call.id, "reason": reason})

    def _stop_after_denial(self, denied: ToolCall) -> TurnResult:
        self._record("turn_stopped", {"reason": "denied"})
        last_text = next(m.text for m in reversed(self.state.messages) if m.role == "assistant")
        return TurnResult(last_text, denied_call=denied)

    def _record(self, type: str, payload: dict[str, Any]) -> None:
        """Append the event (durable once this returns), then fold it into the state."""
        apply(self.state, self.log.append(self.session_id, type, payload))
