"""The agent loop for one session, including resuming a turn after a crash.

One turn = the user's message, then model steps until the model replies without tool calls:

    record user_message
    repeat (at most max_steps):
        next request near the context window?  -> compact (see compaction.py)
        reply = model(system, messages, tools)
        cut off at the output token cap?  -> record turn_stopped, raise ReplyTruncated
        record assistant_message
        no tool calls?  -> the turn is over, return the reply text
        resolve the tool calls (see _resolve_open_calls):
            1. in call order: validate each call, ask for approval where needed
            2. run every call that may run, at the same time (at most max_parallel_tools)
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

Everything runs on one asyncio event loop in one thread. Model requests and tool calls are
awaited, so while they wait on the network or a worker thread, other tool calls make progress.
Events are only ever written from the loop, so the log still has a single writer.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal

from runtime.approval import Approver, Decision, DenyAll
from runtime.compaction import (
    SUMMARY_SYSTEM_PROMPT,
    CompactionPolicy,
    choose_cut,
    estimate_tokens,
    policy_from_env,
    render_transcript,
    summary_message_text,
)
from runtime.events import Event, EventLog
from runtime.messages import Message, ToolCall
from runtime.model import ModelClient
from runtime.state import SessionState, ToolCallState, ToolCallStatus, apply, project
from runtime.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools to look things up instead of guessing. "
    "When you have the answer, reply without calling a tool."
)

DEFAULT_MAX_PARALLEL_TOOLS = 8

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


class CompactionFailed(RuntimeError):
    """The summary request was cut off at the output token cap."""


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
        max_parallel_tools: int = DEFAULT_MAX_PARALLEL_TOOLS,
        compaction: CompactionPolicy | None = None,
        observer: Callable[[Event], None] | None = None,
    ):
        """Use Session.create() or Session.open() rather than calling this directly.

        Without an approver, every tool that requires approval is denied. Without a compaction
        policy, the context window comes from AGENT_CONTEXT_WINDOW or the default. `observer`
        is called with every event right after it is recorded (a UI uses it to show progress).
        """
        self.log = log
        self.session_id = session_id
        self.state = state
        self.model = model
        self.tools = tools
        self.approver = approver or DenyAll()
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_parallel_tools = max_parallel_tools
        self.compaction = compaction or policy_from_env()
        self.observer = observer

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

    async def run_turn(self, user_text: str) -> TurnResult:
        """Run one turn: until the model answers without tools, or the user stops it."""
        if self.state.turn_open:
            raise TurnInProgress("the previous turn was interrupted; call resume() first")
        self._record("user_message", {"text": user_text})
        return await self._continue_turn()

    async def resume(self) -> TurnResult | None:
        """Finish a turn interrupted by a crash. Returns None if there was nothing to resume."""
        if not self.state.turn_open:
            return None
        denied = await self._resolve_open_calls()
        if denied is not None:
            return self._stop_after_denial(denied)
        return await self._continue_turn()

    def cancel(self) -> None:
        """End an interrupted turn without running anything more (the user pressed Ctrl+C).

        Calls that never started are skipped. A call that had started gets the interrupted
        result, since it may have had its effect. Then the turn is recorded as stopped.
        """
        if not self.state.turn_open:
            return
        self._abandon_open_calls(reason="cancelled by the user")
        self._record("turn_stopped", {"reason": "cancelled"})

    async def _continue_turn(self) -> TurnResult:
        """Model steps until a final answer, a denial that stops the turn, or the step limit."""
        for _ in range(self.max_steps):
            result = await self._step()
            if result is not None:
                return result
        self._record("turn_stopped", {"reason": "max_steps"})
        raise MaxStepsReached(f"no final answer after {self.max_steps} steps")

    async def _step(self) -> TurnResult | None:
        """One model call, then its tool calls. Returns None if the turn continues.

        A truncated reply is not recorded: its last tool call may have incomplete arguments,
        and its text may be cut mid-sentence. The turn is stopped instead.
        """
        await self._compact_if_needed()
        reply = await self.model.complete(self.system_prompt, self.state.messages, self.tools.specs())
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
        denied = await self._resolve_open_calls()
        if denied is not None:
            return self._stop_after_denial(denied)
        return None

    async def _compact_if_needed(self) -> None:
        """Summarize the oldest messages if the next request would pass the trigger.

        The summary request runs before anything is recorded, so a crash during it leaves the
        log unchanged and the next attempt simply compacts again.
        """
        specs = self.tools.specs()
        if estimate_tokens(self.state, self.system_prompt, specs) < self.compaction.trigger_tokens:
            return
        cut = choose_cut(self.state.messages, self.compaction.keep_tokens)
        if cut is None:
            return
        transcript = render_transcript(self.state.messages[:cut])
        reply = await self.model.complete(SUMMARY_SYSTEM_PROMPT, [Message("user", transcript)], [])
        if reply.truncated:
            raise CompactionFailed("the summary hit the output token cap")
        self._record(
            "compaction",
            {
                "replaced_count": cut,
                "summary": summary_message_text(reply.text, self.state.turn_request),
                "usage": asdict(reply.usage),
            },
        )

    async def _resolve_open_calls(self) -> ToolCall | None:
        """Give every open call a result.

        Phase 1, one call at a time in call order: validate it and, if it needs approval, ask.
        Approvals are asked one by one because a person answers them one by one.
        Phase 2: run every call that phase 1 cleared, all at the same time.

        A denial without feedback stops the turn before anything runs: every call still open,
        including ones approved a moment earlier, is skipped. Returns that denied call, or None.
        """
        to_run: list[tuple[ToolCall, bool]] = []  # (call, is a re-run after a crash)
        for tool in self.state.unfinished_calls():
            action = await self._prepare(tool)
            if action == "stop":
                self._abandon_open_calls(reason=f"skipped: the user denied {tool.call.name}")
                return tool.call
            if action in ("run", "rerun"):
                to_run.append((tool.call, action == "rerun"))
        await self._run_all(to_run)
        return None

    async def _prepare(self, tool: ToolCallState) -> Literal["run", "rerun", "done", "stop"]:
        """Decide what happens to one open call, from whatever status it is in.

        run    it may run now          done  it already has its result (invalid, denied with
        rerun  idempotent, re-run it         feedback, or interrupted)
        stop   denied without feedback: the turn ends
        """
        call = tool.call
        if tool.status is ToolCallStatus.STARTED:
            if self.tools.is_idempotent(call.name):
                return "rerun"
            self._finish(call, INTERRUPTED_RESULT, is_error=True, interrupted=True)
            return "done"
        if tool.status is ToolCallStatus.REQUESTED:
            error = self.tools.check(call)
            if error is not None:
                self._record("tool_skipped", {"call_id": call.id, "reason": error.output})
                return "done"
            if not self.tools.requires_approval(call.name):
                return "run"
            self._record("approval_requested", {"call_id": call.id})
        if tool.status is ToolCallStatus.AWAITING_APPROVAL:
            decision = await self._decide(call)
            if not decision.approved:
                return "done" if decision.feedback else "stop"
        return "run"  # status is APPROVED here

    async def _run_all(self, calls: list[tuple[ToolCall, bool]]) -> None:
        """Run the calls concurrently, at most max_parallel_tools at a time.

        Each call's tool_started (or tool_restarted) is written right before it starts, and its
        tool_finished as soon as it ends, so the log always shows which calls were running.
        If one call is interrupted (cancelled, or a crash in the process), the others are
        cancelled too and the exception propagates.
        """
        limit = asyncio.Semaphore(self.max_parallel_tools)

        async def run_one(call: ToolCall, rerun: bool) -> None:
            async with limit:
                self._record("tool_restarted" if rerun else "tool_started", {"call_id": call.id})
                result = await self.tools.execute(call)
                self._finish(call, result.output, result.is_error)

        tasks = [asyncio.create_task(run_one(call, rerun)) for call, rerun in calls]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise

    async def _decide(self, call: ToolCall) -> Decision:
        """Get and record a decision for a call awaiting approval.

        A tool approved for the session is approved without asking. Otherwise the approver is
        asked; this is where an interactive approver blocks until a person answers.
        """
        if call.name in self.state.approved_tools:
            decision, source = Decision("approve"), "session"
        else:
            decision, source = await self.approver.ask(call), "approver"
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

    def _finish(self, call: ToolCall, output: str, is_error: bool, interrupted: bool = False) -> None:
        payload = {"call_id": call.id, "output": output, "is_error": is_error}
        if interrupted:
            payload["interrupted"] = True
        self._record("tool_finished", payload)

    def _abandon_open_calls(self, reason: str) -> None:
        """Give every call still open a result without running anything.

        A call that had started gets the interrupted result, since it may have had its effect;
        the others are skipped with `reason`.
        """
        for tool in self.state.unfinished_calls():
            if tool.status is ToolCallStatus.STARTED:
                self._finish(tool.call, INTERRUPTED_RESULT, is_error=True, interrupted=True)
            else:
                self._record("tool_skipped", {"call_id": tool.call.id, "reason": reason})

    def _stop_after_denial(self, denied: ToolCall) -> TurnResult:
        self._record("turn_stopped", {"reason": "denied"})
        last_text = next(m.text for m in reversed(self.state.messages) if m.role == "assistant")
        return TurnResult(last_text, denied_call=denied)

    def _record(self, type: str, payload: dict[str, Any]) -> None:
        """Append the event (durable once this returns), fold it into the state, tell the observer."""
        event = self.log.append(self.session_id, type, payload)
        apply(self.state, event)
        if self.observer is not None:
            self.observer(event)
