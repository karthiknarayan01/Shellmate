"""The agent loop for one session.

One turn = the user's message, then model steps until the model replies without tool calls:

    record user_message
    repeat (at most max_steps):
        reply = model(system, messages, tools)       record assistant_message
        no tool calls?  -> the turn is over, return the reply text
        for each call:  record tool_started -> run -> record tool_finished

Every change goes through `_record()`, which appends the event to the log first and then
applies it to the in-memory state. The state is therefore always what a replay of the log
would produce, and a process that dies at any point leaves a log that `Session.open()` can
rebuild.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from runtime.events import EventLog
from runtime.model import ModelClient, ModelReply
from runtime.state import SessionState, apply, project
from runtime.tools import ToolRegistry

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the tools to look things up instead of guessing. "
    "When you have the answer, reply without calling a tool."
)


class MaxStepsReached(RuntimeError):
    """The model kept calling tools for max_steps steps without giving a final answer."""


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
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 20,
    ):
        """Use Session.create() or Session.open() rather than calling this directly."""
        self.log = log
        self.session_id = session_id
        self.state = state
        self.model = model
        self.tools = tools
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

    def run_turn(self, user_text: str) -> str:
        """Run one turn and return the model's final text."""
        if self.state.open_calls:
            raise OpenToolCalls(f"tool calls without a result: {list(self.state.open_calls)}")
        self._record("user_message", {"text": user_text})
        for _ in range(self.max_steps):
            reply = self._step()
            if not reply.tool_calls:
                return reply.text
        raise MaxStepsReached(f"no final answer after {self.max_steps} steps")

    def _step(self) -> ModelReply:
        """One model call, then every tool call it asked for, in order."""
        reply = self.model.complete(self.system_prompt, self.state.messages, self.tools.specs())
        self._record(
            "assistant_message",
            {
                "text": reply.text,
                "tool_calls": [asdict(c) for c in reply.tool_calls],
                "usage": asdict(reply.usage),
            },
        )
        for call in reply.tool_calls:
            self._record("tool_started", {"call_id": call.id})
            result = self.tools.run(call)
            self._record(
                "tool_finished",
                {"call_id": call.id, "output": result.output, "is_error": result.is_error},
            )
        return reply

    def _record(self, type: str, payload: dict[str, Any]) -> None:
        """Append the event (durable once this returns), then fold it into the state."""
        apply(self.state, self.log.append(self.session_id, type, payload))
