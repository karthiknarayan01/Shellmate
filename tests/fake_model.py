"""A scripted ModelClient for tests: returns prepared replies in order and records every request."""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.messages import Message, ToolCall
from runtime.model import ModelReply, ToolSpec


@dataclass
class Request:
    system: str
    messages: list[Message]
    tools: list[ToolSpec]


@dataclass
class FakeModel:
    replies: list[ModelReply]
    requests: list[Request] = field(default_factory=list)

    async def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        self.requests.append(Request(system, list(messages), list(tools)))
        if not self.replies:
            raise AssertionError("FakeModel ran out of scripted replies")
        return self.replies.pop(0)


def say(text: str) -> ModelReply:
    """A final reply with no tool calls."""
    return ModelReply(text=text)


def call(name: str, call_id: str = "c1", **arguments) -> ModelReply:
    """A reply that calls one tool."""
    return ModelReply(tool_calls=(ToolCall(call_id, name, arguments),))
