"""A scripted ModelClient for tests: returns prepared replies in order and records every request.

Like the real clients, it streams: given `on_delta`, it reports a reply's reasoning and text
in small pieces before returning the whole reply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from runtime.messages import Message, ToolCall
from runtime.model import DeltaSink, ModelReply, StreamDelta, ToolSpec


@dataclass
class Request:
    system: str
    messages: list[Message]
    tools: list[ToolSpec]


@dataclass
class FakeModel:
    replies: list[ModelReply]
    requests: list[Request] = field(default_factory=list)

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: DeltaSink | None = None,
    ) -> ModelReply:
        self.requests.append(Request(system, list(messages), list(tools)))
        if not self.replies:
            raise AssertionError("FakeModel ran out of scripted replies")
        reply = self.replies.pop(0)
        if on_delta is not None:
            for kind, text in (("reasoning", reply.reasoning), ("text", reply.text)):
                for piece in _pieces(text):
                    on_delta(StreamDelta(kind, piece))
        return reply


def say(text: str) -> ModelReply:
    """A final reply with no tool calls."""
    return ModelReply(text=text)


def call(name: str, call_id: str = "c1", **arguments) -> ModelReply:
    """A reply that calls one tool."""
    return ModelReply(tool_calls=(ToolCall(call_id, name, arguments),))


def _pieces(text: str) -> list[str]:
    """Split text the way a stream delivers it, a few words at a time. Joined, they give `text`."""
    words = text.split(" ") if text else []
    pieces = []
    for i in range(0, len(words), 3):
        piece = " ".join(words[i : i + 3])
        if i + 3 < len(words):
            piece += " "
        pieces.append(piece)
    return pieces
