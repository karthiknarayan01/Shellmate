"""What the agent loop needs from a language model, independent of provider.

A model client takes the system prompt, the conversation and the tool specs, and returns one
reply: text, tool calls, or both. Implemented by anthropic_model.py and deepseek_model.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from runtime.messages import Message, ToolCall

# Default output cap per model request; the model stops generating at this many tokens.
# Only tokens actually generated are billed, so the cap bounds the worst case per request.
# Override with AGENT_MAX_OUTPUT_TOKENS (see providers.py).
DEFAULT_MAX_TOKENS = 16_000


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool as the model sees it: its name, what it does, and a JSON schema for its arguments."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class ModelReply:
    """One model response. `truncated` is True when generation stopped at the output cap,
    so the text or a tool call's arguments may be incomplete."""

    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    usage: Usage = field(default_factory=Usage)
    truncated: bool = False


class ModelError(RuntimeError):
    """The provider's API failed after the client's retries (auth, quota, outage, bad request)."""


class ModelClient(Protocol):
    async def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelReply: ...
