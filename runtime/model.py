"""What the agent loop needs from a language model, independent of provider.

A model client takes the system prompt, the conversation and the tool specs, and returns one
reply: text, tool calls, or both. Provider adapters implement this:
anthropic_model.py (Anthropic Messages API) and openai_compatible_model.py (DeepSeek, OpenAI,
and self-hosted open-source models served by vLLM, SGLang or Ollama).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from runtime.messages import Message, ToolCall


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
    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    usage: Usage = field(default_factory=Usage)


class ModelClient(Protocol):
    def complete(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ModelReply: ...
