"""The conversation as the model sees it: messages and tool calls, independent of any provider.

The model client converts these to and from a provider's wire format. Everything else in the
runtime works with these types only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["user", "assistant", "tool"]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A request from the model to run one tool. `id` is chosen by the model provider."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Message:
    """One message in the conversation.

    user       text
    assistant  text and/or tool_calls
    tool       the result of one tool call: text, tool_call_id, is_error
    """

    role: Role
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None
    is_error: bool = False
