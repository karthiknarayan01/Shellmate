"""ModelClient for Anthropic's Messages API.

Converts the runtime's Message list to the API's format and the API's response back to a
ModelReply. Retries on rate limits, overload, 5xx and connection errors are done by the SDK
(exponential backoff with jitter, `max_retries` attempts).
"""

from __future__ import annotations

from typing import Any

import os

import anthropic

from runtime.messages import Message, ToolCall
from runtime.model import DEFAULT_MAX_TOKENS, ModelError, ModelReply, ToolSpec, Usage

DEFAULT_MODEL = "claude-sonnet-5-5"
# Above this, the SDK refuses non-streaming requests (they could run past 10 minutes).
MAX_NON_STREAMING_TOKENS = 21_333


class AnthropicModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        client: anthropic.Anthropic | None = None,
    ):
        """`client` defaults to one reading ANTHROPIC_API_KEY from the environment."""
        if max_tokens > MAX_NON_STREAMING_TOKENS:
            raise ValueError(
                f"max_tokens {max_tokens} is above {MAX_NON_STREAMING_TOKENS}, the most the "
                "Anthropic SDK allows without streaming"
            )
        self._model = model
        self._max_tokens = max_tokens
        if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self._client = client or anthropic.Anthropic(max_retries=max_retries)

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        """Send one request and return the parsed reply. API failures raise ModelError."""
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=to_api_messages(messages),
                tools=[
                    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                    for t in tools
                ],
            )
        except anthropic.APIError as e:
            raise ModelError(f"Anthropic API error: {e}") from e
        return from_api_response(response)


def to_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """Runtime messages -> Messages API `messages`.

    The API has only two roles. Tool results are sent as `tool_result` blocks inside a user
    message, and all results answering one assistant reply must be in the same user message,
    so consecutive tool messages are merged into one.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": [{"type": "text", "text": m.text}]})
        elif m.role == "assistant":
            out.append({"role": "assistant", "content": _assistant_blocks(m)})
        else:
            block = {
                "type": "tool_result",
                "tool_use_id": m.tool_call_id,
                "content": m.text,
                "is_error": m.is_error,
            }
            previous = out[-1] if out else None
            if previous and previous["role"] == "user" and _is_tool_results(previous):
                previous["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
    return out


def _assistant_blocks(m: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if m.text:
        blocks.append({"type": "text", "text": m.text})
    for call in m.tool_calls:
        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
    return blocks


def _is_tool_results(api_message: dict[str, Any]) -> bool:
    return all(block["type"] == "tool_result" for block in api_message["content"])


def from_api_response(response: Any) -> ModelReply:
    """Messages API response -> ModelReply. Text blocks are joined; tool_use blocks become calls.

    stop_reason "max_tokens" means generation hit the output cap.
    """
    texts = [block.text for block in response.content if block.type == "text"]
    calls = tuple(
        ToolCall(id=block.id, name=block.name, arguments=dict(block.input))
        for block in response.content
        if block.type == "tool_use"
    )
    usage = Usage(response.usage.input_tokens, response.usage.output_tokens)
    return ModelReply(
        text="\n".join(texts),
        tool_calls=calls,
        usage=usage,
        truncated=response.stop_reason == "max_tokens",
    )
