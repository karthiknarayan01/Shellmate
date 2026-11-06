"""ModelClient for DeepSeek's API, which uses the OpenAI Chat Completions format.

Two things DeepSeek needs that the Anthropic adapter doesn't:
- Reasoning text. DeepSeek's thinking mode returns it as `reasoning_content`, and rejects a
  request with tools (HTTP 400) unless every earlier assistant message carries its
  `reasoning_content` back. It is stored on each assistant Message and always sent back.
- Tool arguments arrive as a JSON string, which the model sometimes gets wrong. A string that
  isn't a JSON object becomes `arguments_error` on the ToolCall, and the model gets an error
  result for that call instead of the runtime crashing.

Retries on 429, 5xx and connection errors are done by the SDK (`max_retries` attempts).
"""

from __future__ import annotations

import json
import os
from typing import Any

import openai

from runtime.messages import Message, ToolCall
from runtime.model import DEFAULT_MAX_TOKENS, ModelReply, ToolSpec, Usage

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


class DeepSeekModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        client: openai.OpenAI | None = None,
    ):
        """`client` defaults to one for DeepSeek's endpoint, keyed by DEEPSEEK_API_KEY."""
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or openai.OpenAI(
            base_url=BASE_URL, api_key=_require_api_key(), max_retries=max_retries
        )

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        """Send one request and return the parsed reply."""
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": to_api_messages(system, messages),
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]
        return from_api_response(self._client.chat.completions.create(**request))


def _require_api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    return key


def to_api_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Runtime messages -> Chat Completions `messages`, with the system prompt first.

    Each tool result is its own `role: tool` message. The API has no error flag on tool
    results, so an error result is sent with an "ERROR: " prefix.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            out.append(_assistant_message(m))
        else:
            content = f"ERROR: {m.text}" if m.is_error else m.text
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
    return out


def _assistant_message(m: Message) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": m.text or None}
    if m.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
            }
            for c in m.tool_calls
        ]
    if m.reasoning:
        message["reasoning_content"] = m.reasoning
    return message


def from_api_response(response: Any) -> ModelReply:
    """Chat Completions response -> ModelReply. finish_reason "length" means it hit the output cap."""
    choice = response.choices[0]
    message = choice.message
    calls = tuple(_parse_tool_call(c) for c in (message.tool_calls or []))
    usage = response.usage
    return ModelReply(
        text=message.content or "",
        reasoning=getattr(message, "reasoning_content", None) or "",
        tool_calls=calls,
        usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage(),
        truncated=choice.finish_reason == "length",
    )


def _parse_tool_call(api_call: Any) -> ToolCall:
    """Decode the JSON argument string; anything that isn't a JSON object is an arguments_error."""
    raw = api_call.function.arguments or "{}"
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as e:
        return ToolCall(api_call.id, api_call.function.name, {}, f"not valid JSON ({e.msg}): {raw!r}")
    if not isinstance(arguments, dict):
        return ToolCall(api_call.id, api_call.function.name, {}, f"not a JSON object: {raw!r}")
    return ToolCall(api_call.id, api_call.function.name, arguments)
