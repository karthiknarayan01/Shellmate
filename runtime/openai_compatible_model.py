"""ModelClient for any OpenAI-compatible Chat Completions API.

That covers DeepSeek's API and self-hosted open-source models (vLLM, SGLang and Ollama all
serve this API), as well as OpenAI itself. The provider is chosen by `base_url` and `model`;
see providers.py for the presets.

Two things differ between these servers and are handled here:
- Reasoning text. Thinking models return it as `reasoning_content` (some servers use
  `reasoning`). DeepSeek rejects a request with tools unless every earlier assistant message
  carries its `reasoning_content` back, so with `send_reasoning=True` it is included.
- Tool arguments arrive as a JSON string, which open-source models sometimes get wrong. A
  string that isn't a JSON object becomes `arguments_error` on the ToolCall, and the model
  gets an error result for that call instead of the runtime crashing.

Retries on 429, 5xx and connection errors are done by the SDK (`max_retries` attempts).
"""

from __future__ import annotations

import json
from typing import Any

import openai

from runtime.messages import Message, ToolCall
from runtime.model import ModelReply, ToolSpec, Usage


class OpenAICompatibleModel:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        send_reasoning: bool = False,
        max_tokens: int = 4096,
        max_retries: int = 3,
        client: openai.OpenAI | None = None,
    ):
        """`base_url=None` means OpenAI. Local servers that need no key accept any `api_key`."""
        self._model = model
        self._send_reasoning = send_reasoning
        self._max_tokens = max_tokens
        self._client = client or openai.OpenAI(
            base_url=base_url, api_key=api_key, max_retries=max_retries
        )

    def complete(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ModelReply:
        """Send one request and return the parsed reply."""
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": to_api_messages(system, messages, self._send_reasoning),
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


def to_api_messages(
    system: str, messages: list[Message], send_reasoning: bool
) -> list[dict[str, Any]]:
    """Runtime messages -> Chat Completions `messages`, with the system prompt first.

    Each tool result is its own `role: tool` message. The API has no error flag on tool
    results, so an error result is sent with an "ERROR: " prefix.
    """
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            out.append(_assistant_message(m, send_reasoning))
        else:
            content = f"ERROR: {m.text}" if m.is_error else m.text
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
    return out


def _assistant_message(m: Message, send_reasoning: bool) -> dict[str, Any]:
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
    if send_reasoning and m.reasoning:
        message["reasoning_content"] = m.reasoning
    return message


def from_api_response(response: Any) -> ModelReply:
    """Chat Completions response -> ModelReply."""
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
    calls = tuple(_parse_tool_call(c) for c in (message.tool_calls or []))
    usage = response.usage
    return ModelReply(
        text=message.content or "",
        reasoning=reasoning or "",
        tool_calls=calls,
        usage=Usage(usage.prompt_tokens, usage.completion_tokens) if usage else Usage(),
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
