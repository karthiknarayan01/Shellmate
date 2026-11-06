"""ModelClient for DeepSeek's API, which uses the OpenAI Chat Completions format.

Two things DeepSeek needs that the Anthropic adapter doesn't:
- Reasoning text. DeepSeek's thinking mode returns it as `reasoning_content`, and rejects a
  request with tools (HTTP 400) unless every earlier assistant message carries its
  `reasoning_content` back. It is stored on each assistant Message and always sent back.
- Tool arguments arrive as a JSON string, which the model sometimes gets wrong. A string that
  isn't a JSON object becomes `arguments_error` on the ToolCall, and the model gets an error
  result for that call instead of the runtime crashing.

Replies are streamed. Each chunk carries a delta: a piece of text, a piece of reasoning, or a
piece of a tool call (its id and name come first, then the JSON arguments in fragments).
StreamAssembler puts the pieces back together into one ModelReply.

Retries on 429, 5xx and connection errors are done by the SDK (`max_retries` attempts) while
the stream is being opened; a stream that breaks after it started raises ModelError.
"""

from __future__ import annotations

import json
import os
from typing import Any

import openai

from runtime.messages import Message, ToolCall
from runtime.model import (
    DEFAULT_MAX_TOKENS,
    DeltaSink,
    ModelError,
    ModelReply,
    StreamDelta,
    ToolSpec,
    Usage,
)

BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"


class DeepSeekModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        client: openai.AsyncOpenAI | None = None,
    ):
        """`client` defaults to one for DeepSeek's endpoint, keyed by DEEPSEEK_API_KEY."""
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or openai.AsyncOpenAI(
            base_url=BASE_URL, api_key=_require_api_key(), max_retries=max_retries
        )

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: DeltaSink | None = None,
    ) -> ModelReply:
        """Stream one reply, passing text and reasoning to `on_delta` as they arrive; return the
        whole reply. API failures, including a stream that breaks halfway, raise ModelError."""
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": to_api_messages(system, messages),
            "stream": True,
            "stream_options": {"include_usage": True},  # token counts arrive in the last chunk
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
        assembler = StreamAssembler(on_delta)
        try:
            async for chunk in await self._client.chat.completions.create(**request):
                assembler.add(chunk)
        except openai.APIError as e:
            raise ModelError(f"DeepSeek API error: {e}") from e
        return assembler.reply()


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


class StreamAssembler:
    """Collects a streamed reply's chunks into one ModelReply, reporting text as it arrives."""

    def __init__(self, on_delta: DeltaSink | None = None):
        self._on_delta = on_delta
        self._text: list[str] = []
        self._reasoning: list[str] = []
        self._calls: dict[int, dict[str, Any]] = {}  # by index: id, name, argument fragments
        self._finish_reason: str | None = None
        self._usage = Usage()

    def add(self, chunk: Any) -> None:
        """Fold in one chunk. The last chunk has no choices, only usage."""
        if getattr(chunk, "usage", None):
            self._usage = Usage(chunk.usage.prompt_tokens, chunk.usage.completion_tokens)
        for choice in chunk.choices:
            delta = choice.delta
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                self._reasoning.append(reasoning)
                self._emit("reasoning", reasoning)
            if delta.content:
                self._text.append(delta.content)
                self._emit("text", delta.content)
            for piece in delta.tool_calls or []:
                call = self._calls.setdefault(piece.index, {"id": "", "name": "", "arguments": []})
                if piece.id:
                    call["id"] = piece.id
                if piece.function is not None:
                    if piece.function.name:
                        call["name"] = piece.function.name
                    if piece.function.arguments:
                        call["arguments"].append(piece.function.arguments)
            if choice.finish_reason:
                self._finish_reason = choice.finish_reason

    def reply(self) -> ModelReply:
        """The complete reply. finish_reason "length" means it stopped at the output cap."""
        calls = tuple(
            _parse_tool_call(c["id"], c["name"], "".join(c["arguments"]))
            for _, c in sorted(self._calls.items())
        )
        return ModelReply(
            text="".join(self._text),
            reasoning="".join(self._reasoning),
            tool_calls=calls,
            usage=self._usage,
            truncated=self._finish_reason == "length",
        )

    def _emit(self, kind: str, text: str) -> None:
        if self._on_delta is not None:
            self._on_delta(StreamDelta(kind, text))


def _parse_tool_call(call_id: str, name: str, raw: str) -> ToolCall:
    """Decode the JSON argument string; anything that isn't a JSON object is an arguments_error."""
    raw = raw or "{}"
    try:
        arguments = json.loads(raw)
    except json.JSONDecodeError as e:
        return ToolCall(call_id, name, {}, f"not valid JSON ({e.msg}): {raw!r}")
    if not isinstance(arguments, dict):
        return ToolCall(call_id, name, {}, f"not a JSON object: {raw!r}")
    return ToolCall(call_id, name, arguments)
