"""ModelClient for OpenAI's models, through the Responses API.

OpenAI's newest models take tool calls only through the Responses API, not Chat Completions, so
this client uses it for every model. How it differs from Chat Completions:
- The system prompt goes in `instructions`, and the conversation is a list of input items. A
  tool call is its own `function_call` item and its result a `function_call_output` item, joined
  by `call_id`; there is no error flag, so an error result is sent with an "ERROR: " prefix.
- Requests are stateless (`store=False`): the whole conversation is sent each time and nothing
  is kept on OpenAI's side. A reasoning model's hidden reasoning isn't carried from one reply to
  the next; its reasoning summary, when there is one, is shown and recorded like DeepSeek's.
- Tool arguments arrive as a JSON string, which the model sometimes gets wrong. A string that
  isn't a JSON object becomes `arguments_error` on the ToolCall, as in deepseek_model.py.

Replies are streamed. Text and reasoning-summary deltas are passed on for display as they come;
the reply itself is read from the final response the stream ends with. A reply cut off at
`max_output_tokens` comes back as an incomplete response, and is marked truncated.

Retries on 429, 5xx and connection errors are done by the SDK (`max_retries` attempts) while
the stream is being opened; a stream that breaks or fails after it started raises ModelError.
"""

from __future__ import annotations

import json
import os
from typing import Any

import openai

from runtime.deepseek_model import _parse_tool_call
from runtime.messages import Message
from runtime.model import (
    DEFAULT_MAX_TOKENS,
    DeltaSink,
    ModelError,
    ModelReply,
    StreamDelta,
    ToolSpec,
    Usage,
)

DEFAULT_MODEL = "gpt-6.1-sol"
DELTAS = {  # stream events passed on for display, and what kind of piece each carries
    "response.output_text.delta": "text",
    "response.reasoning_summary_text.delta": "reasoning",
}
FINAL = {"response.completed", "response.incomplete"}


class OpenAIModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_retries: int = 3,
        client: openai.AsyncOpenAI | None = None,
    ):
        """`client` defaults to one keyed by OPENAI_API_KEY."""
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or openai.AsyncOpenAI(
            api_key=_require_api_key(), max_retries=max_retries
        )

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: DeltaSink | None = None,
    ) -> ModelReply:
        """Stream one reply, passing text and reasoning to `on_delta` as they arrive; return the
        whole reply. API failures, including a stream that breaks or fails, raise ModelError."""
        request: dict[str, Any] = {
            "model": self._model,
            "instructions": system,
            "input": to_input_items(messages),
            "max_output_tokens": self._max_tokens,
            "store": False,
            "stream": True,
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                }
                for t in tools
            ]
        final = None
        try:
            async for event in await self._client.responses.create(**request):
                kind = getattr(event, "type", "")
                if kind in DELTAS and on_delta is not None:
                    on_delta(StreamDelta(DELTAS[kind], event.delta))
                elif kind in FINAL:
                    final = event.response
                elif kind == "response.failed":
                    raise ModelError(f"OpenAI API error: {_failure(event.response)}")
                elif kind == "error":
                    raise ModelError(f"OpenAI API error: {getattr(event, 'message', event)}")
        except openai.APIError as e:
            raise ModelError(f"OpenAI API error: {e}") from e
        if final is None:
            raise ModelError("OpenAI API error: the stream ended without a response")
        return to_reply(final)


def _require_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return key


def to_input_items(messages: list[Message]) -> list[dict[str, Any]]:
    """Runtime messages -> Responses API input items, in order. An assistant message becomes its
    text (if any) followed by one function_call item per tool call."""
    items: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "user":
            items.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            if m.text:
                items.append({"role": "assistant", "content": m.text})
            items += [
                {
                    "type": "function_call",
                    "call_id": c.id,
                    "name": c.name,
                    "arguments": json.dumps(c.arguments),
                }
                for c in m.tool_calls
            ]
        else:
            output = f"ERROR: {m.text}" if m.is_error else m.text
            items.append(
                {"type": "function_call_output", "call_id": m.tool_call_id, "output": output}
            )
    return items


def to_reply(response: Any) -> ModelReply:
    """The final response -> one ModelReply: text, reasoning summaries, tool calls and usage."""
    text, reasoning, calls = [], [], []
    for item in response.output or []:
        if item.type == "message":
            text += [p.text for p in item.content if getattr(p, "type", "") == "output_text"]
        elif item.type == "reasoning":
            reasoning += [part.text for part in getattr(item, "summary", None) or []]
        elif item.type == "function_call":
            calls.append(_parse_tool_call(item.call_id, item.name, item.arguments))
    usage = getattr(response, "usage", None)
    details = getattr(response, "incomplete_details", None)
    return ModelReply(
        text="".join(text),
        reasoning="\n".join(reasoning),
        tool_calls=tuple(calls),
        usage=Usage(usage.input_tokens, usage.output_tokens) if usage else Usage(),
        truncated=getattr(details, "reason", None) == "max_output_tokens",
    )


def _failure(response: Any) -> str:
    error = getattr(response, "error", None)
    return getattr(error, "message", None) or "the response failed"

