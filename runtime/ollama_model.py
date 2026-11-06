"""ModelClient for open models running on this computer through Ollama. Default: Qwen 3.5 9B.

    ollama pull qwen3.5:9b          once: downloads the model (~6.6 GB)
    agent-runtime --provider ollama

Ollama (https://ollama.com) serves models over HTTP on localhost:11434 (OLLAMA_HOST to change
it). This adapter uses its native chat API through the `ollama` package, streamed: text and
thinking arrive in pieces, tool calls arrive whole. Nothing leaves the machine and there is no
API key or cost.

Differences from the hosted providers, handled here:
- Tool calls have no ids, so each gets one (`call_<hex>`); results are matched by those ids
  in our log and sent back to Ollama in order, with the tool's name.
- Arguments arrive as an object already, not a JSON string.
- Ollama gives a model a small context by default, so the request sets `num_ctx` explicitly;
  the same number is the window compaction works against (`context_window`).
- Qwen 3.5 thinks before it answers (Ollama's `think`); the thinking streams as reasoning and
  is kept in the log, but isn't sent back, since Qwen doesn't expect earlier thinking.
"""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import ollama

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

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_CONTEXT_WINDOW = 32_768


class OllamaModel:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        client: Any = None,
    ):
        """`client` defaults to ollama.AsyncClient(), which honours OLLAMA_HOST."""
        self._model = model
        self._max_tokens = max_tokens
        self.context_window = context_window
        self._client = client or ollama.AsyncClient()

    async def check(self) -> None:
        """Fail early, with what to do, if Ollama isn't running or the model isn't downloaded."""
        try:
            await self._client.show(self._model)
        except ollama.ResponseError as e:
            if e.status_code == 404:
                raise ModelError(
                    f"the model {self._model} isn't downloaded; run: ollama pull {self._model}"
                ) from None
            raise ModelError(f"Ollama error: {e.error}") from None
        except (httpx.ConnectError, ConnectionError):
            raise ModelError(_NOT_RUNNING) from None

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[ToolSpec],
        on_delta: DeltaSink | None = None,
    ) -> ModelReply:
        """Stream one reply from the local model; return the whole reply."""
        assembler = ChunkAssembler(on_delta)
        try:
            stream = await self._client.chat(
                model=self._model,
                messages=to_api_messages(system, messages),
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        },
                    }
                    for t in tools
                ] or None,
                stream=True,
                options={"num_ctx": self.context_window, "num_predict": self._num_predict()},
            )
            async for chunk in stream:
                assembler.add(chunk)
        except ollama.ResponseError as e:
            raise ModelError(f"Ollama error: {e.error}") from None
        except (httpx.ConnectError, ConnectionError):
            raise ModelError(_NOT_RUNNING) from None
        except httpx.HTTPError as e:
            raise ModelError(f"Ollama connection failed: {e}") from None
        return assembler.reply()

    def _num_predict(self) -> int:
        """The output cap, kept to half the window so a reply can't crowd out its own prompt."""
        return min(self._max_tokens, self.context_window // 2)


_NOT_RUNNING = "Ollama isn't running: start the Ollama app, or run `ollama serve`"


def to_api_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    """Runtime messages -> Ollama chat messages, with the system prompt first.

    Tool results carry the tool's name (Ollama has no call ids); an error result is sent with
    an "ERROR: " prefix, since there is no error flag.
    """
    names = {c.id: c.name for m in messages for c in m.tool_calls}
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": m.text}
            if m.tool_calls:
                message["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls
                ]
            out.append(message)
        else:
            content = f"ERROR: {m.text}" if m.is_error else m.text
            out.append({"role": "tool", "content": content, "tool_name": names.get(m.tool_call_id, "")})
    return out


class ChunkAssembler:
    """Collects streamed chunks into one ModelReply, reporting text and thinking as they arrive."""

    def __init__(self, on_delta: DeltaSink | None = None):
        self._on_delta = on_delta
        self._text: list[str] = []
        self._thinking: list[str] = []
        self._calls: list[ToolCall] = []
        self._done_reason: str | None = None
        self._usage = Usage()

    def add(self, chunk: Any) -> None:
        message = chunk.message
        if getattr(message, "thinking", None):
            self._thinking.append(message.thinking)
            self._emit("reasoning", message.thinking)
        if message.content:
            self._text.append(message.content)
            self._emit("text", message.content)
        for call in message.tool_calls or []:
            arguments = call.function.arguments
            self._calls.append(ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=call.function.name,
                arguments=dict(arguments) if isinstance(arguments, dict) else {},
                arguments_error=None if isinstance(arguments, dict) else f"not a JSON object: {arguments!r}",
            ))
        if chunk.done:
            self._done_reason = chunk.done_reason
            self._usage = Usage(chunk.prompt_eval_count or 0, chunk.eval_count or 0)

    def reply(self) -> ModelReply:
        """The complete reply. done_reason "length" means it stopped at the output cap."""
        return ModelReply(
            text="".join(self._text),
            reasoning="".join(self._thinking),
            tool_calls=tuple(self._calls),
            usage=self._usage,
            truncated=self._done_reason == "length",
        )

    def _emit(self, kind: str, text: str) -> None:
        if self._on_delta is not None:
            self._on_delta(StreamDelta(kind, text))
