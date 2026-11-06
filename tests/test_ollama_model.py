"""OllamaModel against a fake ollama.AsyncClient: request shape, streaming, tool calls, errors."""

import httpx
import ollama
import pytest

from runtime.messages import Message, ToolCall
from runtime.model import ModelError, ToolSpec
from runtime.ollama_model import DEFAULT_CONTEXT_WINDOW, DEFAULT_MODEL, OllamaModel, to_api_messages


def chunk(content="", thinking=None, tool_calls=None, done=False, **final):
    return ollama.ChatResponse(
        model=DEFAULT_MODEL,
        message=ollama.Message(role="assistant", content=content, thinking=thinking, tool_calls=tool_calls),
        done=done,
        **final,
    )


def tool_call(name, arguments):
    return ollama.Message.ToolCall(function=ollama.Message.ToolCall.Function(name=name, arguments=arguments))


class FakeClient:
    def __init__(self, chunks=(), error=None, show_error=None):
        self.chunks = list(chunks)
        self.error = error
        self.show_error = show_error
        self.requests = []

    async def chat(self, **request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error

        async def stream():
            for c in self.chunks:
                yield c

        return stream()

    async def show(self, model):
        if self.show_error is not None:
            raise self.show_error
        return {}


SPEC = ToolSpec("read_file", "Read a file.", {"type": "object", "properties": {"path": {"type": "string"}}})


async def test_text_and_thinking_stream_as_deltas_and_make_one_reply():
    client = FakeClient([
        chunk(thinking="Let me "),
        chunk(thinking="think."),
        chunk(content="Hello"),
        chunk(content=" there"),
        chunk(done=True, done_reason="stop", prompt_eval_count=120, eval_count=8),
    ])
    deltas = []
    reply = await OllamaModel(client=client).complete("sys", [Message("user", "hi")], [], on_delta=deltas.append)

    assert [(d.kind, d.text) for d in deltas] == [
        ("reasoning", "Let me "), ("reasoning", "think."), ("text", "Hello"), ("text", " there"),
    ]
    assert reply.text == "Hello there"
    assert reply.reasoning == "Let me think."
    assert (reply.usage.input_tokens, reply.usage.output_tokens) == (120, 8)
    assert not reply.truncated


async def test_request_sets_the_window_the_output_cap_and_the_tools():
    client = FakeClient([chunk(done=True, done_reason="stop")])
    await OllamaModel(max_tokens=4000, context_window=16384, client=client).complete("sys", [Message("user", "hi")], [SPEC])

    request = client.requests[0]
    assert request["model"] == DEFAULT_MODEL
    assert request["stream"] is True
    assert request["options"] == {"num_ctx": 16384, "num_predict": 4000}
    assert request["tools"] == [{
        "type": "function",
        "function": {"name": "read_file", "description": "Read a file.", "parameters": SPEC.input_schema},
    }]
    assert request["messages"][0] == {"role": "system", "content": "sys"}


async def test_output_cap_never_exceeds_half_the_window():
    client = FakeClient([chunk(done=True, done_reason="stop")])
    await OllamaModel(max_tokens=16000, context_window=8192, client=client).complete("sys", [], [])
    assert client.requests[0]["options"]["num_predict"] == 4096


async def test_no_tools_are_sent_as_none():
    client = FakeClient([chunk(done=True, done_reason="stop")])
    await OllamaModel(client=client).complete("sys", [], [])
    assert client.requests[0]["tools"] is None


async def test_tool_calls_get_unique_ids_and_their_arguments():
    client = FakeClient([
        chunk(tool_calls=[tool_call("read_file", {"path": "a.txt"}), tool_call("read_file", {"path": "b.txt"})]),
        chunk(done=True, done_reason="stop"),
    ])
    reply = await OllamaModel(client=client).complete("sys", [Message("user", "read both")], [SPEC])

    assert [c.arguments for c in reply.tool_calls] == [{"path": "a.txt"}, {"path": "b.txt"}]
    ids = [c.id for c in reply.tool_calls]
    assert len(set(ids)) == 2 and all(i.startswith("call_") for i in ids)
    assert all(c.arguments_error is None for c in reply.tool_calls)


async def test_stopping_at_the_output_cap_marks_the_reply_truncated():
    client = FakeClient([chunk(content="cut of"), chunk(done=True, done_reason="length")])
    reply = await OllamaModel(client=client).complete("sys", [], [])
    assert reply.truncated


def test_tool_results_carry_the_tool_name_and_errors_are_marked():
    messages = [
        Message("user", "read it"),
        Message("assistant", "", tool_calls=(ToolCall("call_1", "read_file", {"path": "a"}),), reasoning="hmm"),
        Message("tool", "contents", tool_call_id="call_1"),
        Message("assistant", "", tool_calls=(ToolCall("call_2", "read_file", {"path": "b"}),)),
        Message("tool", "no such file", tool_call_id="call_2", is_error=True),
    ]
    assert to_api_messages("sys", messages)[1:] == [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "a"}}}]},
        {"role": "tool", "content": "contents", "tool_name": "read_file"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read_file", "arguments": {"path": "b"}}}]},
        {"role": "tool", "content": "ERROR: no such file", "tool_name": "read_file"},
    ]


async def test_server_errors_become_model_errors():
    client = FakeClient(error=ollama.ResponseError("model requires more system memory", 500))
    with pytest.raises(ModelError, match="Ollama error: model requires more system memory"):
        await OllamaModel(client=client).complete("sys", [], [])


async def test_connection_refused_says_to_start_ollama():
    client = FakeClient(error=httpx.ConnectError("refused"))
    with pytest.raises(ModelError, match="Ollama isn't running"):
        await OllamaModel(client=client).complete("sys", [], [])


async def test_check_passes_when_the_model_is_there():
    await OllamaModel(client=FakeClient()).check()


async def test_check_says_how_to_download_a_missing_model():
    client = FakeClient(show_error=ollama.ResponseError("model 'qwen3.5:9b' not found", 404))
    with pytest.raises(ModelError, match="run: ollama pull qwen3.5:9b"):
        await OllamaModel(client=client).check()


async def test_check_says_to_start_ollama_when_it_is_not_running():
    client = FakeClient(show_error=ConnectionError("Failed to connect to Ollama"))
    with pytest.raises(ModelError, match="Ollama isn't running"):
        await OllamaModel(client=client).check()


def test_defaults():
    model = OllamaModel(client=FakeClient())
    assert model._model == "qwen3.5:9b"
    assert model.context_window == DEFAULT_CONTEXT_WINDOW
