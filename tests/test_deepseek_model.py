"""DeepSeekModel: message conversion, assembling a streamed reply, reasoning pass-back, malformed
tool arguments. No network, except the smoke test, which runs only when DEEPSEEK_API_KEY is set."""

import os
from types import SimpleNamespace as NS

import openai
import pytest

from runtime.deepseek_model import BASE_URL, DeepSeekModel, StreamAssembler, to_api_messages
from runtime.messages import Message, ToolCall
from runtime.model import DEFAULT_MAX_TOKENS, ModelError, StreamDelta, ToolSpec, Usage

THINKING_CALL = Message(
    "assistant", "", reasoning="I should read it.",
    tool_calls=(ToolCall("t1", "read_file", {"path": "a"}),),
)


def test_system_prompt_comes_first_and_roles_map_one_to_one():
    api = to_api_messages(
        "be brief",
        [THINKING_CALL, Message("tool", "print(1)", tool_call_id="t1"), Message("user", "thanks")],
    )
    assert [m["role"] for m in api] == ["system", "assistant", "tool", "user"]
    assert api[0] == {"role": "system", "content": "be brief"}
    assert api[1]["content"] is None
    assert api[1]["tool_calls"] == [
        {"id": "t1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a"}'}}
    ]
    assert api[2] == {"role": "tool", "tool_call_id": "t1", "content": "print(1)"}


def test_reasoning_is_sent_back():
    assert to_api_messages("s", [THINKING_CALL])[1]["reasoning_content"] == "I should read it."


def test_error_results_are_prefixed():
    api = to_api_messages("s", [Message("tool", "no such file", tool_call_id="t1", is_error=True)])
    assert api[1]["content"] == "ERROR: no such file"


def test_streamed_chunks_assemble_into_one_reply():
    reply = _assemble(
        _chunk(reasoning_content="I should "),
        _chunk(reasoning_content="look."),
        _chunk(content="Let me "),
        _chunk(content="look."),
        _chunk(tool_calls=[_piece(0, id="t1", name="read_file", arguments='{"pa')]),
        _chunk(tool_calls=[_piece(0, arguments='th": "a"}')]),
        _chunk(finish_reason="tool_calls"),
        _usage_chunk(100, 20),
    )
    assert reply.reasoning == "I should look."
    assert reply.text == "Let me look."
    assert reply.tool_calls == (ToolCall("t1", "read_file", {"path": "a"}),)
    assert reply.usage == Usage(100, 20)
    assert reply.truncated is False


def test_parallel_tool_calls_are_kept_apart_and_in_index_order():
    reply = _assemble(
        _chunk(tool_calls=[_piece(1, id="t2", name="b", arguments="{}")]),
        _chunk(tool_calls=[_piece(0, id="t1", name="a", arguments="{}")]),
    )
    assert [c.id for c in reply.tool_calls] == ["t1", "t2"]


def test_text_and_reasoning_are_reported_as_they_arrive():
    deltas = []
    assembler = StreamAssembler(deltas.append)
    assembler.add(_chunk(reasoning_content="hmm"))
    assembler.add(_chunk(content="Hel"))
    assembler.add(_chunk(content="lo"))
    assert deltas == [StreamDelta("reasoning", "hmm"), StreamDelta("text", "Hel"), StreamDelta("text", "lo")]


def test_hitting_the_output_cap_marks_the_reply_truncated():
    assert _assemble(_chunk(content="The answer is"), _chunk(finish_reason="length")).truncated


@pytest.mark.parametrize(
    "raw, error",
    [('{"path": ', "not valid JSON"), ('["a"]', "not a JSON object")],
    ids=["truncated", "not-an-object"],
)
def test_bad_tool_arguments_become_arguments_error(raw, error):
    (call,) = _assemble(_chunk(tool_calls=[_piece(0, id="t1", name="read_file", arguments=raw)])).tool_calls
    assert call.arguments == {} and error in call.arguments_error


async def test_request_streams_and_omits_tools_when_there_are_none():
    sent = []

    async def create(**kwargs):
        sent.append(kwargs)
        return _stream(_chunk(content="ok"), _usage_chunk(1, 1))

    model = DeepSeekModel(model="m", client=NS(chat=NS(completions=NS(create=create))))
    deltas = []
    reply = await model.complete("s", [Message("user", "hi")], [], on_delta=deltas.append)
    await model.complete("s", [Message("user", "hi")], [ToolSpec("t", "does t", {"type": "object"})])

    assert reply.text == "ok" and deltas == [StreamDelta("text", "ok")]
    assert sent[0]["stream"] is True and sent[0]["stream_options"] == {"include_usage": True}
    assert "tools" not in sent[0]
    assert sent[1]["tools"] == [
        {"type": "function", "function": {"name": "t", "description": "does t", "parameters": {"type": "object"}}}
    ]
    assert sent[1]["model"] == "m"
    assert sent[1]["max_tokens"] == DEFAULT_MAX_TOKENS


async def test_a_stream_that_breaks_halfway_is_a_model_error():
    async def broken():
        yield _chunk(content="Hel")
        raise openai.APIConnectionError(request=None)

    async def create(**kwargs):
        return broken()

    model = DeepSeekModel(client=NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(ModelError, match="DeepSeek API error"):
        await model.complete("s", [Message("user", "hi")], [])


async def test_api_failures_become_model_errors():
    async def create(**kwargs):
        raise openai.APIConnectionError(request=None)

    model = DeepSeekModel(client=NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(ModelError, match="DeepSeek API error"):
        await model.complete("s", [Message("user", "hi")], [])


def test_default_client_targets_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert str(DeepSeekModel()._client.base_url).rstrip("/") == BASE_URL


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekModel()


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="needs DEEPSEEK_API_KEY")
async def test_smoke_full_tool_round_trip():
    """Two requests: the second carries the first reply's reasoning, which DeepSeek requires."""
    model = DeepSeekModel()
    tool = ToolSpec(
        "get_weather", "Weather for a city.",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    history = [Message("user", "What's the weather in Paris? Use the tool.")]
    first = await model.complete("Use tools when they help.", history, [tool])
    assert first.tool_calls

    history += [
        Message("assistant", first.text, reasoning=first.reasoning, tool_calls=first.tool_calls),
        Message("tool", "18°C, sunny", tool_call_id=first.tool_calls[0].id),
    ]
    second = await model.complete("Use tools when they help.", history, [tool])
    assert "18" in second.text


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _assemble(*chunks):
    assembler = StreamAssembler()
    for chunk in chunks:
        assembler.add(chunk)
    return assembler.reply()


def _chunk(content=None, tool_calls=None, finish_reason=None, **extra):
    delta = NS(content=content, tool_calls=tool_calls, **extra)
    return NS(choices=[NS(delta=delta, finish_reason=finish_reason)], usage=None)


def _usage_chunk(prompt_tokens, completion_tokens):
    return NS(choices=[], usage=NS(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens))


def _piece(index, id=None, name=None, arguments=None):
    return NS(index=index, id=id, function=NS(name=name, arguments=arguments))


async def _stream(*chunks):
    for chunk in chunks:
        yield chunk
