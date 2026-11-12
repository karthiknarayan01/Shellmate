"""OpenAIModel: input items, the streamed reply, truncation, failures, malformed tool arguments.
No network, except the smoke test, which runs only when OPENAI_API_KEY is set."""

import os
from types import SimpleNamespace as NS

import openai
import pytest

from runtime.messages import Message, ToolCall
from runtime.model import ModelError, StreamDelta, ToolSpec, Usage
from runtime.openai_model import DEFAULT_MODEL, OpenAIModel, to_input_items, to_reply

CALL = Message(
    "assistant", "Let me look.", tool_calls=(ToolCall("t1", "read_file", {"path": "a"}),)
)


def test_messages_become_input_items_in_order():
    items = to_input_items(
        [Message("user", "hi"), CALL, Message("tool", "print(1)", tool_call_id="t1")]
    )
    assert items == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Let me look."},
        {
            "type": "function_call", "call_id": "t1", "name": "read_file",
            "arguments": '{"path": "a"}',
        },
        {"type": "function_call_output", "call_id": "t1", "output": "print(1)"},
    ]


def test_an_assistant_message_with_only_tool_calls_has_no_text_item():
    items = to_input_items([Message("assistant", "", tool_calls=CALL.tool_calls)])
    assert [item.get("type") for item in items] == ["function_call"]


def test_error_results_are_prefixed():
    items = to_input_items([Message("tool", "no such file", tool_call_id="t1", is_error=True)])
    assert items[0]["output"] == "ERROR: no such file"


def test_the_final_response_becomes_one_reply():
    reply = to_reply(_response(
        _reasoning("I should read it."),
        _message("Let me ", "look."),
        _call("t1", "read_file", '{"path": "a"}'),
    ))
    assert reply.text == "Let me look."
    assert reply.reasoning == "I should read it."
    assert reply.tool_calls == (ToolCall("t1", "read_file", {"path": "a"}),)
    assert reply.usage == Usage(100, 20)
    assert reply.truncated is False


def test_hitting_the_output_cap_marks_the_reply_truncated():
    reply = to_reply(_response(_message("half a"), incomplete="max_output_tokens"))
    assert reply.truncated is True


@pytest.mark.parametrize(
    "raw, error", [('{"path": ', "not valid JSON"), ("[1]", "not a JSON object")]
)
def test_bad_tool_arguments_become_arguments_error(raw, error):
    (call,) = to_reply(_response(_call("t1", "read_file", raw))).tool_calls
    assert call.arguments == {} and error in call.arguments_error


async def test_request_is_stateless_and_streams_text_and_reasoning_as_they_arrive():
    seen, deltas = {}, []

    async def create(**kwargs):
        seen.update(kwargs)
        return _stream(
            NS(type="response.reasoning_summary_text.delta", delta="Thinking"),
            NS(type="response.output_text.delta", delta="Hel"),
            NS(type="response.output_text.delta", delta="lo"),
            NS(type="response.completed", response=_response(_message("Hello"))),
        )

    model = OpenAIModel(model="m", max_tokens=500, client=NS(responses=NS(create=create)))
    tool = ToolSpec("shell", "run a command", {"type": "object"})
    reply = await model.complete("be brief", [Message("user", "hi")], [tool], deltas.append)
    assert reply.text == "Hello"
    assert deltas == [
        StreamDelta("reasoning", "Thinking"), StreamDelta("text", "Hel"), StreamDelta("text", "lo")
    ]
    assert (seen["model"], seen["instructions"]) == ("m", "be brief")
    assert seen["max_output_tokens"] == 500
    assert seen["store"] is False and seen["stream"] is True
    assert seen["tools"] == [
        {
            "type": "function", "name": "shell", "description": "run a command",
            "parameters": {"type": "object"},
        }
    ]


async def test_no_tools_means_no_tools_field():
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return _stream(NS(type="response.completed", response=_response(_message("ok"))))

    await _model(create).complete("s", [Message("user", "hi")], [])
    assert "tools" not in seen


@pytest.mark.parametrize(
    "events, message",
    [
        ([NS(type="response.failed", response=NS(error=NS(message="overloaded")))], "overloaded"),
        ([NS(type="error", message="bad request")], "bad request"),
        ([NS(type="response.output_text.delta", delta="Hel")], "ended without a response"),
    ],
)
async def test_a_failed_or_unfinished_stream_is_a_model_error(events, message):
    async def create(**kwargs):
        return _stream(*events)

    with pytest.raises(ModelError, match=message):
        await _model(create).complete("s", [Message("user", "hi")], [])


async def test_api_failures_become_model_errors():
    async def create(**kwargs):
        raise openai.APIConnectionError(request=None)

    with pytest.raises(ModelError, match="OpenAI API error"):
        await _model(create).complete("s", [Message("user", "hi")], [])


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY is not set"):
        OpenAIModel()


def test_default_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert OpenAIModel()._model == DEFAULT_MODEL


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
async def test_smoke_full_tool_round_trip():
    model = OpenAIModel(max_tokens=2_000)
    tool = ToolSpec("add", "Add two integers.", {
        "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    })
    first = await model.complete("Use the add tool.", [Message("user", "What is 2 + 3?")], [tool])
    (call,) = first.tool_calls
    result = Message("tool", "5", tool_call_id=call.id)
    second = await model.complete(
        "Use the add tool.",
        [
            Message("user", "What is 2 + 3?"),
            Message("assistant", first.text, tool_calls=first.tool_calls),
            result,
        ],
        [tool],
    )
    assert "5" in second.text


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _model(create):
    return OpenAIModel(client=NS(responses=NS(create=create)))


def _response(*output, incomplete=None):
    return NS(
        output=list(output),
        usage=NS(input_tokens=100, output_tokens=20),
        incomplete_details=NS(reason=incomplete) if incomplete else None,
    )


def _message(*texts):
    return NS(type="message", content=[NS(type="output_text", text=t) for t in texts])


def _reasoning(*summaries):
    return NS(type="reasoning", summary=[NS(type="summary_text", text=s) for s in summaries])


def _call(call_id, name, arguments):
    return NS(type="function_call", call_id=call_id, name=name, arguments=arguments)


async def _stream(*events):
    for event in events:
        yield event
