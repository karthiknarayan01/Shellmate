"""DeepSeekModel: message conversion both ways, reasoning pass-back, malformed tool arguments.
No network, except the smoke test, which runs only when DEEPSEEK_API_KEY is set."""

import os
from types import SimpleNamespace as NS

import openai
import pytest

from runtime.deepseek_model import BASE_URL, DeepSeekModel, from_api_response, to_api_messages
from runtime.messages import Message, ToolCall
from runtime.model import DEFAULT_MAX_TOKENS, ModelError, ToolSpec, Usage

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


def test_response_is_parsed_into_text_reasoning_calls_and_usage():
    reply = from_api_response(_response(
        content="Let me look.",
        reasoning_content="thinking...",
        tool_calls=[_api_call("t1", "read_file", '{"path": "a"}')],
    ))
    assert reply.text == "Let me look."
    assert reply.reasoning == "thinking..."
    assert reply.tool_calls == (ToolCall("t1", "read_file", {"path": "a"}),)
    assert reply.usage == Usage(100, 20)
    assert reply.truncated is False


def test_hitting_the_output_cap_marks_the_reply_truncated():
    assert from_api_response(_response(content="The answer is", finish_reason="length")).truncated


@pytest.mark.parametrize(
    "raw, error",
    [('{"path": ', "not valid JSON"), ('["a"]', "not a JSON object")],
    ids=["truncated", "not-an-object"],
)
def test_bad_tool_arguments_become_arguments_error(raw, error):
    (call,) = from_api_response(_response(tool_calls=[_api_call("t1", "read_file", raw)])).tool_calls
    assert call.arguments == {} and error in call.arguments_error


def test_request_omits_tools_when_there_are_none_and_includes_them_otherwise():
    sent = []
    client = NS(chat=NS(completions=NS(create=lambda **kw: sent.append(kw) or _response(content="ok"))))
    model = DeepSeekModel(model="m", client=client)

    model.complete("s", [Message("user", "hi")], [])
    model.complete("s", [Message("user", "hi")], [ToolSpec("t", "does t", {"type": "object"})])

    assert "tools" not in sent[0]
    assert sent[1]["tools"] == [
        {"type": "function", "function": {"name": "t", "description": "does t", "parameters": {"type": "object"}}}
    ]
    assert sent[1]["model"] == "m"
    assert sent[1]["max_tokens"] == DEFAULT_MAX_TOKENS


def test_api_failures_become_model_errors():
    def create(**kwargs):
        raise openai.APIConnectionError(request=None)

    model = DeepSeekModel(client=NS(chat=NS(completions=NS(create=create))))
    with pytest.raises(ModelError, match="DeepSeek API error"):
        model.complete("s", [Message("user", "hi")], [])


def test_default_client_targets_deepseek(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert str(DeepSeekModel()._client.base_url).rstrip("/") == BASE_URL


def test_missing_api_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        DeepSeekModel()


@pytest.mark.skipif(not os.environ.get("DEEPSEEK_API_KEY"), reason="needs DEEPSEEK_API_KEY")
def test_smoke_full_tool_round_trip():
    """Two requests: the second carries the first reply's reasoning, which DeepSeek requires."""
    model = DeepSeekModel()
    tool = ToolSpec(
        "get_weather", "Weather for a city.",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    history = [Message("user", "What's the weather in Paris? Use the tool.")]
    first = model.complete("Use tools when they help.", history, [tool])
    assert first.tool_calls

    history += [
        Message("assistant", first.text, reasoning=first.reasoning, tool_calls=first.tool_calls),
        Message("tool", "18°C, sunny", tool_call_id=first.tool_calls[0].id),
    ]
    second = model.complete("Use tools when they help.", history, [tool])
    assert "18" in second.text


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _response(content=None, tool_calls=None, finish_reason="stop", **extra):
    message = NS(content=content, tool_calls=tool_calls, **extra)
    return NS(
        choices=[NS(message=message, finish_reason=finish_reason)],
        usage=NS(prompt_tokens=100, completion_tokens=20),
    )


def _api_call(call_id, name, arguments):
    return NS(id=call_id, function=NS(name=name, arguments=arguments))
