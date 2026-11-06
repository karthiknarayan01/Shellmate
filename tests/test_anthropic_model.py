"""AnthropicModel: message conversion both ways, and the request it sends. No network involved,
except the smoke test, which runs only when ANTHROPIC_API_KEY is set."""

import os
from types import SimpleNamespace as NS

import pytest

from runtime.anthropic_model import AnthropicModel, from_api_response, to_api_messages
from runtime.messages import Message, ToolCall
from runtime.model import DEFAULT_MAX_TOKENS, ToolSpec, Usage


def test_user_and_assistant_messages():
    api = to_api_messages([Message("user", "hi"), Message("assistant", "hello")])
    assert api == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
    ]


def test_tool_calls_become_tool_use_blocks():
    m = Message("assistant", "", tool_calls=(ToolCall("t1", "read_file", {"path": "a"}),))
    assert to_api_messages([m])[0]["content"] == [
        {"type": "tool_use", "id": "t1", "name": "read_file", "input": {"path": "a"}}
    ]


def test_consecutive_tool_results_share_one_user_message():
    calls = (ToolCall("t1", "x", {}), ToolCall("t2", "x", {}))
    api = to_api_messages([
        Message("user", "go"),
        Message("assistant", "", tool_calls=calls),
        Message("tool", "r1", tool_call_id="t1"),
        Message("tool", "r2", tool_call_id="t2", is_error=True),
    ])
    assert [m["role"] for m in api] == ["user", "assistant", "user"]
    assert api[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "r1", "is_error": False},
        {"type": "tool_result", "tool_use_id": "t2", "content": "r2", "is_error": True},
    ]


def test_response_is_parsed_into_text_calls_and_usage():
    response = NS(
        content=[
            NS(type="text", text="Let me look."),
            NS(type="tool_use", id="t1", name="read_file", input={"path": "a"}),
        ],
        usage=NS(input_tokens=120, output_tokens=30),
        stop_reason="tool_use",
    )
    reply = from_api_response(response)
    assert reply.text == "Let me look."
    assert reply.tool_calls == (ToolCall("t1", "read_file", {"path": "a"}),)
    assert reply.usage == Usage(120, 30)
    assert reply.truncated is False


def test_hitting_the_output_cap_marks_the_reply_truncated():
    response = NS(content=[NS(type="text", text="The answer is")],
                  usage=NS(input_tokens=1, output_tokens=16_000), stop_reason="max_tokens")
    assert from_api_response(response).truncated is True


def test_default_output_cap_is_sent(monkeypatch):
    sent = {}
    client = NS(messages=NS(create=lambda **kw: sent.update(kw) or NS(
        content=[], usage=NS(input_tokens=1, output_tokens=1), stop_reason="end_turn")))
    AnthropicModel(client=client).complete("s", [Message("user", "hi")], [])
    assert sent["max_tokens"] == DEFAULT_MAX_TOKENS


def test_request_carries_model_system_and_tools():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return NS(content=[NS(type="text", text="ok")],
                  usage=NS(input_tokens=1, output_tokens=1), stop_reason="end_turn")

    client = NS(messages=NS(create=create))
    model = AnthropicModel(model="m", max_tokens=99, client=client)
    model.complete("be brief", [Message("user", "hi")], [ToolSpec("t", "does t", {"type": "object"})])

    assert sent["model"] == "m" and sent["max_tokens"] == 99 and sent["system"] == "be brief"
    assert sent["tools"] == [{"name": "t", "description": "does t", "input_schema": {"type": "object"}}]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY")
def test_smoke_real_api_calls_a_tool():
    tool = ToolSpec(
        "get_weather", "Weather for a city.",
        {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    )
    reply = AnthropicModel().complete(
        "Always use the tool.", [Message("user", "What's the weather in Paris?")], [tool]
    )
    assert reply.tool_calls and reply.tool_calls[0].name == "get_weather"
