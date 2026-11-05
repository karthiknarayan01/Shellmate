"""The agent loop: turns, tool steps, step limit, and that the log always replays to the live state."""

import pytest

from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.model import ModelReply
from runtime.session import MaxStepsReached, OpenToolCalls, Session
from runtime.state import project
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, call, say


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


def test_turn_without_tools_returns_the_reply(log):
    model = FakeModel([say("hello")])
    session = Session.create(log, model, ToolRegistry([]))

    assert session.run_turn("hi") == "hello"
    assert [m.role for m in session.state.messages] == ["user", "assistant"]


def test_tool_result_is_sent_back_to_the_model(log):
    model = FakeModel([call("echo", text="ping"), say("got pong")])
    session = Session.create(log, model, ToolRegistry([_echo()]))

    assert session.run_turn("go") == "got pong"
    second_request = model.requests[1].messages
    assert [m.role for m in second_request] == ["user", "assistant", "tool"]
    assert second_request[-1].text == "pong:ping"


def test_several_calls_in_one_reply_run_in_order(log):
    ran = []
    tool = Tool("rec", "", {"type": "object", "properties": {"n": {"type": "integer"}}},
                lambda n: ran.append(n) or "ok")
    reply = ModelReply(tool_calls=(ToolCall("a", "rec", {"n": 1}), ToolCall("b", "rec", {"n": 2})))
    session = Session.create(log, FakeModel([reply, say("done")]), ToolRegistry([tool]))

    session.run_turn("go")
    assert ran == [1, 2]


def test_tool_errors_go_back_to_the_model_and_the_turn_continues(log):
    model = FakeModel([call("missing_tool"), say("sorry")])
    session = Session.create(log, model, ToolRegistry([]))

    assert session.run_turn("go") == "sorry"
    result = model.requests[1].messages[-1]
    assert result.is_error and "unknown tool" in result.text


def test_reasoning_is_kept_for_the_next_request(log):
    thinking_call = ModelReply(reasoning="need the tool", tool_calls=(ToolCall("c1", "echo", {"text": "x"}),))
    model = FakeModel([thinking_call, say("done")])
    Session.create(log, model, ToolRegistry([_echo()])).run_turn("go")
    assert model.requests[1].messages[1].reasoning == "need the tool"


def test_tools_are_offered_to_the_model(log):
    model = FakeModel([say("ok")])
    Session.create(log, model, ToolRegistry([_echo()])).run_turn("hi")
    assert [t.name for t in model.requests[0].tools] == ["echo"]


def test_step_limit_stops_a_model_that_never_finishes(log):
    model = FakeModel([call("echo", call_id=f"c{i}", text="x") for i in range(3)])
    session = Session.create(log, model, ToolRegistry([_echo()]), max_steps=3)

    with pytest.raises(MaxStepsReached):
        session.run_turn("go")
    assert session.state.open_calls == {}  # every call still got its result


def test_live_state_always_equals_a_replay_of_the_log(log):
    model = FakeModel([call("echo", text="a"), call("echo", call_id="c2", text="b"), say("done")])
    session = Session.create(log, model, ToolRegistry([_echo()]))
    session.run_turn("go")

    assert session.state == project(log.load(session.session_id))


def test_open_continues_a_session_from_the_log(log):
    first = Session.create(log, FakeModel([say("one")]), ToolRegistry([]))
    first.run_turn("first")

    model = FakeModel([say("two")])
    reopened = Session.open(log, first.session_id, model, ToolRegistry([]))
    assert reopened.run_turn("second") == "two"
    assert [m.text for m in model.requests[0].messages] == ["first", "one", "second"]


def test_a_session_left_with_open_tool_calls_refuses_a_new_turn(log):
    sid = log.new_session()
    log.append(sid, "user_message", {"text": "go"})
    log.append(sid, "assistant_message",
               {"text": "", "tool_calls": [{"id": "c1", "name": "echo", "arguments": {}}]})

    session = Session.open(log, sid, FakeModel([]), ToolRegistry([_echo()]))
    with pytest.raises(OpenToolCalls):
        session.run_turn("hello?")
    assert len(log.load(sid)) == 2  # nothing was written


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _echo():
    return Tool(
        "echo",
        "Echo the text back.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        lambda text: f"pong:{text}",
    )
