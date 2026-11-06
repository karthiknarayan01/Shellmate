"""Approvals in the agent loop: who gets asked, what a denial does, and that a denied tool never runs."""

import pytest

from runtime.approval import AutoApprove, Decision
from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.model import ModelReply
from runtime.session import Session
from runtime.state import project
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, call, say


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


async def test_risky_tool_runs_only_after_approval(log):
    tools, ran = _tools()
    approver = ScriptedApprover([Decision("approve")])
    session = Session.create(log, FakeModel([call("delete", path="a"), say("deleted")]), tools,
                             approver=approver)

    assert (await session.run_turn("delete a")).text == "deleted"
    assert ran == ["a"]
    assert [c.name for c in approver.asked] == ["delete"]
    assert _types(log, session) == [
        "user_message", "assistant_message",
        "approval_requested", "approval_decided", "tool_started", "tool_finished",
        "assistant_message",
    ]


async def test_safe_tool_runs_without_asking(log):
    tools, _ = _tools()
    approver = ScriptedApprover([])
    session = Session.create(log, FakeModel([call("read", path="a"), say("ok")]), tools, approver=approver)

    await session.run_turn("read a")
    assert approver.asked == []


async def test_denied_tool_never_runs_and_the_turn_stops(log):
    tools, ran = _tools()
    model = FakeModel([call("delete", path="a")])
    session = Session.create(log, model, tools, approver=ScriptedApprover([Decision("deny")]))

    result = await session.run_turn("delete a")
    assert ran == []
    assert result.denied_call.name == "delete"
    assert "tool_started" not in _types(log, session)
    assert len(model.requests) == 1  # the model was not asked again


async def test_denial_with_feedback_is_sent_to_the_model_and_the_turn_continues(log):
    tools, ran = _tools()
    model = FakeModel([call("delete", path="a"), say("ok, I'll leave it")])
    approver = ScriptedApprover([Decision("deny", feedback="keep a, it's important")])
    session = Session.create(log, model, tools, approver=approver)

    result = await session.run_turn("delete a")
    assert ran == [] and result.denied_call is None
    assert result.text == "ok, I'll leave it"
    assert "keep a, it's important" in model.requests[1].messages[-1].text


async def test_denial_skips_the_remaining_calls_in_the_same_reply(log):
    tools, ran = _tools()
    reply = ModelReply(tool_calls=(ToolCall("c1", "delete", {"path": "a"}),
                                   ToolCall("c2", "read", {"path": "b"})))
    session = Session.create(log, FakeModel([reply]), tools, approver=ScriptedApprover([Decision("deny")]))

    await session.run_turn("go")
    assert ran == []
    assert session.state.messages[-1].text == "skipped: the user denied delete"
    assert session.state.open_calls == {}


async def test_approve_for_session_stops_asking_for_that_tool(log):
    tools, ran = _tools()
    model = FakeModel([call("delete", "c1", path="a"), call("delete", "c2", path="b"), say("done")])
    approver = ScriptedApprover([Decision("approve_for_session")])
    session = Session.create(log, model, tools, approver=approver)

    await session.run_turn("delete both")
    assert ran == ["a", "b"]
    assert len(approver.asked) == 1
    sources = [e.payload["source"] for e in log.load(session.session_id) if e.type == "approval_decided"]
    assert sources == ["approver", "session"]


async def test_approve_for_session_survives_reopening(log):
    tools, _ = _tools()
    first = Session.create(log, FakeModel([call("delete", path="a"), say("ok")]), tools,
                           approver=ScriptedApprover([Decision("approve_for_session")]))
    await first.run_turn("delete a")

    approver = ScriptedApprover([])
    reopened = Session.open(log, first.session_id, FakeModel([call("delete", "c2", path="b"), say("ok")]),
                            tools, approver=approver)
    await reopened.run_turn("delete b")
    assert approver.asked == []


async def test_without_an_approver_risky_tools_are_denied(log):
    tools, ran = _tools()
    session = Session.create(log, FakeModel([call("delete", path="a")]), tools)

    assert (await session.run_turn("delete a")).denied_call is not None
    assert ran == []


async def test_auto_approve_runs_everything(log):
    tools, ran = _tools()
    session = Session.create(log, FakeModel([call("delete", path="a"), say("ok")]), tools,
                             approver=AutoApprove())
    await session.run_turn("delete a")
    assert ran == ["a"]


async def test_malformed_risky_call_is_rejected_without_asking(log):
    tools, ran = _tools()
    approver = ScriptedApprover([])
    session = Session.create(log, FakeModel([call("delete"), say("oops")]), tools, approver=approver)

    await session.run_turn("delete")
    assert approver.asked == [] and ran == []
    assert "invalid arguments" in session.state.messages[-2].text


async def test_live_state_equals_replay_after_approvals_and_denials(log):
    tools, _ = _tools()
    model = FakeModel([call("delete", "c1", path="a"), call("delete", "c2", path="b"), say("done")])
    approver = ScriptedApprover([Decision("approve"), Decision("deny", feedback="not b")])
    session = Session.create(log, model, tools, approver=approver)
    await session.run_turn("go")

    assert session.state == project(log.load(session.session_id))


# ── helpers ──────────────────────────────────────────────────────────────────────────────


class ScriptedApprover:
    """Returns prepared decisions in order and records which calls it was asked about."""

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.asked = []

    async def ask(self, call):
        self.asked.append(call)
        if not self.decisions:
            raise AssertionError(f"approver asked about {call.name} unexpectedly")
        return self.decisions.pop(0)


def _tools():
    """A safe `read` tool and a risky `delete` tool that records the paths it was run on."""
    ran = []
    schema = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
    tools = ToolRegistry([
        Tool("read", "Read.", schema, lambda path: f"contents of {path}"),
        Tool("delete", "Delete.", schema, lambda path: ran.append(path) or "deleted",
             requires_approval=True),
    ])
    return tools, ran


def _types(log, session):
    return [e.type for e in log.load(session.session_id)]
