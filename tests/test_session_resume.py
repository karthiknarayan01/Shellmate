"""Resuming a turn after a crash, from every point a crash can leave it at.

A crash is simulated by raising `Crash`, a BaseException: tool error handling only catches
Exception, so it escapes run_turn() the way a dying process would stop it, leaving the log
exactly as it was at that moment. tests/test_crash_resume.py repeats the key cases with real
SIGKILLed processes.
"""

import pytest

from runtime.approval import AutoApprove, Decision
from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.model import ModelReply
from runtime.session import INTERRUPTED_RESULT, Session, TurnInProgress
from runtime.state import ToolCallStatus, project
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, call, say


class Crash(BaseException):
    pass


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


def test_started_non_idempotent_tool_is_not_run_again(log):
    """The tool's side effect happened, then the process died before tool_finished."""
    effects = Effects(crash_in="send", crash_after_effect=True)
    session = Session.create(log, FakeModel([call("send", text="hi")]), effects.tools(),
                             approver=AutoApprove())
    with pytest.raises(Crash):
        session.run_turn("send hi")
    assert effects.ran == ["send"]

    model = FakeModel([say("I can't tell whether it was sent")])
    resumed = Session.open(log, session.session_id, model, effects.tools())
    assert resumed.needs_resume

    result = resumed.resume()
    assert effects.ran == ["send"]  # not repeated
    assert result.text == "I can't tell whether it was sent"
    tool_result = model.requests[0].messages[-1]
    assert tool_result.text == INTERRUPTED_RESULT and tool_result.is_error
    finished = [e for e in log.load(session.session_id) if e.type == "tool_finished"]
    assert finished[-1].payload["interrupted"] is True


def test_started_idempotent_tool_is_run_again(log):
    effects = Effects(crash_in="read", crash_after_effect=True)
    session = Session.create(log, FakeModel([call("read", path="a")]), effects.tools())
    with pytest.raises(Crash):
        session.run_turn("read a")

    resumed = Session.open(log, session.session_id, FakeModel([say("a says hi")]), effects.tools())
    assert resumed.resume().text == "a says hi"
    assert effects.ran == ["read", "read"]
    assert "tool_restarted" in _types(log, session)
    assert resumed.state.messages[-2].text == "contents of a"


def test_crash_while_waiting_for_approval_asks_again(log):
    effects = Effects()
    session = Session.create(log, FakeModel([call("send", text="hi")]), effects.tools(),
                             approver=CrashingApprover())
    with pytest.raises(Crash):
        session.run_turn("send hi")
    assert session.state.tool_calls["c1"].status is ToolCallStatus.AWAITING_APPROVAL

    approver = ScriptedApprover([Decision("approve")])
    resumed = Session.open(log, session.session_id, FakeModel([say("sent")]), effects.tools(),
                           approver=approver)
    assert resumed.resume().text == "sent"
    assert len(approver.asked) == 1 and effects.ran == ["send"]
    assert _types(log, session).count("approval_requested") == 1  # not requested twice


def test_approved_but_not_started_call_runs_without_asking_again(log):
    sid = log.new_session()
    log.append(sid, "user_message", {"text": "send hi"})
    log.append(sid, "assistant_message",
               {"text": "", "tool_calls": [{"id": "c1", "name": "send", "arguments": {"text": "hi"}}]})
    log.append(sid, "approval_requested", {"call_id": "c1"})
    log.append(sid, "approval_decided",
               {"call_id": "c1", "decision": "approve", "feedback": "", "source": "approver"})

    effects = Effects()
    approver = ScriptedApprover([])
    resumed = Session.open(log, sid, FakeModel([say("sent")]), effects.tools(), approver=approver)
    resumed.resume()
    assert effects.ran == ["send"] and approver.asked == []


def test_crash_during_the_next_model_call_does_not_rerun_finished_tools(log):
    effects = Effects()
    model = CrashingModel([call("send", text="hi"), CRASH])
    session = Session.create(log, model, effects.tools(), approver=ScriptedApprover([Decision("approve")]))
    with pytest.raises(Crash):
        session.run_turn("send hi")

    resumed = Session.open(log, session.session_id, FakeModel([say("sent")]), effects.tools())
    assert resumed.resume().text == "sent"
    assert effects.ran == ["send"]


def test_crash_before_the_first_model_reply_retries_the_model(log):
    session = Session.create(log, CrashingModel([CRASH]), ToolRegistry([]))
    with pytest.raises(Crash):
        session.run_turn("hi")

    model = FakeModel([say("hello")])
    resumed = Session.open(log, session.session_id, model, ToolRegistry([]))
    assert resumed.resume().text == "hello"
    assert [m.text for m in model.requests[0].messages] == ["hi"]


def test_open_calls_are_resolved_in_order_from_their_own_status(log):
    """One reply with three calls: the first finished, the second was running, the third untouched."""
    effects = Effects(crash_in="send", crash_after_effect=True)
    reply = ModelReply(tool_calls=(
        ToolCall("c1", "read", {"path": "a"}),
        ToolCall("c2", "send", {"text": "hi"}),
        ToolCall("c3", "read", {"path": "b"}),
    ))
    session = Session.create(log, FakeModel([reply]), effects.tools(),
                             approver=ScriptedApprover([Decision("approve")]))
    with pytest.raises(Crash):
        session.run_turn("go")

    resumed = Session.open(log, session.session_id, FakeModel([say("done")]), effects.tools())
    resumed.resume()
    assert effects.ran == ["read", "send", "read"]  # c1 once, c2 not repeated, c3 run now
    results = {m.tool_call_id: m.text for m in resumed.state.messages if m.role == "tool"}
    assert results == {"c1": "contents of a", "c2": INTERRUPTED_RESULT, "c3": "contents of b"}


def test_denial_during_resume_stops_the_turn(log):
    effects = Effects()
    session = Session.create(log, FakeModel([call("send", text="hi")]), effects.tools(),
                             approver=CrashingApprover())
    with pytest.raises(Crash):
        session.run_turn("send hi")

    resumed = Session.open(log, session.session_id, FakeModel([]), effects.tools(),
                           approver=ScriptedApprover([Decision("deny")]))
    result = resumed.resume()
    assert result.denied_call.name == "send" and effects.ran == []
    assert not resumed.needs_resume


def test_nothing_to_resume_after_a_completed_turn(log):
    session = Session.create(log, FakeModel([say("hello")]), ToolRegistry([]))
    session.run_turn("hi")

    reopened = Session.open(log, session.session_id, FakeModel([]), ToolRegistry([]))
    assert not reopened.needs_resume
    assert reopened.resume() is None


def test_new_turn_is_refused_until_resumed_then_allowed(log):
    session = Session.create(log, CrashingModel([CRASH]), ToolRegistry([]))
    with pytest.raises(Crash):
        session.run_turn("hi")

    resumed = Session.open(log, session.session_id, FakeModel([say("hello"), say("again")]), ToolRegistry([]))
    with pytest.raises(TurnInProgress):
        resumed.run_turn("are you there?")
    resumed.resume()
    assert resumed.run_turn("are you there?").text == "again"


def test_live_state_equals_replay_after_resume(log):
    effects = Effects(crash_in="send", crash_after_effect=True)
    session = Session.create(log, FakeModel([call("send", text="hi")]), effects.tools(),
                             approver=AutoApprove())
    with pytest.raises(Crash):
        session.run_turn("send hi")
    resumed = Session.open(log, session.session_id, FakeModel([say("ok")]), effects.tools())
    resumed.resume()

    assert resumed.state == project(log.load(session.session_id))


# ── helpers ──────────────────────────────────────────────────────────────────────────────


CRASH = object()  # put in a CrashingModel script to crash on that model call


class CrashingModel(FakeModel):
    """FakeModel that raises Crash when it reaches a CRASH entry in its script."""

    def complete(self, system, messages, tools):
        if self.replies and self.replies[0] is CRASH:
            self.replies.pop(0)
            raise Crash()
        return super().complete(system, messages, tools)


class Effects:
    """A `read` tool (idempotent) and a `send` tool (not idempotent, requires approval).

    `ran` records every run. The first run of the tool named `crash_in` raises Crash, either
    before its effect or, with crash_after_effect, after it.
    """

    def __init__(self, crash_in=None, crash_after_effect=False):
        self.ran = []
        self.crash_in = crash_in
        self.crash_after_effect = crash_after_effect
        self.crashed = False

    def tools(self):
        schema = {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}}
        return ToolRegistry([
            Tool("read", "Read.", schema, lambda path: self._run("read", f"contents of {path}"),
                 idempotent=True),
            Tool("send", "Send.", schema, lambda text: self._run("send", "sent"),
                 requires_approval=True),
        ])

    def _run(self, name, output):
        crash_now = name == self.crash_in and not self.crashed
        if crash_now and not self.crash_after_effect:
            self.crashed = True
            raise Crash()
        self.ran.append(name)
        if crash_now:
            self.crashed = True
            raise Crash()
        return output


class ScriptedApprover:
    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.asked = []

    def ask(self, call):
        self.asked.append(call)
        if not self.decisions:
            raise AssertionError(f"approver asked about {call.name} unexpectedly")
        return self.decisions.pop(0)


class CrashingApprover:
    def ask(self, call):
        raise Crash()


def _types(log, session):
    return [e.type for e in log.load(session.session_id)]
