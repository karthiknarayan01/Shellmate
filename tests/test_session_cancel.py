"""Session.cancel(), which ends an interrupted turn without running anything more."""

import pytest

from runtime.approval import AutoApprove
from runtime.events import EventLog
from runtime.session import INTERRUPTED_RESULT, Session
from runtime.state import project
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, call, say


class Stop(BaseException):
    pass


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


async def test_cancel_during_a_tool_gives_it_the_interrupted_result(log):
    session = Session.create(log, FakeModel([call("slow")]), _tools(stop_in="slow"), approver=AutoApprove())
    with pytest.raises(Stop):
        await session.run_turn("go")

    session.cancel()
    assert not session.needs_resume
    assert session.state.messages[-1].text == INTERRUPTED_RESULT
    assert _types(log, session)[-1] == "turn_stopped"


async def test_cancel_while_waiting_for_approval_skips_the_call(log):
    class StopApprover:
        async def ask(self, call):
            raise Stop()

    ran = []
    tools = ToolRegistry([Tool("send", "", {"type": "object"}, lambda: ran.append(1) or "ok",
                               requires_approval=True)])
    session = Session.create(log, FakeModel([call("send")]), tools, approver=StopApprover())
    with pytest.raises(Stop):
        await session.run_turn("go")

    session.cancel()
    assert ran == []
    assert session.state.messages[-1].text == "cancelled by the user"


async def test_after_cancel_a_new_turn_works(log):
    session = Session.create(log, FakeModel([call("slow"), say("fresh start")]), _tools(stop_in="slow"))
    with pytest.raises(Stop):
        await session.run_turn("go")
    session.cancel()

    assert (await session.run_turn("something else")).text == "fresh start"
    assert session.state == project(log.load(session.session_id))


async def test_cancel_with_no_open_turn_does_nothing(log):
    session = Session.create(log, FakeModel([say("hi")]), _tools())
    await session.run_turn("hello")
    before = _types(log, session)
    session.cancel()
    assert _types(log, session) == before


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _tools(stop_in=None):
    def slow():
        if stop_in == "slow":
            raise Stop()
        return "done"

    return ToolRegistry([Tool("slow", "", {"type": "object"}, slow)])


def _types(log, session):
    return [e.type for e in log.load(session.session_id)]
