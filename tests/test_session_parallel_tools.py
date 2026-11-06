"""Tool calls from one reply run at the same time; approvals are asked first, one by one."""

import asyncio
import threading
import time

import pytest

from runtime.approval import AutoApprove, Decision
from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.model import ModelReply
from runtime.session import Session
from runtime.state import project
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, say


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


async def test_async_tools_from_one_reply_run_concurrently(log):
    """Each call waits for the other to start; run one after the other, this would never finish."""
    started = {"a": asyncio.Event(), "b": asyncio.Event()}

    async def meet(me, other):
        started[me].set()
        await asyncio.wait_for(started[other].wait(), timeout=5)
        return f"{me} met {other}"

    tools = ToolRegistry([Tool("meet", "", {"type": "object"}, meet)])
    reply = ModelReply(tool_calls=(ToolCall("c1", "meet", {"me": "a", "other": "b"}),
                                   ToolCall("c2", "meet", {"me": "b", "other": "a"})))
    session = Session.create(log, FakeModel([reply, say("done")]), tools)

    assert (await session.run_turn("go")).text == "done"
    results = {m.tool_call_id: m.text for m in session.state.messages if m.role == "tool"}
    assert results == {"c1": "a met b", "c2": "b met a"}


async def test_plain_function_tools_run_concurrently_in_threads(log):
    barrier = threading.Barrier(3, timeout=5)  # all three must be running at once
    tools = ToolRegistry([Tool("wait", "", {"type": "object"}, lambda: barrier.wait() or "ok")])
    reply = ModelReply(tool_calls=tuple(ToolCall(f"c{i}", "wait", {}) for i in range(3)))
    session = Session.create(log, FakeModel([reply, say("done")]), tools)

    assert (await session.run_turn("go")).text == "done"
    assert all(not m.is_error for m in session.state.messages if m.role == "tool")


async def test_parallel_calls_are_faster_than_sequential(log):
    tools = ToolRegistry([Tool("slow", "", {"type": "object"}, lambda: time.sleep(0.3) or "ok")])
    reply = ModelReply(tool_calls=tuple(ToolCall(f"c{i}", "slow", {}) for i in range(4)))
    session = Session.create(log, FakeModel([reply, say("done")]), tools)

    started = time.monotonic()
    await session.run_turn("go")
    assert time.monotonic() - started < 0.9  # sequential would take 1.2s


async def test_at_most_max_parallel_tools_run_at_once(log):
    running = 0
    peak = 0

    async def track():
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.02)
        running -= 1
        return "ok"

    tools = ToolRegistry([Tool("track", "", {"type": "object"}, track)])
    reply = ModelReply(tool_calls=tuple(ToolCall(f"c{i}", "track", {}) for i in range(10)))
    session = Session.create(log, FakeModel([reply, say("done")]), tools, max_parallel_tools=3)

    await session.run_turn("go")
    assert peak == 3


async def test_all_approvals_are_asked_before_any_call_runs(log):
    ran = []
    approver = RecordingApprover(ran, [Decision("approve"), Decision("approve")])
    tools = ToolRegistry([Tool("send", "", {"type": "object", "properties": {"n": {"type": "integer"}}},
                               lambda n: ran.append(n) or "sent", requires_approval=True)])
    reply = ModelReply(tool_calls=(ToolCall("c1", "send", {"n": 1}), ToolCall("c2", "send", {"n": 2})))
    session = Session.create(log, FakeModel([reply, say("done")]), tools, approver=approver)

    await session.run_turn("go")
    assert approver.ran_before_each_ask == [[], []]  # nothing had run when either was asked
    assert sorted(ran) == [1, 2]


async def test_a_denial_without_feedback_stops_calls_approved_earlier_in_the_reply(log):
    ran = []
    approver = RecordingApprover(ran, [Decision("approve"), Decision("deny")])
    tools = ToolRegistry([Tool("send", "", {"type": "object", "properties": {"n": {"type": "integer"}}},
                               lambda n: ran.append(n) or "sent", requires_approval=True)])
    reply = ModelReply(tool_calls=(ToolCall("c1", "send", {"n": 1}), ToolCall("c2", "send", {"n": 2})))
    session = Session.create(log, FakeModel([reply]), tools, approver=approver)

    result = await session.run_turn("go")
    assert ran == []
    assert result.denied_call.id == "c2"
    results = {m.tool_call_id: m.text for m in session.state.messages if m.role == "tool"}
    assert results["c1"] == "skipped: the user denied send"


async def test_a_denial_with_feedback_lets_the_other_approved_calls_run(log):
    ran = []
    approver = RecordingApprover(ran, [Decision("approve"), Decision("deny", feedback="not 2")])
    tools = ToolRegistry([Tool("send", "", {"type": "object", "properties": {"n": {"type": "integer"}}},
                               lambda n: ran.append(n) or "sent", requires_approval=True)])
    reply = ModelReply(tool_calls=(ToolCall("c1", "send", {"n": 1}), ToolCall("c2", "send", {"n": 2})))
    session = Session.create(log, FakeModel([reply, say("ok")]), tools, approver=approver)

    await session.run_turn("go")
    assert ran == [1]


async def test_live_state_equals_replay_after_parallel_calls(log):
    tools = ToolRegistry([Tool("slow", "", {"type": "object"}, lambda: time.sleep(0.01) or "ok")])
    reply = ModelReply(tool_calls=tuple(ToolCall(f"c{i}", "slow", {}) for i in range(5)))
    session = Session.create(log, FakeModel([reply, say("done")]), tools, approver=AutoApprove())
    await session.run_turn("go")

    assert session.state == project(log.load(session.session_id))


async def test_cancelling_a_turn_cancels_its_running_calls(log):
    async def forever():
        await asyncio.sleep(60)

    tools = ToolRegistry([Tool("forever", "", {"type": "object"}, forever)])
    reply = ModelReply(tool_calls=(ToolCall("c1", "forever", {}), ToolCall("c2", "forever", {})))
    session = Session.create(log, FakeModel([reply]), tools)

    turn = asyncio.create_task(session.run_turn("go"))
    await asyncio.sleep(0.05)
    turn.cancel()
    with pytest.raises(asyncio.CancelledError):
        await turn
    session.cancel()

    assert not session.needs_resume
    assert all(m.is_error for m in session.state.messages if m.role == "tool")


# ── helpers ──────────────────────────────────────────────────────────────────────────────


class RecordingApprover:
    """Scripted decisions; also records what had already run each time it was asked."""

    def __init__(self, ran, decisions):
        self.ran = ran
        self.decisions = list(decisions)
        self.ran_before_each_ask = []

    async def ask(self, call):
        self.ran_before_each_ask.append(list(self.ran))
        return self.decisions.pop(0)
