"""Following a session live: EventHub / EventStream, and session.subscribe() during a turn."""

import asyncio

import pytest

from runtime.events import Event, EventLog
from runtime.messages import ToolCall
from runtime.model import ModelReply
from runtime.session import Session
from runtime.subscriptions import EventHub
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, call, say


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


async def test_a_stream_gets_events_in_order_and_ends_when_closed():
    hub = EventHub()
    stream = hub.subscribe()
    for seq in (1, 2, 3):
        hub.publish(_event(seq))
    stream.close()
    hub.publish(_event(4))  # after close: not delivered

    assert [e.seq async for e in stream] == [1, 2, 3]


async def test_every_subscriber_gets_every_event():
    hub = EventHub()
    a, b = hub.subscribe(), hub.subscribe()
    hub.publish(_event(1))
    a.close(), b.close()
    assert [e.seq async for e in a] == [1] and [e.seq async for e in b] == [1]


async def test_only_events_after_subscribing_are_delivered():
    hub = EventHub()
    hub.publish(_event(1))
    stream = hub.subscribe()
    hub.publish(_event(2))
    stream.close()
    assert [e.seq async for e in stream] == [2]


async def test_drained_waits_until_the_subscriber_has_handled_everything():
    hub = EventHub()
    stream = hub.subscribe()
    handled = []

    async def consume():
        async for event in stream:
            await asyncio.sleep(0.01)
            handled.append(event.seq)

    consumer = asyncio.create_task(consume())
    for seq in (1, 2, 3):
        hub.publish(_event(seq))
    await asyncio.wait_for(stream.drained(), timeout=2)
    assert handled == [1, 2, 3]
    stream.close()
    await consumer


async def test_a_subscriber_follows_a_turn_while_it_runs(log):
    """The display task sees each event as it's recorded, not all at the end."""
    gate = asyncio.Event()

    async def slow():
        await gate.wait()
        return "done"

    tools = ToolRegistry([Tool("slow", "", {"type": "object"}, slow)])
    session = Session.create(log, FakeModel([call("slow"), say("all done")]), tools)

    async with session.subscribe() as events:
        turn = asyncio.create_task(session.run_turn("go"))
        seen = []
        async for event in events:
            seen.append(event.type)
            if event.type == "tool_started":
                assert not turn.done()  # the turn is still waiting on the tool
                gate.set()
            if event.type == "assistant_message" and not event.payload["tool_calls"]:
                break
        await turn

    assert seen == [log_event.type for log_event in log.load(session.session_id)]


async def test_a_subscriber_that_never_reads_does_not_block_the_turn(log):
    session = Session.create(log, FakeModel([call("echo"), say("done")]),
                             ToolRegistry([Tool("echo", "", {"type": "object"}, lambda: "ok")]))
    session.subscribe()  # never read
    assert (await asyncio.wait_for(session.run_turn("go"), timeout=5)).text == "done"


async def test_state_is_up_to_date_when_an_event_arrives(log):
    reply = ModelReply(tool_calls=(ToolCall("c1", "echo", {}),))
    session = Session.create(log, FakeModel([reply, say("done")]),
                             ToolRegistry([Tool("echo", "", {"type": "object"}, lambda: "ok")]))
    async with session.subscribe() as events:
        turn = asyncio.create_task(session.run_turn("go"))
        async for event in events:
            assert session.state.last_seq >= event.seq
            if event.type == "tool_started":
                assert "c1" in session.state.tool_calls
            if event.type == "assistant_message" and not event.payload["tool_calls"]:
                break
        await turn


def _event(seq):
    return Event("s", seq, "x", {}, "ts")
