"""In-memory publish/subscribe for a session's events, so a UI can follow a turn as it runs.

    session ──_record()──► EventHub.publish(event) ──► EventStream (one queue per subscriber)
    model stream ─────────► EventHub.publish(delta)         │
                       UI task:  async for item in stream: show it

Two kinds of item are published, in the order they happen:
    Event        a recorded event (it is already in the log)
    StreamDelta  a piece of a model reply while it is being generated (never in the log; the
                 complete reply follows as an assistant_message Event)

Publishing never waits: each subscriber has its own unbounded queue, so a slow UI can't slow
the agent down, and the agent never sees who is listening. Every event is published after it
is in the log and in SessionState, so a subscriber that looks at the state sees it up to date.

A subscriber can wait until it has handled everything published so far (`drained()`), e.g.
before printing an approval prompt, so the prompt comes after the progress lines that led to it.
"""

from __future__ import annotations

import asyncio

from runtime.events import Event
from runtime.model import StreamDelta

Item = Event | StreamDelta

_END = object()


class EventHub:
    def __init__(self) -> None:
        self._streams: list[EventStream] = []

    def subscribe(self) -> EventStream:
        """A new stream that receives every event published from now on."""
        stream = EventStream(self)
        self._streams.append(stream)
        return stream

    def publish(self, item: Item) -> None:
        for stream in list(self._streams):
            stream._put(item)

    def _remove(self, stream: EventStream) -> None:
        if stream in self._streams:
            self._streams.remove(stream)


class EventStream:
    """One subscriber's items, in order. Iterate with `async for`; close() ends the iteration.

    Use as `async with session.subscribe() as events:` so it is closed (and stops receiving)
    when the block ends.
    """

    def __init__(self, hub: EventHub):
        self._hub = hub
        self._queue: asyncio.Queue = asyncio.Queue()
        self._handling = False
        self._closed = False

    async def __aenter__(self) -> EventStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()

    def __aiter__(self) -> EventStream:
        return self

    async def __anext__(self) -> Item:
        """The next item. Asking for it also marks the previous one as handled."""
        self._mark_handled()
        item = await self._queue.get()
        if item is _END:
            self._queue.task_done()
            raise StopAsyncIteration
        self._handling = True
        return item

    async def drained(self) -> None:
        """Wait until the subscriber has handled every event published so far."""
        await self._queue.join()

    def close(self) -> None:
        """Stop receiving; iteration ends after the events already queued."""
        if self._closed:
            return
        self._closed = True
        self._hub._remove(self)
        self._queue.put_nowait(_END)

    def _put(self, item: Item) -> None:
        if not self._closed:
            self._queue.put_nowait(item)

    def _mark_handled(self) -> None:
        if self._handling:
            self._handling = False
            self._queue.task_done()
