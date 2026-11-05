"""EventLog across processes: events survive reopen and SIGKILL; concurrent writers never collide.

These tests use real OS processes (multiprocessing with "spawn"), because the
property being tested is what happens when a process dies or races another one.
"""

import multiprocessing as mp
import os
import signal

from runtime.events import EventLog


def test_events_survive_reopen(db):
    log = EventLog(db)
    sid = log.new_session()
    log.append(sid, "user_message", {"text": "remember me"})
    log.close()

    with EventLog(db) as reopened:
        assert reopened.latest_session() == sid
        assert reopened.load(sid)[0].payload == {"text": "remember me"}


def test_event_survives_sigkill(db):
    """A child appends one event, then kills itself with no close() and no cleanup."""
    with EventLog(db) as log:
        sid = log.new_session()

    ctx = mp.get_context("spawn")
    appended = ctx.Event()
    child = ctx.Process(target=_append_then_sigkill, args=(db, sid, appended))
    child.start()
    child.join(timeout=30)
    assert appended.is_set() and child.exitcode == -signal.SIGKILL

    with EventLog(db) as log:
        assert [e.type for e in log.load(sid)] == ["tool_started"]


def test_concurrent_writers_never_share_a_seq(db):
    """4 processes x 50 appends to one session -> 200 events, 200 distinct increasing seqs."""
    with EventLog(db) as log:
        sid = log.new_session()

    ctx = mp.get_context("spawn")
    writers = [ctx.Process(target=_append_many, args=(db, sid, 50)) for _ in range(4)]
    for w in writers:
        w.start()
    for w in writers:
        w.join(timeout=60)
        assert w.exitcode == 0

    with EventLog(db) as log:
        seqs = [e.seq for e in log.load(sid)]
    assert len(seqs) == 200
    assert seqs == sorted(set(seqs))


# ── child-process bodies (must be module-level so "spawn" can import them) ───────────────


def _append_then_sigkill(db, sid, appended):
    log = EventLog(db)
    log.append(sid, "tool_started", {"call_id": "c1"})
    appended.set()
    os.kill(os.getpid(), signal.SIGKILL)


def _append_many(db, sid, n):
    with EventLog(db) as log:
        for _ in range(n):
            log.append(sid, "x", {"pid": os.getpid()})
