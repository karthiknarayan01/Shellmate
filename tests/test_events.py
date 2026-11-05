import multiprocessing as mp
import os
import signal

import pytest

from runtime.events import EventLog, UnknownSession


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "events.db")


def test_append_and_load_in_order(db):
    with EventLog(db) as log:
        sid = log.new_session()
        log.append(sid, "user_message", {"text": "hi"})
        log.append(sid, "assistant_message", {"text": "hello", "tool_calls": []})
        log.append(sid, "user_message", {"text": "ünïcode ✓"})

        events = log.load(sid)
        assert [e.seq for e in events] == [1, 2, 3]
        assert [e.type for e in events] == ["user_message", "assistant_message", "user_message"]
        assert events[1].payload == {"text": "hello", "tool_calls": []}
        assert events[2].payload["text"] == "ünïcode ✓"


def test_sessions_have_independent_seqs(db):
    with EventLog(db) as log:
        a, b = log.new_session(), log.new_session()
        log.append(a, "x", {})
        log.append(a, "x", {})
        assert log.append(b, "x", {}).seq == 1
        assert log.append(a, "x", {}).seq == 3


def test_load_after_seq(db):
    with EventLog(db) as log:
        sid = log.new_session()
        for i in range(5):
            log.append(sid, "x", {"i": i})
        assert [e.payload["i"] for e in log.load(sid, after_seq=3)] == [3, 4]


def test_unknown_session_rejected(db):
    with EventLog(db) as log:
        with pytest.raises(UnknownSession):
            log.append("nope", "x", {})
        with pytest.raises(UnknownSession):
            log.load("nope")


def test_events_survive_reopen(db):
    log = EventLog(db)
    sid = log.new_session()
    log.append(sid, "user_message", {"text": "remember me"})
    log.close()

    with EventLog(db) as log2:
        assert log2.latest_session() == sid
        assert log2.load(sid)[0].payload == {"text": "remember me"}


def _append_then_die(db, sid, ready):
    log = EventLog(db)
    log.append(sid, "tool_started", {"call_id": "c1"})
    ready.set()
    os.kill(os.getpid(), signal.SIGKILL)  # no close(), no cleanup: a real crash


def test_event_survives_sigkill(db):
    with EventLog(db) as log:
        sid = log.new_session()

    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    p = ctx.Process(target=_append_then_die, args=(db, sid, ready))
    p.start()
    p.join(timeout=30)
    assert ready.is_set() and p.exitcode == -signal.SIGKILL

    with EventLog(db) as log:
        assert [(e.seq, e.type) for e in log.load(sid)] == [(1, "tool_started")]


def _append_many(db, sid, n):
    with EventLog(db) as log:
        for _ in range(n):
            log.append(sid, "x", {"pid": os.getpid()})


def test_concurrent_writers_never_share_a_seq(db):
    with EventLog(db) as log:
        sid = log.new_session()

    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(db, sid, 50)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0

    with EventLog(db) as log:
        seqs = [e.seq for e in log.load(sid)]
    assert seqs == list(range(1, 201))
