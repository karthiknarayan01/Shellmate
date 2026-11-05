"""Behaviour of EventLog in a single process: ordering, filtering, errors, schema check."""

import pytest

from runtime.events import EventLog, SchemaOutOfDate, UnknownSession


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


def test_append_and_load_in_order(log):
    sid = log.new_session()
    log.append(sid, "user_message", {"text": "hi"})
    log.append(sid, "assistant_message", {"text": "hello", "tool_calls": []})
    log.append(sid, "user_message", {"text": "ünïcode ✓"})

    events = log.load(sid)
    assert [e.type for e in events] == ["user_message", "assistant_message", "user_message"]
    assert [e.seq for e in events] == sorted(e.seq for e in events)
    assert events[1].payload == {"text": "hello", "tool_calls": []}
    assert events[2].payload["text"] == "ünïcode ✓"


def test_append_returns_the_stored_seq(log):
    sid = log.new_session()
    written = log.append(sid, "x", {"k": 1})
    assert log.load(sid) == [written]


def test_sessions_only_see_their_own_events(log):
    a, b = log.new_session(), log.new_session()
    log.append(a, "x", {"from": "a"})
    log.append(b, "x", {"from": "b"})
    log.append(a, "x", {"from": "a"})
    assert [e.payload["from"] for e in log.load(a)] == ["a", "a"]
    assert [e.payload["from"] for e in log.load(b)] == ["b"]


def test_load_after_seq_returns_only_newer_events(log):
    sid = log.new_session()
    seqs = [log.append(sid, "x", {"i": i}).seq for i in range(5)]
    assert [e.payload["i"] for e in log.load(sid, after_seq=seqs[2])] == [3, 4]


def test_new_session_loads_empty(log):
    assert log.load(log.new_session()) == []


def test_unknown_session_is_rejected(log):
    with pytest.raises(UnknownSession):
        log.append("nope", "x", {})
    with pytest.raises(UnknownSession):
        log.load("nope")


def test_latest_session_is_the_newest(log):
    assert log.latest_session() is None
    log.new_session()
    newest = log.new_session()
    assert log.latest_session() == newest


def test_unmigrated_database_is_refused(tmp_path):
    with pytest.raises(SchemaOutOfDate, match="python -m runtime.migrate"):
        EventLog(str(tmp_path / "fresh.db"))
