"""Behaviour of EventLog in a single process: ordering, per-session seqs, filtering, errors."""

import pytest

from runtime.events import EventLog, UnknownSession


@pytest.fixture
def log(tmp_path):
    with EventLog(str(tmp_path / "events.db")) as log:
        yield log


def test_append_and_load_in_order(log):
    sid = log.new_session()
    log.append(sid, "user_message", {"text": "hi"})
    log.append(sid, "assistant_message", {"text": "hello", "tool_calls": []})
    log.append(sid, "user_message", {"text": "ünïcode ✓"})

    events = log.load(sid)
    assert [e.seq for e in events] == [1, 2, 3]
    assert [e.type for e in events] == ["user_message", "assistant_message", "user_message"]
    assert events[1].payload == {"text": "hello", "tool_calls": []}
    assert events[2].payload["text"] == "ünïcode ✓"


def test_sessions_have_independent_seqs(log):
    a, b = log.new_session(), log.new_session()
    log.append(a, "x", {})
    log.append(a, "x", {})
    assert log.append(b, "x", {}).seq == 1
    assert log.append(a, "x", {}).seq == 3


def test_load_after_seq_returns_only_newer_events(log):
    sid = log.new_session()
    for i in range(5):
        log.append(sid, "x", {"i": i})
    assert [e.payload["i"] for e in log.load(sid, after_seq=3)] == [3, 4]


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
