"""Replaying events into SessionState: messages, tool call status, and rejection of corrupt logs."""

import pytest

from runtime.events import Event, EventLog
from runtime.messages import Message, ToolCall
from runtime.state import InvalidEventSequence, SessionState, ToolCallStatus, apply, project


def test_empty_log_is_empty_state():
    state = project([])
    assert state.messages == [] and state.tool_calls == {} and state.last_seq == 0


def test_plain_conversation():
    state = project(
        _events(
            ("user_message", {"text": "hi"}),
            ("assistant_message", {"text": "hello", "tool_calls": []}),
        )
    )
    assert state.messages == [
        Message(role="user", text="hi"),
        Message(role="assistant", text="hello"),
    ]
    assert state.last_seq == 2


def test_full_tool_round_trip():
    call = {"id": "c1", "name": "read_file", "arguments": {"path": "a.py"}}
    state = project(
        _events(
            ("user_message", {"text": "read a.py"}),
            ("assistant_message", {"text": "", "tool_calls": [call]}),
            ("tool_started", {"call_id": "c1"}),
            ("tool_finished", {"call_id": "c1", "output": "print(1)", "is_error": False}),
            ("assistant_message", {"text": "it prints 1", "tool_calls": []}),
        )
    )
    assert [m.role for m in state.messages] == ["user", "assistant", "tool", "assistant"]
    assert state.messages[1].tool_calls == (ToolCall("c1", "read_file", {"path": "a.py"}),)
    assert state.messages[2] == Message(role="tool", text="print(1)", tool_call_id="c1")
    assert state.tool_calls["c1"].status is ToolCallStatus.FINISHED
    assert state.unfinished_calls() == []


def test_tool_error_result_is_marked():
    state = project(
        _tool_turn(("tool_started", {"call_id": "c1"}),
                   ("tool_finished", {"call_id": "c1", "output": "boom", "is_error": True}))
    )
    assert state.messages[-1].is_error is True


@pytest.mark.parametrize(
    "tail, expected",
    [
        ((), ToolCallStatus.REQUESTED),
        ((("tool_started", {"call_id": "c1"}),), ToolCallStatus.STARTED),
    ],
    ids=["crashed-before-start", "crashed-mid-tool"],
)
def test_crash_points_are_visible_after_replay(tail, expected):
    """A log that stops mid-turn replays to a call left open, in the status it reached."""
    state = project(_tool_turn(*tail))
    assert [t.call.id for t in state.unfinished_calls()] == ["c1"]
    assert state.tool_calls["c1"].status is expected
    assert state.messages[-1].role == "assistant"  # no tool result was added


def test_parallel_calls_stay_open_in_call_order():
    calls = [{"id": i, "name": "t", "arguments": {}} for i in ("c1", "c2", "c3")]
    state = project(
        _events(
            ("user_message", {"text": "go"}),
            ("assistant_message", {"text": "", "tool_calls": calls}),
            ("tool_started", {"call_id": "c2"}),
            ("tool_finished", {"call_id": "c2", "output": "ok", "is_error": False}),
        )
    )
    assert [t.call.id for t in state.unfinished_calls()] == ["c1", "c3"]


def test_replay_is_deterministic():
    events = _tool_turn(("tool_started", {"call_id": "c1"}))
    assert project(events) == project(events)


def test_applying_events_one_by_one_equals_full_replay():
    events = _tool_turn(("tool_started", {"call_id": "c1"}))
    state = SessionState()
    for event in events:
        apply(state, event)
    assert state == project(events)


def test_replay_from_a_real_event_log(db):
    with EventLog(db) as log:
        sid = log.new_session()
        log.append(sid, "user_message", {"text": "hi"})
        log.append(sid, "assistant_message", {"text": "hello", "tool_calls": []})
        state = project(log.load(sid))
    assert [m.text for m in state.messages] == ["hi", "hello"]


@pytest.mark.parametrize(
    "events, error",
    [
        ([("mystery", {})], "unknown event type"),
        ([("user_message", {"text": "x"}), ("tool_started", {"call_id": "nope"})], "unknown call"),
    ],
    ids=["unknown-type", "unknown-call"],
)
def test_corrupt_logs_are_rejected(events, error):
    with pytest.raises(InvalidEventSequence, match=error):
        project(_events(*events))


def test_finish_without_start_is_rejected():
    with pytest.raises(InvalidEventSequence, match="expected started"):
        project(_tool_turn(("tool_finished", {"call_id": "c1", "output": "", "is_error": False})))


def test_starting_twice_is_rejected():
    with pytest.raises(InvalidEventSequence, match="expected requested"):
        project(_tool_turn(("tool_started", {"call_id": "c1"}), ("tool_started", {"call_id": "c1"})))


def test_conversation_cannot_move_on_with_open_calls():
    with pytest.raises(InvalidEventSequence, match="have no result"):
        project(_tool_turn(("user_message", {"text": "never mind"})))


def test_duplicate_call_ids_are_rejected():
    call = {"id": "c1", "name": "t", "arguments": {}}
    with pytest.raises(InvalidEventSequence, match="duplicate tool call id"):
        project(
            _events(
                ("user_message", {"text": "go"}),
                ("assistant_message", {"text": "", "tool_calls": [call, call]}),
            )
        )


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _tool_turn(*tail):
    """A user message, then a reply with one tool call c1, then the given events."""
    call = {"id": "c1", "name": "t", "arguments": {}}
    return _events(
        ("user_message", {"text": "go"}),
        ("assistant_message", {"text": "", "tool_calls": [call]}),
        *tail,
    )


def _events(*pairs):
    return [Event("s", seq, type, payload, "ts") for seq, (type, payload) in enumerate(pairs, 1)]
