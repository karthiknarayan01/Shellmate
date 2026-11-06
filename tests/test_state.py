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


def test_reasoning_and_argument_errors_survive_replay():
    call = {"id": "c1", "name": "t", "arguments": {}, "arguments_error": "not valid JSON"}
    state = project(
        _events(
            ("user_message", {"text": "go"}),
            ("assistant_message", {"text": "", "reasoning": "hmm", "tool_calls": [call]}),
        )
    )
    assert state.messages[1].reasoning == "hmm"
    assert state.messages[1].tool_calls[0].arguments_error == "not valid JSON"


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
    with pytest.raises(InvalidEventSequence, match="expected approved or requested"):
        project(_tool_turn(("tool_started", {"call_id": "c1"}), ("tool_started", {"call_id": "c1"})))


def test_a_new_turn_cannot_start_while_one_is_open():
    with pytest.raises(InvalidEventSequence, match="turn is still open"):
        project(_tool_turn(("user_message", {"text": "never mind"})))


def test_model_cannot_reply_while_calls_have_no_result():
    with pytest.raises(InvalidEventSequence, match="have no result"):
        project(_tool_turn(("assistant_message", {"text": "anyway", "tool_calls": []})))


def test_duplicate_call_ids_are_rejected():
    call = {"id": "c1", "name": "t", "arguments": {}}
    with pytest.raises(InvalidEventSequence, match="duplicate tool call id"):
        project(
            _events(
                ("user_message", {"text": "go"}),
                ("assistant_message", {"text": "", "tool_calls": [call, call]}),
            )
        )


# ── approvals and skips ──────────────────────────────────────────────────────────────────


def test_approved_call_runs_after_approval():
    state = project(_tool_turn(
        ("approval_requested", {"call_id": "c1"}),
        ("approval_decided", {"call_id": "c1", "decision": "approve", "feedback": "", "source": "approver"}),
        ("tool_started", {"call_id": "c1"}),
        ("tool_finished", {"call_id": "c1", "output": "done", "is_error": False}),
    ))
    assert state.messages[-1].text == "done"
    assert state.approved_tools == set()


def test_waiting_for_approval_is_visible_after_replay():
    state = project(_tool_turn(("approval_requested", {"call_id": "c1"})))
    assert state.tool_calls["c1"].status is ToolCallStatus.AWAITING_APPROVAL
    assert [t.call.id for t in state.unfinished_calls()] == ["c1"]


@pytest.mark.parametrize(
    "feedback, text",
    [("", "The user denied this tool call."),
     ("edit b.py instead", "The user denied this tool call. Their feedback: edit b.py instead")],
    ids=["no-feedback", "with-feedback"],
)
def test_denied_call_is_finished_with_a_denial_result(feedback, text):
    state = project(_tool_turn(
        ("approval_requested", {"call_id": "c1"}),
        ("approval_decided", {"call_id": "c1", "decision": "deny", "feedback": feedback, "source": "approver"}),
    ))
    assert state.tool_calls["c1"].status is ToolCallStatus.FINISHED
    assert state.messages[-1] == Message(role="tool", text=text, tool_call_id="c1", is_error=True)


def test_approve_for_session_is_remembered():
    state = project(_tool_turn(
        ("approval_requested", {"call_id": "c1"}),
        ("approval_decided", {"call_id": "c1", "decision": "approve_for_session", "feedback": "", "source": "approver"}),
    ))
    assert state.approved_tools == {"t"}


def test_skipped_call_gets_an_error_result_without_starting():
    state = project(_tool_turn(("tool_skipped", {"call_id": "c1", "reason": "skipped: denied"})))
    assert state.messages[-1].text == "skipped: denied" and state.messages[-1].is_error
    assert state.open_calls == {}


@pytest.mark.parametrize(
    "tail, error",
    [
        ((("approval_decided", {"call_id": "c1", "decision": "approve", "feedback": "", "source": "x"}),),
         "expected awaiting_approval"),
        ((("approval_requested", {"call_id": "c1"}), ("tool_started", {"call_id": "c1"})),
         "expected approved or requested"),
        ((("approval_requested", {"call_id": "c1"}),
          ("approval_decided", {"call_id": "c1", "decision": "maybe", "feedback": "", "source": "x"})),
         "unknown decision"),
        ((("tool_started", {"call_id": "c1"}), ("tool_skipped", {"call_id": "c1", "reason": "r"})),
         "expected requested"),
    ],
    ids=["decided-without-request", "started-while-awaiting", "unknown-decision", "skip-after-start"],
)
def test_out_of_order_approval_events_are_rejected(tail, error):
    with pytest.raises(InvalidEventSequence, match=error):
        project(_tool_turn(*tail))



# ── turns and restarts ───────────────────────────────────────────────────────────────────


def test_turn_is_open_until_a_reply_without_tool_calls():
    call = {"id": "c1", "name": "t", "arguments": {}}
    steps = [
        ("user_message", {"text": "go"}),
        ("assistant_message", {"text": "", "tool_calls": [call]}),
        ("tool_started", {"call_id": "c1"}),
        ("tool_finished", {"call_id": "c1", "output": "ok", "is_error": False}),
        ("assistant_message", {"text": "done", "tool_calls": []}),
    ]
    open_after_each = [project(_events(*steps[:n])).turn_open for n in range(len(steps) + 1)]
    assert open_after_each == [False, True, True, True, True, False]


def test_turn_stopped_closes_the_turn():
    state = project(_tool_turn(
        ("tool_skipped", {"call_id": "c1", "reason": "skipped"}),
        ("turn_stopped", {"reason": "denied"}),
    ))
    assert state.turn_open is False


@pytest.mark.parametrize(
    "events, error",
    [
        ([("turn_stopped", {"reason": "x"})], "no open turn"),
        ([("user_message", {"text": "go"}),
          ("assistant_message", {"text": "", "tool_calls": [{"id": "c1", "name": "t", "arguments": {}}]}),
          ("turn_stopped", {"reason": "x"})], "have no result"),
    ],
    ids=["no-turn", "open-calls"],
)
def test_turn_stopped_needs_an_open_turn_with_no_open_calls(events, error):
    with pytest.raises(InvalidEventSequence, match=error):
        project(_events(*events))


def test_restart_keeps_the_call_started():
    state = project(_tool_turn(("tool_started", {"call_id": "c1"}), ("tool_restarted", {"call_id": "c1"})))
    assert state.tool_calls["c1"].status is ToolCallStatus.STARTED


def test_restart_of_a_call_that_never_started_is_rejected():
    with pytest.raises(InvalidEventSequence, match="expected started"):
        project(_tool_turn(("tool_restarted", {"call_id": "c1"})))


# ── compaction and token measurement ─────────────────────────────────────────────────────


def test_compaction_replaces_the_first_messages_with_the_summary():
    state = project(_events(
        ("user_message", {"text": "q1"}),
        ("assistant_message", {"text": "a1", "tool_calls": []}),
        ("user_message", {"text": "q2"}),
        ("compaction", {"replaced_count": 2, "summary": "they talked"}),
    ))
    assert [(m.role, m.text) for m in state.messages] == [("user", "they talked"), ("user", "q2")]


def test_usage_is_recorded_as_a_measurement_and_reset_by_compaction():
    usage = {"input_tokens": 900, "output_tokens": 100}
    events = [
        ("user_message", {"text": "q1"}),
        ("assistant_message", {"text": "a1", "tool_calls": [], "usage": usage}),
        ("user_message", {"text": "q2"}),
    ]
    state = project(_events(*events))
    assert (state.measured_tokens, state.measured_through) == (1_000, 2)
    state = project(_events(*events, ("compaction", {"replaced_count": 2, "summary": "s"})))
    assert state.measured_tokens is None


def test_zero_usage_is_not_a_measurement():
    state = project(_events(
        ("user_message", {"text": "q"}),
        ("assistant_message", {"text": "a", "tool_calls": [], "usage": {"input_tokens": 0, "output_tokens": 0}}),
    ))
    assert state.measured_tokens is None


def test_turn_request_is_the_latest_user_message():
    assert project(_events(("user_message", {"text": "fix the bug"}))).turn_request == "fix the bug"


@pytest.mark.parametrize(
    "count, error",
    [(0, "compaction of 0"), (3, "compaction of 3"), (2, "orphan a tool result")],
    ids=["nothing", "everything", "orphans-result"],
)
def test_invalid_compactions_are_rejected(count, error):
    events = _tool_turn(
        ("tool_started", {"call_id": "c1"}),
        ("tool_finished", {"call_id": "c1", "output": "ok", "is_error": False}),
        ("compaction", {"replaced_count": count, "summary": "s"}),
    )
    with pytest.raises(InvalidEventSequence, match=error):
        project(events)


def test_compaction_with_open_calls_is_rejected():
    with pytest.raises(InvalidEventSequence, match="have no result"):
        project(_tool_turn(("compaction", {"replaced_count": 1, "summary": "s"})))

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
