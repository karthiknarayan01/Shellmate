"""Compaction helpers: token estimate, where to cut, the transcript, the summary message, the policy."""

import pytest

from runtime.compaction import (
    REQUEST_HEADER,
    SUMMARY_HEADER,
    CompactionPolicy,
    choose_cut,
    estimate_tokens,
    message_tokens,
    policy_from_env,
    render_transcript,
    summary_message_text,
)
from runtime.messages import Message, ToolCall
from runtime.model import ToolSpec
from runtime.state import SessionState


def test_message_tokens_counts_text_reasoning_and_tool_calls():
    plain = Message("user", "x" * 400)
    assert message_tokens(plain) == 101
    rich = Message("assistant", "x" * 40, reasoning="y" * 40,
                   tool_calls=(ToolCall("c", "read_file", {"path": "a.py"}),))
    assert message_tokens(rich) > message_tokens(Message("assistant", "x" * 40))


def test_estimate_without_a_measurement_counts_everything():
    state = SessionState(messages=[Message("user", "x" * 4_000)])
    tools = [ToolSpec("t", "d" * 400, {"type": "object"})]
    assert estimate_tokens(state, "s" * 400, tools) > estimate_tokens(state, "", [])
    assert estimate_tokens(state, "", []) == 1_001


def test_estimate_uses_the_measured_count_plus_newer_messages():
    state = SessionState(
        messages=[Message("user", "x" * 40_000), Message("assistant", "ok"), Message("tool", "y" * 400)],
        measured_tokens=12_000,
        measured_through=2,
    )
    assert estimate_tokens(state, "ignored", []) == 12_000 + message_tokens(state.messages[2])


def test_cut_keeps_the_recent_tail_within_budget():
    messages = [Message("user", "x" * 4_000) for _ in range(10)]  # ~1001 tokens each
    assert choose_cut(messages, keep_tokens=3_000) == 8  # keep the last 2


def test_cut_never_starts_the_kept_part_with_a_tool_result():
    call = ToolCall("c1", "read", {})
    messages = [
        Message("user", "x" * 4_000),
        Message("assistant", "", tool_calls=(call,)),
        Message("tool", "y" * 4_000, tool_call_id="c1"),
        Message("assistant", "done"),
    ]
    cut = choose_cut(messages, keep_tokens=1_100)
    assert messages[cut].role != "tool"
    assert cut == 1  # keeps the call together with its result


def test_cut_keeps_at_least_one_message_even_if_it_is_over_budget():
    messages = [Message("user", "a"), Message("assistant", "x" * 40_000)]
    assert choose_cut(messages, keep_tokens=10) == 1


def test_nothing_to_cut_in_a_one_message_history():
    assert choose_cut([Message("user", "hi")], keep_tokens=10) is None


def test_transcript_labels_roles_calls_and_clips_long_results():
    transcript = render_transcript([
        Message("user", "read a.py"),
        Message("assistant", "Reading.", tool_calls=(ToolCall("c1", "read_file", {"path": "a.py"}),)),
        Message("tool", "z" * 5_000, tool_call_id="c1"),
        Message("tool", "no such file", tool_call_id="c2", is_error=True),
    ])
    assert "USER: read a.py" in transcript
    assert 'ASSISTANT CALLED read_file({"path": "a.py"})' in transcript
    assert "[clipped]" in transcript and "z" * 2_001 not in transcript
    assert "TOOL ERROR: no such file" in transcript


def test_summary_message_ends_with_the_request_verbatim():
    text = summary_message_text("  Read three files.  ", "Fix the failing test in app.py")
    assert text.startswith(SUMMARY_HEADER)
    assert text.endswith(f"{REQUEST_HEADER}\nFix the failing test in app.py")


def test_policy_thresholds():
    policy = CompactionPolicy(context_window=100_000)
    assert policy.trigger_tokens == 80_000 and policy.keep_tokens == 25_000


def test_policy_from_env(monkeypatch):
    monkeypatch.delenv("AGENT_CONTEXT_WINDOW", raising=False)
    assert policy_from_env() == CompactionPolicy()
    monkeypatch.setenv("AGENT_CONTEXT_WINDOW", "200000")
    assert policy_from_env().context_window == 200_000


@pytest.mark.parametrize("raw, error", [("big", "must be an integer"), ("10", "at least 1000")])
def test_bad_context_window_is_rejected(monkeypatch, raw, error):
    monkeypatch.setenv("AGENT_CONTEXT_WINDOW", raw)
    with pytest.raises(ValueError, match=error):
        policy_from_env()
