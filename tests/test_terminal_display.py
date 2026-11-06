"""How a turn is drawn: tool lines, result summaries, streamed answers, notes. Plain-text output."""

import io

import pytest
from rich.console import Console

from runtime.events import Event
from runtime.messages import ToolCall
from runtime.model import StreamDelta
from runtime.state import ToolCallState
from runtime.terminal_display import TurnDisplay, summarize


class FakeSession:
    """Just enough of a Session for the display: the tool calls it looks names up in."""

    def __init__(self, *calls):
        self.state = type("State", (), {"tool_calls": {c.id: ToolCallState(c) for c in calls}})()


def test_a_tool_call_and_its_result():
    out = _draw(
        [ToolCall("c1", "read_file", {"path": "config.py"})],
        _event("tool_started", call_id="c1"),
        _event("tool_finished", call_id="c1", output="    1  a\n    2  b", is_error=False),
    )
    assert "● read_file  path=config.py" in out
    assert "⎿  2 lines" in out


def test_an_error_result():
    out = _draw(
        [ToolCall("c1", "read_file", {"path": "x"})],
        _event("tool_started", call_id="c1"),
        _event("tool_finished", call_id="c1", output="read_file failed: FileNotFoundError: x", is_error=True),
    )
    assert "⎿  ✗ read_file failed: FileNotFoundError: x" in out


def test_parallel_results_say_which_call_they_belong_to():
    out = _draw(
        [ToolCall("c1", "read_file", {"path": "a"}), ToolCall("c2", "list_files", {})],
        _event("tool_started", call_id="c1"),
        _event("tool_started", call_id="c2"),
        _event("tool_finished", call_id="c2", output="a\nb\nc", is_error=False),
        _event("tool_finished", call_id="c1", output="    1  x", is_error=False),
    )
    assert "⎿  list_files: 3 files" in out
    assert "⎿  read_file: 1 line" in out


def test_multi_line_arguments_are_shown_by_size():
    out = _draw(
        [ToolCall("c1", "write_file", {"path": "a.py", "content": "one\ntwo\nthree\n"})],
        _event("tool_started", call_id="c1"),
    )
    assert "● write_file  path=a.py  content=⟨3 lines⟩" in out


def test_a_restarted_call_is_marked():
    out = _draw([ToolCall("c1", "read_file", {"path": "a"})], _event("tool_restarted", call_id="c1"))
    assert "(again, after a crash)" in out


def test_skipped_calls_and_compactions():
    out = _draw(
        [ToolCall("c1", "write_file", {"path": "a"})],
        _event("tool_skipped", call_id="c1", reason="skipped: the user denied write_file"),
        _event("compaction", replaced_count=12, summary="s"),
    )
    assert "write_file  skipped: skipped: the user denied write_file" in out
    assert "Compacted 12 earlier messages" in out


def test_streamed_answer_is_rendered_once_as_markdown():
    out = _draw(
        [],
        StreamDelta("text", "The **answer** "),
        StreamDelta("text", "is 42."),
        _event("assistant_message", text="The **answer** is 42.", tool_calls=[]),
    )
    assert out.count("The answer is 42.") == 1 and "**" not in out


def test_an_answer_that_was_not_streamed_is_printed_from_its_event():
    out = _draw([], _event("assistant_message", text="Plain *reply*", tool_calls=[]))
    assert "Plain reply" in out


def test_the_spinner_stops_when_an_approval_is_requested():
    """The approval panel needs the screen; a spinner redrawing over it would garble the prompt."""
    display = TurnDisplay(Console(file=io.StringIO()), FakeSession(ToolCall("c1", "write_file", {})))
    display._show_spinner("Working")
    display.show(_event("approval_requested", call_id="c1"))
    assert display._live is None


def test_usage_is_added_up_for_the_turn():
    display = TurnDisplay(Console(file=io.StringIO()), FakeSession())
    for tokens in ((100, 10), (250, 40)):
        display.show(_event("assistant_message", text="", tool_calls=[],
                            usage={"input_tokens": tokens[0], "output_tokens": tokens[1]}))
    assert (display.stats.input_tokens, display.stats.output_tokens) == (350, 50)


@pytest.mark.parametrize(
    "tool, output, summary",
    [
        ("read_file", "    1  a", "1 line"),
        ("list_files", "(no files)", "no files"),
        ("web_search", "[1] A\nu\nx\n\n[2] B\nv\ny", "2 results"),
        ("web_search", "No results.", "no results"),
        ("read_web_page", "x" * 1500, "1,500 characters"),
        ("write_file", "wrote 5 characters to a.txt", "wrote 5 characters to a.txt"),
    ],
)
def test_summaries(tool, output, summary):
    assert summarize(tool, output) == summary


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def _draw(calls, *items):
    out = io.StringIO()
    display = TurnDisplay(Console(file=out, width=100), FakeSession(*calls))
    for item in items:
        display.show(item)
    display._stop_live()
    return out.getvalue()


def _event(type, **payload):
    return Event("s", 1, type, payload, "ts")
