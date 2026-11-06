"""The approval panel: what it shows (arguments, a diff for write_file) and how answers map."""

import io

import pytest
from rich.console import Console

from runtime.approval import Decision
from runtime.messages import ToolCall
from runtime.terminal_approval import TerminalApprover, parse_answer
from runtime.terminal_input import Terminal


@pytest.mark.parametrize(
    "answer, decision",
    [
        ("y", Decision("approve")),
        ("YES", Decision("approve")),
        ("a", Decision("approve_for_session")),
        ("always", Decision("approve_for_session")),
        ("", Decision("deny")),
        ("n", Decision("deny")),
        ("use b.txt instead", Decision("deny", feedback="use b.txt instead")),
    ],
)
def test_answers_map_to_decisions(answer, decision):
    assert parse_answer(answer) == decision


async def test_panel_shows_a_diff_against_the_current_file(tmp_path):
    (tmp_path / "config.py").write_text("DEBGU = True\nPORT = 8080\n")
    out = await _ask(tmp_path, ToolCall("c1", "write_file",
                                        {"path": "config.py", "content": "DEBUG = True\nPORT = 8080\n"}))
    assert "Allow write_file?" in out
    assert "-DEBGU = True" in out and "+DEBUG = True" in out
    assert " PORT = 8080" in out
    assert "a always allow write_file" in out


async def test_panel_previews_a_new_file(tmp_path):
    out = await _ask(tmp_path, ToolCall("c1", "write_file", {"path": "new.txt", "content": "one\ntwo\n"}))
    assert "new file, 2 lines" in out and "one" in out and "two" in out


async def test_panel_says_when_nothing_would_change(tmp_path):
    (tmp_path / "same.txt").write_text("same")
    out = await _ask(tmp_path, ToolCall("c1", "write_file", {"path": "same.txt", "content": "same"}))
    assert "no change" in out


async def test_a_path_outside_the_workspace_is_previewed_as_new_not_read(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("CONTENTS-OUTSIDE-THE-WORKSPACE")
    out = await _ask(tmp_path, ToolCall("c1", "write_file", {"path": "../outside.txt", "content": "x"}))
    assert "CONTENTS-OUTSIDE-THE-WORKSPACE" not in out
    assert "new file" in out


async def test_other_tools_show_their_arguments_shortened(tmp_path):
    out = await _ask(tmp_path, ToolCall("c1", "send", {"to": "sam", "body": "x" * 1_000}))
    assert "Allow send?" in out and "sam" in out and "(1000 characters)" in out


async def _ask(workspace, call):
    out = io.StringIO()
    console = Console(file=out, width=100)
    approver = TerminalApprover(Terminal(lambda prompt: "y", out), console, workspace)
    assert await approver.ask(call) == Decision("approve")
    return out.getvalue()
