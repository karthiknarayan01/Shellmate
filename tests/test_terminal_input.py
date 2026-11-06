"""Terminal input: the line reader, and the prompt_toolkit backend driven through a pipe."""

import asyncio
import io
import threading

import pytest
from prompt_toolkit.document import Document
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from runtime.terminal_input import Interrupted, SlashCommandCompleter, Terminal


async def test_a_cancelled_prompt_does_not_swallow_the_next_line():
    """Ctrl+C at a prompt leaves its read outstanding; that read answers the next prompt."""
    release = threading.Event()
    reads = []

    def read(_prompt):
        reads.append(1)
        release.wait(timeout=5)
        return "hello"

    terminal = Terminal(read, io.StringIO())
    first = asyncio.create_task(terminal.ask("> "))
    await asyncio.sleep(0.05)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    second = asyncio.create_task(terminal.ask("> "))
    release.set()
    assert await second == "hello"
    assert len(reads) == 1


async def test_lines_are_read_only_when_asked_for():
    reads = []
    terminal = Terminal(lambda prompt: reads.append(1) or "y", io.StringIO())
    assert await terminal.ask("> ") == "y"
    await asyncio.sleep(0.05)
    assert len(reads) == 1


async def test_ctrl_c_and_end_of_input_from_the_line_reader():
    answers = iter([KeyboardInterrupt, EOFError])

    def read(_prompt):
        raise next(answers)()

    terminal = Terminal(read, io.StringIO())
    with pytest.raises(Interrupted):
        await terminal.ask("> ")
    with pytest.raises(EOFError):
        await terminal.ask("> ")
    with pytest.raises(EOFError):  # stays closed
        await terminal.ask("> ")


async def test_interactive_prompt_reads_a_line_and_keeps_history(tmp_path):
    history = tmp_path / "state" / "history"
    with create_pipe_input() as pipe:
        terminal = Terminal.interactive(history, ["/help"], input=pipe, output=DummyOutput())
        pipe.send_text("first message\r")
        assert await terminal.ask("❯ ", chat=True) == "first message"
    assert "first message" in history.read_text()


async def test_alt_enter_inserts_a_new_line(tmp_path):
    with create_pipe_input() as pipe:
        terminal = Terminal.interactive(tmp_path / "h", [], input=pipe, output=DummyOutput())
        pipe.send_text("line one\x1b\rline two\r")  # Esc+Enter, then Enter
        assert await terminal.ask("❯ ", chat=True) == "line one\nline two"


async def test_ctrl_c_at_an_interactive_prompt_is_interrupted(tmp_path):
    with create_pipe_input() as pipe:
        terminal = Terminal.interactive(tmp_path / "h", [], input=pipe, output=DummyOutput())
        pipe.send_text("\x03")
        with pytest.raises(Interrupted):
            await terminal.ask("❯ ", chat=True)


async def test_ctrl_d_at_an_interactive_prompt_ends_input(tmp_path):
    with create_pipe_input() as pipe:
        terminal = Terminal.interactive(tmp_path / "h", [], input=pipe, output=DummyOutput())
        pipe.send_text("\x04")
        with pytest.raises(EOFError):
            await terminal.ask("❯ ", chat=True)


def test_slash_commands_complete_only_at_the_start():
    completer = SlashCommandCompleter(["/help", "/new", "/exit"])

    def complete(text):
        return [c.text for c in completer.get_completions(Document(text), None)]

    assert complete("/") == ["/help", "/new", "/exit"]
    assert complete("/n") == ["/new"]
    assert complete("hello /n") == []
    assert complete("/new now") == []
