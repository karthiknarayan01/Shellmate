"""Terminal input for the chat: prompts, history, completion, and Ctrl+C.

Two ways to read a line, behind one interface (`Terminal.ask`):

    interactive   prompt_toolkit, when stdin is a terminal: line editing, history across runs
                  (up arrow, and suggestions from history as you type), Tab-completion of slash
                  commands, Alt+Enter (or Esc then Enter) for a new line, and a bottom toolbar
    line reader   a background thread calling a read function (input() or a test's script),
                  used when stdin isn't a terminal

Ctrl+C is turned into cancelling the current task (a turn, or the wait at a prompt) rather than
an exception that would stop the event loop. At a prompt it raises Interrupted.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
from pathlib import Path
from typing import Awaitable, Callable, Iterable, TextIO

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from runtime.subscriptions import EventStream


class Interrupted(Exception):
    """Ctrl+C at a prompt (an ordinary exception, so it can't stop the event loop)."""


class Terminal:
    """Line input plus Ctrl+C handling, for one chat."""

    def __init__(self, read: Callable[[str], str] | None = None, out: TextIO | None = None):
        """`read` is a blocking function returning one line; it runs in a background thread.
        Prompts are written to `out`."""
        self._read = read or input
        self._out = out or sys.stdout
        self._lines: asyncio.Queue | None = None
        self._wanted = threading.Semaphore(0)  # one release = read one line
        self._read_outstanding = False
        self._input_closed = False
        self._current: asyncio.Task | None = None
        self._followed: EventStream | None = None
        self.toolbar: Callable[[], str] | None = None  # bottom toolbar text, if the backend has one

    @classmethod
    def interactive(cls, history_file: Path, commands: Iterable[str], **io) -> Terminal:
        """A Terminal that reads with prompt_toolkit. Use only when stdin is a terminal.

        `io` (prompt_toolkit's `input` / `output`) is for tests.
        """
        return _PromptToolkitTerminal(history_file, sorted(commands), **io)

    def catch_ctrl_c(self) -> None:
        """Make Ctrl+C cancel the current task. Not available on every platform; then it quits."""
        try:
            asyncio.get_running_loop().add_signal_handler(signal.SIGINT, self._cancel_current)
        except (NotImplementedError, RuntimeError, ValueError):
            pass

    async def cancellable(self, work: Awaitable):
        """Run `work` as the task Ctrl+C cancels."""
        task = asyncio.ensure_future(work)
        self._current = task
        try:
            return await task
        finally:
            self._current = None

    def follow(self, events: EventStream | None) -> None:
        """The event stream the display is printing, so prompts can wait for it (caught_up)."""
        self._followed = events

    async def caught_up(self) -> None:
        """Wait until the display has printed every event so far, so a prompt comes after them."""
        if self._followed is not None:
            await self._followed.drained()

    async def ask(self, prompt: str, *, chat: bool = False) -> str:
        """Show `prompt` and wait for a line. `chat` marks the main prompt (history, completion).

        Raises EOFError (Ctrl+D, or input ended) or Interrupted (Ctrl+C).
        """
        print(prompt, end="", file=self._out, flush=True)
        if self._input_closed:
            raise EOFError
        if self._lines is None:
            self._start_reader()
        if not self._read_outstanding:
            self._read_outstanding = True
            self._wanted.release()
        line = await self._lines.get()
        self._read_outstanding = False
        if isinstance(line, EOFError):
            self._input_closed = True
        if isinstance(line, BaseException):
            raise line
        return line

    def _start_reader(self) -> None:
        """One daemon thread for the whole run, reading one line per request.

        With a thread per prompt, a prompt abandoned by Ctrl+C would leave its thread waiting on
        stdin, and it would swallow the next line typed; here the outstanding read simply
        answers the next prompt.
        """
        loop = asyncio.get_running_loop()
        self._lines = asyncio.Queue()

        def read_lines() -> None:
            while True:
                self._wanted.acquire()
                try:
                    item: object = self._read("")
                except KeyboardInterrupt:
                    item = Interrupted()
                except EOFError as e:
                    item = e
                try:
                    loop.call_soon_threadsafe(self._lines.put_nowait, item)
                except RuntimeError:  # the loop has closed: nobody is listening any more
                    return
                if isinstance(item, EOFError):
                    return

        threading.Thread(target=read_lines, name="terminal-input", daemon=True).start()

    def _cancel_current(self) -> None:
        if self._current is not None:
            self._current.cancel()


class _PromptToolkitTerminal(Terminal):
    """Reads with prompt_toolkit, which runs on the same event loop."""

    def __init__(self, history_file: Path, commands: list[str], **io):
        super().__init__()
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self._chat = PromptSession(
            **io,
            history=FileHistory(str(history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SlashCommandCompleter(commands),
            complete_while_typing=True,
            key_bindings=_newline_bindings(),
            bottom_toolbar=lambda: self.toolbar() if self.toolbar else None,
        )
        self._plain = PromptSession(**io)  # approval answers: no history, no completion

    async def ask(self, prompt: str, *, chat: bool = False) -> str:
        session = self._chat if chat else self._plain
        try:
            return await session.prompt_async(ANSI(prompt))
        except KeyboardInterrupt:
            raise Interrupted() from None


def _newline_bindings() -> KeyBindings:
    """Alt+Enter (sent as Esc then Enter) inserts a new line; Enter alone submits."""
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    return bindings


class SlashCommandCompleter(Completer):
    """Completes "/..." at the start of the line from the chat's command list."""

    def __init__(self, commands: list[str]):
        self.commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for command in self.commands:
            if command.startswith(text):
                yield Completion(command, start_position=-len(text))
