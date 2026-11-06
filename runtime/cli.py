"""Command-line chat: `agent-runtime`.

    agent-runtime                              new session, Anthropic, workspace = current dir
    agent-runtime --provider deepseek          use DeepSeek
    agent-runtime --resume latest              continue the most recent session
    agent-runtime --resume <session id>        continue a specific session
    agent-runtime --yes                        approve every tool without asking

On start it migrates the database, opens or creates the session, and finishes any turn a crash
left unfinished. Then it reads one message per prompt and runs a turn for it. Tools that change
something ask in the terminal first.

    Ctrl+C during a turn   cancel that turn
    /resume                retry a turn that failed on a model API error
    /exit, Ctrl+D, Ctrl+C  quit (at the prompt)

The session id is printed so it can be resumed later.

The chat runs on one asyncio event loop. Terminal input is read by a single background thread
(see Terminal), and Ctrl+C cancels whichever task is running: the turn, or the prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
import threading
from pathlib import Path
from typing import Awaitable, Callable, TextIO

from runtime.approval import Approver, AutoApprove, Decision
from runtime.events import Event, EventLog
from runtime.messages import ToolCall
from runtime.migrate import migrate
from runtime.model import ModelError
from runtime.providers import PROVIDERS, make_model
from runtime.session import (
    CompactionFailed,
    MaxStepsReached,
    ReplyTruncated,
    Session,
    TurnInProgress,
    TurnResult,
)
from runtime.tools import ToolRegistry
from runtime.workspace_tools import workspace_tools

DEFAULT_DB = "agent.db"
EXIT_COMMANDS = {"/exit", "/quit"}
RESUME_COMMAND = "/resume"
MAX_SHOWN_ARGUMENT_CHARS = 200


def main(
    argv: list[str] | None = None,
    read: Callable[[str], str] = input,
    out: TextIO = sys.stdout,
) -> int:
    """Entry point. `read` and `out` are the terminal; tests pass their own."""
    args = parse_args(argv)
    migrate(args.db)
    try:
        return asyncio.run(run_chat(args, Terminal(read, out), out))
    except KeyboardInterrupt:  # only where Ctrl+C can't be turned into a task cancel
        print(file=out)
        return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-runtime", description="Chat with a tool-using agent.")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file for sessions (default {DEFAULT_DB})")
    parser.add_argument("--workspace", default=".", help="directory the file tools work in (default .)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="anthropic")
    parser.add_argument("--model", help="model name (default: the provider's default)")
    parser.add_argument("--resume", metavar="SESSION", help="session id to continue, or 'latest'")
    parser.add_argument("--yes", action="store_true", help="approve every tool call without asking")
    return parser.parse_args(argv)


async def run_chat(args: argparse.Namespace, terminal: Terminal, out: TextIO) -> int:
    """Open the session, finish an interrupted turn if there is one, then chat."""
    terminal.catch_ctrl_c()
    with EventLog(args.db) as log:
        session = open_session(log, args, terminal, out)
        print(f"session {session.session_id}  (resume with --resume {session.session_id})", file=out)
        if session.needs_resume:
            print("Finishing the turn that was interrupted...", file=out)
            await run_and_show(session.resume, session, terminal, out)
        await chat(session, terminal, out)
    return 0


def open_session(log: EventLog, args: argparse.Namespace, terminal: Terminal, out: TextIO) -> Session:
    """Create a new session, or open the one named by --resume."""
    try:
        model = make_model(args.provider, args.model)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from None
    tools = ToolRegistry(workspace_tools(Path(args.workspace)))
    approver: Approver = AutoApprove() if args.yes else TerminalApprover(terminal, out)
    if args.resume is None:
        session = Session.create(log, model, tools, approver=approver)
    else:
        session_id = log.latest_session() if args.resume == "latest" else args.resume
        if session_id is None or not log.has_session(session_id):
            raise SystemExit(f"no session to resume: {args.resume}")
        session = Session.open(log, session_id, model, tools, approver=approver)
    session.observer = lambda event: show_event(event, session, out)
    return session


async def chat(session: Session, terminal: Terminal, out: TextIO) -> None:
    """The prompt loop: one turn per message until /exit, Ctrl+D or Ctrl+C at the prompt."""
    while True:
        try:
            text = (await terminal.cancellable(terminal.ask("\nyou> "))).strip()
        except (EOFError, Interrupted, asyncio.CancelledError):
            print(file=out)
            return
        if not text:
            continue
        if text in EXIT_COMMANDS:
            return
        if text == RESUME_COMMAND:
            await run_and_show(session.resume, session, terminal, out)
            continue
        await run_and_show(lambda: session.run_turn(text), session, terminal, out)


async def run_and_show(
    run: Callable[[], Awaitable[TurnResult | None]],
    session: Session,
    terminal: Terminal,
    out: TextIO,
) -> None:
    """Run a turn and print its answer. Turn-level errors are printed, not raised."""
    try:
        result = await terminal.cancellable(run())
    except (asyncio.CancelledError, Interrupted, EOFError):
        session.cancel()
        print("\n[turn cancelled]", file=out)
        return
    except (MaxStepsReached, ReplyTruncated, CompactionFailed) as e:
        print(f"\n[turn stopped: {e}]", file=out)
        return
    except ModelError as e:
        print(f"\n[{e}]\n[the turn is kept; type {RESUME_COMMAND} to retry it]", file=out)
        return
    except TurnInProgress:
        print(f"\n[the last turn didn't finish; type {RESUME_COMMAND} to retry it]", file=out)
        return
    if result is None:
        return
    if result.text:
        print(f"\nassistant> {result.text}", file=out)
    if result.denied_call is not None:
        print(f"[stopped: you denied {result.denied_call.name}]", file=out)


def show_event(event: Event, session: Session, out: TextIO) -> None:
    """One line of progress for the events worth showing while a turn runs."""
    if event.type == "tool_started":
        call = session.state.tool_calls[event.payload["call_id"]].call
        arguments = ", ".join(f"{k}={_shorten(v, 60)}" for k, v in call.arguments.items())
        print(f"  -> {call.name}({arguments})", file=out)
    elif event.type == "tool_finished" and event.payload.get("is_error"):
        print(f"  ! {_first_line(event.payload['output'])}", file=out)
    elif event.type == "tool_skipped":
        print(f"  - skipped: {_first_line(event.payload['reason'])}", file=out)
    elif event.type == "compaction":
        print(f"  [compacted {event.payload['replaced_count']} earlier messages]", file=out)


class TerminalApprover:
    """Asks in the terminal before a tool that changes something runs.

        y / yes      approve this call
        a / always   approve this tool for the rest of the session
        n / no / ""  deny, and stop the turn
        anything     deny, and send the text to the model as feedback
    """

    def __init__(self, terminal: Terminal, out: TextIO):
        self.terminal = terminal
        self.out = out

    async def ask(self, call: ToolCall) -> Decision:
        print(f"\n{call.name} wants to run with:", file=self.out)
        for name, value in call.arguments.items():
            print(f"  {name}: {_shorten(value)}", file=self.out)
        answer = await self.terminal.ask("approve? [y]es / [a]lways / [n]o / or type feedback: ")
        return parse_answer(answer.strip())


def parse_answer(answer: str) -> Decision:
    lowered = answer.lower()
    if lowered in {"y", "yes"}:
        return Decision("approve")
    if lowered in {"a", "always"}:
        return Decision("approve_for_session")
    if lowered in {"", "n", "no"}:
        return Decision("deny")
    return Decision("deny", feedback=answer)


class Interrupted(Exception):
    """Ctrl+C arrived through the input reader (an ordinary exception, so it can't stop the loop)."""


class Terminal:
    """Line input and Ctrl+C handling for the event loop.

    One daemon thread reads lines for the whole run, one line per request: ask() requests a line
    (unless a request is still outstanding) and awaits it from a queue. With a thread per prompt,
    a prompt abandoned by Ctrl+C would leave its thread waiting on stdin, and it would swallow
    the next line typed; here the outstanding read simply answers the next prompt. Ctrl+C is
    turned into cancelling the current task (a turn, or the wait at a prompt) instead of an
    exception that would stop the whole loop.
    """

    def __init__(self, read: Callable[[str], str], out: TextIO):
        self._read = read
        self._out = out
        self._lines: asyncio.Queue | None = None
        self._wanted = threading.Semaphore(0)  # one release = read one line
        self._read_outstanding = False
        self._input_closed = False
        self._current: asyncio.Task | None = None

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

    async def ask(self, prompt: str) -> str:
        """Show `prompt` and wait for the next line. Raises EOFError or Interrupted."""
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


def _shorten(value: object, limit: int = MAX_SHOWN_ARGUMENT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    if len(text) > limit:
        return f"{text[:limit]}... ({len(text)} characters)"
    return text


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


if __name__ == "__main__":
    raise SystemExit(main())
