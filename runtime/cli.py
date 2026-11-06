"""Command-line chat: `agent-runtime`.

    agent-runtime                              new session, Anthropic, workspace = current dir
    agent-runtime --provider deepseek          use DeepSeek
    agent-runtime --resume latest              continue the most recent session
    agent-runtime --resume <session id>        continue a specific session
    agent-runtime --yes                        approve every tool without asking
    agent-runtime --no-web                     turn off the web search tools

On start it migrates the database, shows a welcome panel, opens or creates the session, and
finishes any turn a crash left unfinished. Then it reads one message per prompt and runs a turn
for it, showing the work as it happens (terminal_display.py) and asking before tools that
change something (terminal_approval.py). After each turn a dim line shows what it cost.

    /help     commands and keys          /new      start a fresh session
    /status   session, model, context    /resume   retry a turn that failed on an API error
    /clear    clear the screen           /exit     quit (also Ctrl+D)
    Ctrl+C    cancel the running turn; at the prompt, quit

The chat runs on one asyncio event loop. For each message it starts two tasks:

    turn task     session.run_turn(text): the agent loop
    display task  follows session.subscribe() and draws the turn as it happens

The final answer is the last thing the display draws; then control returns to the prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Awaitable, Callable, TextIO

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from runtime.approval import Approver, AutoApprove
from runtime.compaction import estimate_tokens
from runtime.events import EventLog
from runtime.migrate import migrate
from runtime.model import ModelClient, ModelError
from runtime.prompts import system_prompt
from runtime.providers import DEFAULT_MODELS, PROVIDERS, make_model
from runtime.session import (
    CompactionFailed,
    MaxStepsReached,
    ReplyTruncated,
    Session,
    TurnInProgress,
    TurnResult,
)
from runtime.terminal_approval import TerminalApprover
from runtime.terminal_display import TurnDisplay, TurnStats
from runtime.terminal_input import Interrupted, Terminal
from runtime.tools import ToolRegistry
from runtime.web_tools import web_tools
from runtime.workspace_tools import workspace_tools

DEFAULT_DB = "agent.db"
HISTORY_FILE = Path.home() / ".local" / "state" / "agent-runtime" / "history"
COMMANDS = {
    "/help": "show commands and keys",
    "/new": "start a fresh session",
    "/status": "session, model and context use",
    "/resume": "retry a turn that failed on a model API error",
    "/clear": "clear the screen",
    "/exit": "quit (also /quit, Ctrl+D)",
}
EXIT_COMMANDS = {"/exit", "/quit"}


def main(
    argv: list[str] | None = None,
    read: Callable[[str], str] | None = None,
    out: TextIO | None = None,
) -> int:
    """Entry point. Tests pass `read` and `out`; otherwise the real terminal is used."""
    args = parse_args(argv)
    migrate(args.db)
    out = out or sys.stdout
    console = Console(file=out, highlight=False)
    try:
        return asyncio.run(run_chat(args, console, read))
    except KeyboardInterrupt:  # only where Ctrl+C can't be turned into a task cancel
        console.print()
        return 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="agent-runtime", description="Chat with a tool-using agent.")
    parser.add_argument("--db", default=DEFAULT_DB, help=f"SQLite file for sessions (default {DEFAULT_DB})")
    parser.add_argument("--workspace", default=".", help="directory the file tools work in (default .)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS), default="anthropic")
    parser.add_argument("--model", help="model name (default: the provider's default)")
    parser.add_argument("--resume", metavar="SESSION", help="session id to continue, or 'latest'")
    parser.add_argument("--yes", action="store_true", help="approve every tool call without asking")
    parser.add_argument("--no-web", action="store_true", help="don't give the agent web search")
    return parser.parse_args(argv)


@dataclass
class Chat:
    """Everything a chat needs to make sessions and draw them."""

    args: argparse.Namespace
    console: Console
    terminal: Terminal
    log: EventLog
    model: ModelClient
    model_name: str
    tools: ToolRegistry
    options: dict
    session: Session | None = None

    def new_session(self) -> Session:
        self.session = Session.create(self.log, self.model, self.tools, **self.options)
        return self.session

    def open_session(self, session_id: str) -> Session:
        self.session = Session.open(self.log, session_id, self.model, self.tools, **self.options)
        return self.session

    def context_share(self) -> float:
        """How full the context window is, as the next request would send it."""
        session = self.session
        used = estimate_tokens(session.state, session.system_prompt, session.tools.specs())
        return used / session.compaction.context_window


async def run_chat(args: argparse.Namespace, console: Console, read: Callable[[str], str] | None) -> int:
    """Set up the chat, finish an interrupted turn if there is one, then take messages."""
    with EventLog(args.db) as log:
        chat = make_chat(args, console, read, log)
        chat.terminal.catch_ctrl_c()
        show_welcome(chat)
        if chat.session.needs_resume:
            console.print(Text("Finishing the turn that was interrupted…", style="yellow"))
            await run_turn(chat, chat.session.resume)
        await prompt_loop(chat)
    return 0


def make_chat(args: argparse.Namespace, console: Console, read, log: EventLog) -> Chat:
    """The model, tools, approver and session for these arguments."""
    try:
        model = make_model(args.provider, args.model)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from None
    web = not args.no_web
    workspace = Path(args.workspace)
    tools = ToolRegistry(workspace_tools(workspace) + (web_tools() if web else []))
    chat = Chat(
        args=args,
        console=console,
        terminal=_make_terminal(read, console),
        log=log,
        model=model,
        model_name=args.model or DEFAULT_MODELS[args.provider],
        tools=tools,
        options={},
    )
    approver: Approver = AutoApprove() if args.yes else TerminalApprover(chat.terminal, console, workspace)
    chat.options = {"approver": approver, "system_prompt": system_prompt(date.today(), web)}
    chat.terminal.toolbar = lambda: _toolbar(chat)
    if args.resume is None:
        chat.new_session()
    else:
        session_id = log.latest_session() if args.resume == "latest" else args.resume
        if session_id is None or not log.has_session(session_id):
            raise SystemExit(f"no session to resume: {args.resume}")
        chat.open_session(session_id)
    return chat


def show_welcome(chat: Chat) -> None:
    """The panel at the start: what this session is and how to use it."""
    args, session = chat.args, chat.session
    rows = Table.grid(padding=(0, 2))
    rows.add_column(style="dim")
    rows.add_column()
    rows.add_row("model", f"{chat.model_name} ({args.provider})")
    rows.add_row("workspace", str(Path(args.workspace).resolve()))
    if args.resume is None:
        rows.add_row("session", f"{session.session_id}  (new)")
    else:
        rows.add_row("session", f"{session.session_id}  (resumed, {len(session.state.messages)} messages)")
    rows.add_row("web search", "off" if args.no_web else "on")
    rows.add_row("approvals", "automatic (--yes)" if args.yes else "ask before changing anything")
    tips = Text("/help for commands · Ctrl+C cancels a turn · Alt+Enter for a new line", style="dim")
    chat.console.print(Panel.fit(
        rows, title=Text("agent-runtime", style="bold cyan"), title_align="left",
        border_style="cyan", subtitle=tips, subtitle_align="left",
    ))
    chat.console.print(Text(f"resume later with --resume {session.session_id}", style="dim"))


async def prompt_loop(chat: Chat) -> None:
    """One turn per message until /exit, Ctrl+D or Ctrl+C at the prompt."""
    while True:
        try:
            text = (await chat.terminal.cancellable(chat.terminal.ask(_prompt(chat), chat=True))).strip()
        except (EOFError, Interrupted, asyncio.CancelledError):
            chat.console.print()
            return
        if not text:
            continue
        if text.startswith("/"):
            if not await run_command(chat, text):
                return
            continue
        await run_turn(chat, lambda: chat.session.run_turn(text))


async def run_command(chat: Chat, text: str) -> bool:
    """Handle a slash command. Returns False when the chat should end."""
    command = text.split()[0].lower()
    console = chat.console
    if command in EXIT_COMMANDS:
        return False
    if command == "/help":
        table = Table.grid(padding=(0, 3))
        table.add_column(style="bold cyan")
        table.add_column()
        for name, meaning in COMMANDS.items():
            table.add_row(name, meaning)
        table.add_row("", "")
        table.add_row("Ctrl+C", "cancel the running turn; at the prompt, quit")
        table.add_row("Alt+Enter", "new line in a message (or Esc, then Enter)")
        table.add_row("↑ / ↓", "earlier messages")
        console.print(table)
    elif command == "/new":
        session = chat.new_session()
        console.print(Text(f"New session {session.session_id}", style="cyan"))
    elif command == "/status":
        show_status(chat)
    elif command == "/resume":
        if chat.session.needs_resume:
            await run_turn(chat, chat.session.resume)
        else:
            console.print(Text("Nothing to resume: the last turn finished.", style="dim"))
    elif command == "/clear":
        console.clear()
    else:
        console.print(Text(f"Unknown command {command}. Type /help for the list.", style="yellow"))
    return True


async def run_turn(chat: Chat, run: Callable[[], Awaitable[TurnResult | None]]) -> None:
    """Run a turn while a display task draws its events; then print how it ended and its cost.

    The outcome note is printed after the display has finished, so it always comes last.
    """
    session, console, terminal = chat.session, chat.console, chat.terminal
    note: Text | None = None
    display = TurnDisplay(console, session)
    async with session.subscribe() as events:
        drawing = asyncio.create_task(display.follow(events))
        terminal.follow(events)
        try:
            result = await terminal.cancellable(run())
            if result is not None and result.denied_call is not None:
                note = Text(f"Stopped: you denied {result.denied_call.name}.", style="yellow")
        except (asyncio.CancelledError, Interrupted, EOFError):
            session.cancel()
            note = Text("Turn cancelled.", style="yellow")
        except (MaxStepsReached, ReplyTruncated, CompactionFailed) as e:
            note = Text(f"Turn stopped: {e}", style="red")
        except ModelError as e:
            note = Text(f"{e}\nThe turn is kept: type /resume to retry it.", style="red")
        except TurnInProgress:
            note = Text("The last turn didn't finish: type /resume to retry it.", style="yellow")
        finally:
            terminal.follow(None)
            events.close()
            await drawing
    if note is not None:
        console.print()
        console.print(note)
    console.print(_cost_line(chat, display.stats))


def show_status(chat: Chat) -> None:
    session = chat.session
    rows = Table.grid(padding=(0, 2))
    rows.add_column(style="dim")
    rows.add_column()
    rows.add_row("session", session.session_id)
    rows.add_row("model", f"{chat.model_name} ({chat.args.provider})")
    rows.add_row("messages", str(len(session.state.messages)))
    rows.add_row("context", f"{chat.context_share():.0%} of {session.compaction.context_window:,} tokens")
    rows.add_row("tools", ", ".join(spec.name for spec in session.tools.specs()))
    chat.console.print(rows)


def _cost_line(chat: Chat, stats: TurnStats) -> Text:
    """model · tokens in / out · context use · time, dim, after each turn."""
    parts = [chat.model_name]
    if stats.input_tokens or stats.output_tokens:
        parts.append(f"{_k(stats.input_tokens)} in / {_k(stats.output_tokens)} out")
    parts.append(f"context {chat.context_share():.0%}")
    parts.append(f"{stats.seconds:.1f}s")
    return Text("  " + " · ".join(parts), style="dim")


def _toolbar(chat: Chat) -> str:
    """The prompt's bottom toolbar (interactive terminals only)."""
    if chat.session is None:
        return ""
    return f" {chat.model_name} · session {chat.session.session_id[:8]} · context {chat.context_share():.0%} · /help"


def _prompt(chat: Chat) -> str:
    with chat.console.capture() as capture:
        chat.console.print()
        chat.console.print(Text("❯ ", style="bold cyan"), end="")
    return capture.get()


def _make_terminal(read, console: Console) -> Terminal:
    """prompt_toolkit on a real terminal; a line reader for tests and pipes."""
    if read is None and sys.stdin.isatty() and console.is_terminal:
        terminal = Terminal.interactive(HISTORY_FILE, COMMANDS.keys() | EXIT_COMMANDS)
    else:
        terminal = Terminal(read or input, console.file)
    return terminal


def _k(tokens: int) -> str:
    return f"{tokens / 1000:.1f}k" if tokens >= 1000 else str(tokens)


if __name__ == "__main__":
    raise SystemExit(main())
