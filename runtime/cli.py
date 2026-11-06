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
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, TextIO

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
    with EventLog(args.db) as log:
        session = open_session(log, args, read, out)
        print(f"session {session.session_id}  (resume with --resume {session.session_id})", file=out)
        if session.needs_resume:
            print("Finishing the turn that was interrupted...", file=out)
            run_and_show(session.resume, session, out)
        chat(session, read, out)
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


def open_session(log: EventLog, args: argparse.Namespace, read, out: TextIO) -> Session:
    """Create a new session, or open the one named by --resume."""
    try:
        model = make_model(args.provider, args.model)
    except (RuntimeError, ValueError) as e:
        raise SystemExit(str(e)) from None
    tools = ToolRegistry(workspace_tools(Path(args.workspace)))
    approver: Approver = AutoApprove() if args.yes else TerminalApprover(read, out)
    if args.resume is None:
        session = Session.create(log, model, tools, approver=approver)
    else:
        session_id = log.latest_session() if args.resume == "latest" else args.resume
        if session_id is None or not log.has_session(session_id):
            raise SystemExit(f"no session to resume: {args.resume}")
        session = Session.open(log, session_id, model, tools, approver=approver)
    session.observer = lambda event: show_event(event, session, out)
    return session


def chat(session: Session, read, out: TextIO) -> None:
    """The prompt loop: one turn per message until /exit, Ctrl+D or Ctrl+C at the prompt."""
    while True:
        try:
            text = read("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=out)
            return
        if not text:
            continue
        if text in EXIT_COMMANDS:
            return
        if text == RESUME_COMMAND:
            run_and_show(session.resume, session, out)
            continue
        run_and_show(lambda: session.run_turn(text), session, out)


def run_and_show(run: Callable[[], TurnResult | None], session: Session, out: TextIO) -> None:
    """Run a turn and print its answer. Turn-level errors are printed, not raised."""
    try:
        result = run()
    except KeyboardInterrupt:
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

    def __init__(self, read: Callable[[str], str], out: TextIO):
        self.read = read
        self.out = out

    def ask(self, call: ToolCall) -> Decision:
        print(f"\n{call.name} wants to run with:", file=self.out)
        for name, value in call.arguments.items():
            print(f"  {name}: {_shorten(value)}", file=self.out)
        answer = self.read("approve? [y]es / [a]lways / [n]o / or type feedback: ").strip()
        return parse_answer(answer)


def parse_answer(answer: str) -> Decision:
    lowered = answer.lower()
    if lowered in {"y", "yes"}:
        return Decision("approve")
    if lowered in {"a", "always"}:
        return Decision("approve_for_session")
    if lowered in {"", "n", "no"}:
        return Decision("deny")
    return Decision("deny", feedback=answer)


def _shorten(value: object, limit: int = MAX_SHOWN_ARGUMENT_CHARS) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    if len(text) > limit:
        return f"{text[:limit]}... ({len(text)} characters)"
    return text


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text else ""


if __name__ == "__main__":
    raise SystemExit(main())
