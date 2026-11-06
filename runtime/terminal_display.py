"""Rendering a turn in the terminal with rich, from the session's live event stream.

    ⠋ Thinking… 2s                                   spinner while waiting for the model
    ● read_file  path=config.py                      a tool call starts
      ⎿  12 lines  0.1s                              its result: a one-line summary
    ● write_file  path=config.py  content=DEBUG = …
      ⎿  ✗ write_file failed: PermissionError …      an error, in red
    Fixed: `DEBGU` → `DEBUG`.                        the answer, rendered as markdown while
                                                     it streams in

On a terminal, the spinner and the streaming answer redraw in place. Elsewhere (a pipe, a file,
the tests) rich draws nothing transient, so the output is the same lines as plain text.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner
from rich.text import Text

from runtime.events import Event
from runtime.model import StreamDelta
from runtime.session import Session
from runtime.subscriptions import EventStream

MAX_ARGUMENT_CHARS = 60
MAX_SUMMARY_CHARS = 100


@dataclass
class TurnStats:
    """What one turn cost, for the line printed after it."""

    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0


class TurnDisplay:
    """Draws one turn. `follow()` is the display task: it runs until the event stream closes."""

    def __init__(self, console: Console, session: Session):
        self.console = console
        self.session = session
        self.stats = TurnStats()
        self._started = time.monotonic()
        self._live: Live | None = None
        self._spinner: Spinner | None = None
        self._answer = ""  # the streamed text of the reply being generated
        self._running: dict[str, float] = {}  # call id -> start time
        self._parallel = False  # several calls are running at once: label each result
        self._thinking = False

    async def follow(self, events: EventStream) -> None:
        """Show each item as it arrives; the spinner runs whenever the turn is waiting."""
        self._show_spinner("Thinking")
        try:
            async for item in events:
                self.show(item)
        finally:
            self._stop_live()
            self.stats.seconds = time.monotonic() - self._started

    def show(self, item: Event | StreamDelta) -> None:
        if isinstance(item, StreamDelta):
            self._show_delta(item)
            return
        handler = getattr(self, f"_on_{item.type}", None)
        if handler is not None:
            handler(item)

    # ── streamed reply ──────────────────────────────────────────────────────────────────

    def _show_delta(self, delta: StreamDelta) -> None:
        if delta.kind == "reasoning":
            if not self._answer and not self._thinking:
                self._thinking = True
                self._show_spinner("Thinking deeply")
            return
        if not self._answer:
            self._stop_live()
            self.console.print()
            self._live = Live(console=self.console, refresh_per_second=12, transient=False)
            self._live.start()
        self._answer += delta.text
        self._live.update(Markdown(self._answer))

    def _on_assistant_message(self, event: Event) -> None:
        usage = event.payload.get("usage") or {}
        self.stats.input_tokens += usage.get("input_tokens", 0)
        self.stats.output_tokens += usage.get("output_tokens", 0)
        text = event.payload.get("text", "")
        if self._answer:
            self._stop_live()  # the streamed markdown stays on screen as the final render
        elif text:
            self._stop_live()
            self.console.print()
            self.console.print(Markdown(text))
        self._answer = ""
        self._thinking = False
        if event.payload.get("tool_calls"):
            self._show_spinner("Working")

    # ── tool calls ──────────────────────────────────────────────────────────────────────

    def _on_tool_started(self, event: Event, again: bool = False) -> None:
        call = self.session.state.tool_calls[event.payload["call_id"]].call
        self._stop_live()
        line = Text("● ", style="cyan")
        line.append(call.name, style="bold cyan")
        for name, value in call.arguments.items():
            line.append(f"  {name}=", style="dim")
            line.append(_argument(value))
        if again:
            line.append("  (again, after a crash)", style="yellow")
        self.console.print(line)
        self._running[call.id] = time.monotonic()
        self._parallel = self._parallel or len(self._running) > 1
        self._show_running()

    def _on_tool_restarted(self, event: Event) -> None:
        self._on_tool_started(event, again=True)

    def _on_tool_finished(self, event: Event) -> None:
        call_id = event.payload["call_id"]
        call = self.session.state.tool_calls[call_id].call
        seconds = time.monotonic() - self._running.pop(call_id, time.monotonic())
        self._stop_live()
        line = Text("  ⎿  ", style="dim")
        if self._parallel:  # results arrive in any order: say which call this is
            line.append(f"{call.name}: ", style="cyan")
        if event.payload.get("is_error"):
            error = _first_line(event.payload["output"], MAX_SUMMARY_CHARS)
            line.append(f"✗ {error}", style="red")
        else:
            line.append(summarize(call.name, event.payload["output"]), style="dim")
        if seconds >= 0.05:
            line.append(f"  {seconds:.1f}s", style="dim")
        self.console.print(line)
        if not self._running:
            self._parallel = False
        self._show_running()

    def _on_tool_skipped(self, event: Event) -> None:
        call = self.session.state.tool_calls[event.payload["call_id"]].call
        self._stop_live()
        line = Text("● ", style="yellow")
        line.append(call.name, style="bold yellow")
        reason = _first_line(event.payload["reason"], MAX_SUMMARY_CHARS)
        line.append(f"  skipped: {reason}", style="yellow")
        self.console.print(line)

    def _on_approval_requested(self, event: Event) -> None:
        """Stop the spinner: the approval panel is about to be drawn and needs the screen.

        The approver waits until the display has handled this event before drawing the panel.
        """
        self._stop_live()

    def _on_approval_decided(self, event: Event) -> None:
        if event.payload["decision"] == "approve_for_session":
            call = self.session.state.tool_calls[event.payload["call_id"]].call
            message = f"  {call.name} is allowed for the rest of this session"
            self.console.print(Text(message, style="dim"))
        self._show_spinner("Working")

    def _on_compaction(self, event: Event) -> None:
        self._stop_live()
        self.console.print(Text(
            f"✻ Compacted {event.payload['replaced_count']} earlier messages to stay within the "
            "context window", style="magenta"))
        self._show_spinner("Thinking")

    # ── spinner ─────────────────────────────────────────────────────────────────────────

    def _show_running(self) -> None:
        """A spinner while tools run, or "Thinking" once they're all done."""
        if not self._running:
            self._show_spinner("Thinking")
            return
        names = {self.session.state.tool_calls[i].call.name for i in self._running}
        if len(self._running) == 1:
            self._show_spinner(f"Running {next(iter(names))}")
        else:
            self._show_spinner(f"Running {len(self._running)} tools")

    def _show_spinner(self, label: str) -> None:
        self._stop_live()
        self._spinner = Spinner("dots", style="cyan")
        started = time.monotonic()
        spinner = self._spinner

        class Ticking:
            def __rich__(self) -> RenderableType:
                elapsed = time.monotonic() - started
                spinner.update(text=Text(f"{label}… {elapsed:.0f}s", style="dim"))
                return Group(spinner)

        self._live = Live(Ticking(), console=self.console, refresh_per_second=10, transient=True)
        self._live.start()

    def _stop_live(self) -> None:
        if self._live is not None:
            self._live.stop()
            # On a terminal rich ends a kept render with a newline; on a plain file it doesn't.
            if not self._live.transient and not self.console.is_terminal:
                self.console.line()
            self._live = None


def summarize(tool: str, output: str) -> str:
    """A one-line summary of a successful tool result."""
    lines = output.splitlines()
    if tool == "read_file":
        return _count(len(lines), "line")
    if tool == "list_files":
        return "no files" if output == "(no files)" else _count(len(lines), "file")
    if tool == "web_search":
        count = sum(1 for line in lines if line.startswith("[") and "] " in line)
        return "no results" if output == "No results." else _count(count, "result")
    if tool == "read_web_page":
        return f"{len(output):,} characters"
    return _first_line(output, MAX_SUMMARY_CHARS) or "done"


def _count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def _argument(value: object) -> str:
    """An argument for the tool line: short values as they are, multi-line text by its size."""
    if isinstance(value, str) and "\n" in value:
        return f"⟨{_count(len(value.splitlines()), 'line')}⟩"
    return _shorten(value, MAX_ARGUMENT_CHARS)


def _shorten(value: object, limit: int) -> str:
    text = value if isinstance(value, str) else repr(value)
    text = text.replace("\n", "⏎ ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _first_line(text: str, limit: int) -> str:
    return _shorten(text.splitlines()[0], limit) if text else ""
