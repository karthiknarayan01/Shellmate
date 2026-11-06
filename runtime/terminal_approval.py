"""The terminal's approval prompt: a panel showing what the tool will do, then a one-key answer.

    ╭─ Allow write_file? ─────────────────────────────────────╮
    │ path  config.py                                         │
    │                                                         │
    │ --- config.py                                           │
    │ +++ config.py (proposed)                                │
    │ -DEBGU = True                                           │
    │ +DEBUG = True                                           │
    ╰─────────────────────────────────────────────────────────╯
      y yes   a always allow write_file   n no   or type what to do instead
    ❯

For write_file the panel shows a diff against the file as it is now (or the first lines of a
new file), so the change can be judged before it happens.

    y / yes      approve this call
    a / always   approve this tool for the rest of the session
    n / no / ""  deny, and stop the turn
    anything     deny, and send the text to the model as feedback
"""

from __future__ import annotations

import difflib
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from runtime.approval import Decision
from runtime.messages import ToolCall
from runtime.terminal_input import Terminal
from runtime.workspace_tools import resolve_inside

MAX_ARGUMENT_CHARS = 300
MAX_PREVIEW_LINES = 60


class TerminalApprover:
    def __init__(self, terminal: Terminal, console: Console, workspace: Path | None = None):
        """`workspace` lets write_file calls show a diff against the file on disk."""
        self.terminal = terminal
        self.console = console
        self.workspace = workspace.resolve() if workspace is not None else None

    async def ask(self, call: ToolCall) -> Decision:
        """Show the panel once the display has caught up, then read the answer."""
        await self.terminal.caught_up()
        self.console.print()
        self.console.print(Panel(
            self.describe(call),
            title=Text(f"Allow {call.name}?", style="bold yellow"),
            title_align="left",
            border_style="yellow",
            expand=False,
        ))
        options = Text("  ")
        for key, meaning in (("y", "yes"), ("a", f"always allow {call.name}"), ("n", "no")):
            options.append(key, style="bold")
            options.append(f" {meaning}   ", style="dim")
        options.append("or type what to do instead", style="dim")
        self.console.print(options)
        answer = await self.terminal.ask(_prompt_marker(self.console))
        return parse_answer(answer.strip())

    def describe(self, call: ToolCall) -> RenderableType:
        """The panel's body: the arguments, plus a diff for write_file."""
        arguments = Table.grid(padding=(0, 2))
        arguments.add_column(style="dim")
        arguments.add_column()
        for name, value in call.arguments.items():
            if call.name == "write_file" and name == "content":
                continue
            arguments.add_row(name, _shorten(value))
        parts: list[RenderableType] = [arguments]
        if call.name == "write_file":
            preview = self._write_preview(call.arguments.get("path", ""), call.arguments.get("content", ""))
            parts += [Text(""), preview]
        return Group(*parts)

    def _write_preview(self, path: str, content: str) -> RenderableType:
        """A unified diff against the current file, or the start of a new one."""
        current = None
        if self.workspace is not None:
            try:
                target = resolve_inside(self.workspace, path)
                if target.is_file():
                    current = target.read_text(encoding="utf-8")
            except (OSError, ValueError, UnicodeDecodeError):
                current = None
        if current is None:
            lines = content.splitlines()
            body = "\n".join(lines[:MAX_PREVIEW_LINES])
            more = f"\n… {len(lines) - MAX_PREVIEW_LINES} more lines" if len(lines) > MAX_PREVIEW_LINES else ""
            count = f"{len(lines)} line" + ("" if len(lines) == 1 else "s")
            return Group(Text(f"new file, {count}", style="green"), Text(body + more))
        if current == content:
            return Text("no change: the file already has this content", style="dim")
        diff = list(difflib.unified_diff(
            current.splitlines(), content.splitlines(),
            fromfile=path, tofile=f"{path} (proposed)", lineterm="",
        ))
        if len(diff) > MAX_PREVIEW_LINES:
            diff = diff[:MAX_PREVIEW_LINES] + [f"… {len(diff) - MAX_PREVIEW_LINES} more diff lines"]
        return Syntax("\n".join(diff), "diff", theme="ansi_dark", background_color="default")


def parse_answer(answer: str) -> Decision:
    lowered = answer.lower()
    if lowered in {"y", "yes"}:
        return Decision("approve")
    if lowered in {"a", "always"}:
        return Decision("approve_for_session")
    if lowered in {"", "n", "no"}:
        return Decision("deny")
    return Decision("deny", feedback=answer)


def _prompt_marker(console: Console) -> str:
    """"❯ " in yellow, as ANSI text the input backends can display."""
    with console.capture() as capture:
        console.print(Text("❯ ", style="bold yellow"), end="")
    return capture.get()


def _shorten(value: object, limit: int = MAX_ARGUMENT_CHARS) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) > limit:
        return f"{text[:limit]}… ({len(text)} characters)"
    return text
