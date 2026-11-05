"""Read-only tools over one directory (the workspace): list files and read a file.

Every path is resolved and must stay inside the workspace root; `..` or an absolute path
pointing elsewhere is refused.
"""

from __future__ import annotations

from pathlib import Path

from runtime.tools import Tool

MAX_OUTPUT_CHARS = 20_000
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", ".pytest_cache"}


def workspace_tools(root: str | Path) -> list[Tool]:
    """The tools, bound to `root`."""
    root = Path(root).resolve()
    return [
        Tool(
            name="list_files",
            description="List files under a directory of the workspace, recursively.",
            input_schema={
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Directory, default '.'"}},
                "additionalProperties": False,
            },
            fn=lambda path=".": list_files(root, path),
        ),
        Tool(
            name="read_file",
            description="Read a UTF-8 text file of the workspace. Lines are numbered from 1.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer", "minimum": 1},
                    "end_line": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            fn=lambda path, start_line=1, end_line=None: read_file(root, path, start_line, end_line),
        ),
    ]


def list_files(root: Path, path: str) -> str:
    """Relative paths of all files under `path`, sorted, skipping IGNORED_DIRS."""
    base = resolve_inside(root, path)
    if not base.is_dir():
        raise NotADirectoryError(path)
    files = sorted(
        str(p.relative_to(root))
        for p in base.rglob("*")
        if p.is_file() and not IGNORED_DIRS.intersection(p.relative_to(root).parts)
    )
    return _clip("\n".join(files) or "(no files)")


def read_file(root: Path, path: str, start_line: int, end_line: int | None) -> str:
    """The requested line range, each line prefixed with its number."""
    lines = resolve_inside(root, path).read_text(encoding="utf-8").splitlines()
    end = len(lines) if end_line is None else min(end_line, len(lines))
    numbered = (f"{n:>5}  {lines[n - 1]}" for n in range(start_line, end + 1))
    return _clip("\n".join(numbered) or "(empty range)")


def resolve_inside(root: Path, path: str) -> Path:
    """Resolve `path` against `root` (following symlinks) and refuse anything outside it."""
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise PermissionError(f"{path} is outside the workspace")
    return resolved


def _clip(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n... [{len(text) - MAX_OUTPUT_CHARS} more characters]"
