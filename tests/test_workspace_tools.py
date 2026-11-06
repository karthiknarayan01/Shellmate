"""list_files and read_file: output format, line ranges, and refusing paths outside the workspace."""

import pytest

from runtime.messages import ToolCall
from runtime.tools import ToolRegistry
from runtime.workspace_tools import workspace_tools


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("one\ntwo\nthree\n")
    (tmp_path / "README.md").write_text("hi\n")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref")
    return tmp_path


@pytest.fixture
def tools(workspace):
    return ToolRegistry(workspace_tools(workspace))


def test_list_files_is_recursive_sorted_and_skips_ignored_dirs(tools):
    assert _run(tools, "list_files").output == "README.md\nsrc/app.py"


def test_read_file_numbers_lines(tools):
    out = _run(tools, "read_file", path="src/app.py").output
    assert out.splitlines() == ["    1  one", "    2  two", "    3  three"]


def test_read_file_line_range(tools):
    out = _run(tools, "read_file", path="src/app.py", start_line=2, end_line=2).output
    assert out == "    2  two"


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "src/../../x"])
def test_paths_outside_the_workspace_are_refused(tools, path):
    result = _run(tools, "read_file", path=path)
    assert result.is_error and "outside the workspace" in result.output


def test_symlink_pointing_outside_is_refused(tools, workspace, tmp_path_factory):
    secret = tmp_path_factory.mktemp("elsewhere") / "secret.txt"
    secret.write_text("nope")
    (workspace / "link.txt").symlink_to(secret)
    assert "outside the workspace" in _run(tools, "read_file", path="link.txt").output


def test_write_file_creates_parent_directories(tools, workspace):
    result = _run(tools, "write_file", path="new/dir/x.txt", content="hello")
    assert not result.is_error
    assert (workspace / "new" / "dir" / "x.txt").read_text() == "hello"


def test_write_file_outside_the_workspace_is_refused(tools, workspace):
    result = _run(tools, "write_file", path="../escape.txt", content="x")
    assert result.is_error and not (workspace.parent / "escape.txt").exists()


def test_only_write_file_requires_approval(tools):
    assert [n for n in ("list_files", "read_file", "write_file") if tools.requires_approval(n)] == ["write_file"]


def test_missing_file_is_an_error_result(tools):
    assert _run(tools, "read_file", path="nope.py").is_error


def _run(tools, name, **arguments):
    return tools.run(ToolCall("c", name, arguments))
