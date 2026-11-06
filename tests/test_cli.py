"""The command-line chat, driven with scripted input and a fake model."""

import asyncio
import io
import threading

import pytest

from runtime import cli
from runtime.cli import Terminal, TerminalApprover, main, parse_answer
from runtime.approval import Decision
from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.migrate import migrate
from runtime.model import ModelError, ModelReply
from tests.fake_model import FakeModel, call, say


def test_chat_runs_a_turn_and_prints_the_answer(tmp_path, monkeypatch):
    _use_model(monkeypatch, FakeModel([say("hello there")]))
    out = _run(tmp_path, [], ["hi"])
    assert "assistant> hello there" in out
    assert "session " in out


def test_fresh_database_is_migrated_on_start(tmp_path, monkeypatch):
    _use_model(monkeypatch, FakeModel([]))
    _run(tmp_path, [], [])
    with EventLog(str(tmp_path / "agent.db")) as log:
        assert log.latest_session() is not None


def test_resume_latest_continues_the_conversation(tmp_path, monkeypatch):
    _use_model(monkeypatch, FakeModel([say("one")]))
    _run(tmp_path, [], ["first"])

    model = FakeModel([say("two")])
    _use_model(monkeypatch, model)
    out = _run(tmp_path, ["--resume", "latest"], ["second"])
    assert "assistant> two" in out
    assert [m.text for m in model.requests[0].messages] == ["first", "one", "second"]


def test_unknown_session_is_reported(tmp_path, monkeypatch):
    _use_model(monkeypatch, FakeModel([]))
    with pytest.raises(SystemExit, match="no session to resume"):
        _run(tmp_path, ["--resume", "nope"], [])


def test_interrupted_turn_is_finished_on_start(tmp_path, monkeypatch):
    db = str(tmp_path / "agent.db")
    migrate(db)
    with EventLog(db) as log:
        sid = log.new_session()
        log.append(sid, "user_message", {"text": "hi"})  # crashed before the model replied

    _use_model(monkeypatch, FakeModel([say("sorry for the wait")]))
    out = _run(tmp_path, ["--resume", sid], [])
    assert "Finishing the turn that was interrupted" in out
    assert "assistant> sorry for the wait" in out


def test_write_asks_for_approval_and_runs_on_yes(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _use_model(monkeypatch, FakeModel([call("write_file", path="a.txt", content="hi"), say("written")]))
    out = _run(tmp_path, ["--workspace", str(workspace)], ["write a.txt", "y"])

    assert "write_file wants to run with:" in out and "path: a.txt" in out
    assert "-> write_file(path=a.txt, content=hi)" in out
    assert (workspace / "a.txt").read_text() == "hi"


def test_denied_write_never_runs(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _use_model(monkeypatch, FakeModel([call("write_file", path="a.txt", content="hi")]))
    out = _run(tmp_path, ["--workspace", str(workspace)], ["write a.txt", "n"])

    assert "[stopped: you denied write_file]" in out
    assert not (workspace / "a.txt").exists()


def test_progress_lines_come_before_the_approval_prompt(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    narrate_and_write = ModelReply(
        text="I'll create the file.",
        tool_calls=(ToolCall("c1", "write_file", {"path": "a.txt", "content": "hi"}),),
    )
    _use_model(monkeypatch, FakeModel([narrate_and_write, say("written")]))
    out = _run(tmp_path, ["--workspace", str(workspace)], ["write a.txt", "y"])

    narration = out.index("assistant> I'll create the file.")
    prompt = out.index("write_file wants to run with:")
    started = out.index("-> write_file(")
    answer = out.index("assistant> written")
    assert narration < prompt < started < answer


def test_streamed_answer_is_printed_once(tmp_path, monkeypatch):
    _use_model(monkeypatch, FakeModel([say("one two three four five six seven")]))
    out = _run(tmp_path, [], ["count"])
    assert out.count("one two three four five six seven") == 1
    assert "assistant> one two three four five six seven\n" in out


def test_thinking_is_shown_once_while_the_model_reasons(tmp_path, monkeypatch):
    reply = ModelReply(text="42", reasoning="let me think about this carefully for a moment")
    _use_model(monkeypatch, FakeModel([reply]))
    out = _run(tmp_path, [], ["what is the answer?"])
    assert out.count("(thinking...)") == 1
    assert out.index("(thinking...)") < out.index("assistant> 42")
    assert "carefully" not in out  # the reasoning itself isn't printed


def test_web_tools_and_a_dated_prompt_are_given_to_the_model(tmp_path, monkeypatch):
    model = FakeModel([say("ok")])
    _use_model(monkeypatch, model)
    _run(tmp_path, [], ["hi"])
    request = model.requests[0]
    assert {"web_search", "read_web_page"} <= {t.name for t in request.tools}
    assert "Today's date is" in request.system and "web_search" in request.system


def test_no_web_flag_removes_the_web_tools(tmp_path, monkeypatch):
    model = FakeModel([say("ok")])
    _use_model(monkeypatch, model)
    _run(tmp_path, ["--no-web"], ["hi"])
    request = model.requests[0]
    assert not {"web_search", "read_web_page"} & {t.name for t in request.tools}
    assert "web_search" not in request.system


def test_keyless_web_search_is_mentioned_only_without_a_key(tmp_path, monkeypatch):
    _use_model(monkeypatch, FakeModel([]))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert "keyless quota" in _run(tmp_path, [], [])
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    assert "keyless quota" not in _run(tmp_path, ["--resume", "latest"], [])


def test_yes_flag_approves_without_asking(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _use_model(monkeypatch, FakeModel([call("write_file", path="a.txt", content="hi"), say("done")]))
    out = _run(tmp_path, ["--workspace", str(workspace), "--yes"], ["write a.txt"])

    assert "wants to run" not in out
    assert (workspace / "a.txt").read_text() == "hi"


def test_ctrl_c_during_a_turn_cancels_it_and_the_chat_goes_on(tmp_path, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _use_model(monkeypatch, FakeModel([call("write_file", path="a.txt", content="hi"), say("next answer")]))
    out = _run(tmp_path, ["--workspace", str(workspace)], ["write a.txt", KeyboardInterrupt, "something else"])

    assert "[turn cancelled]" in out
    assert "assistant> next answer" in out
    assert not (workspace / "a.txt").exists()


def test_turn_errors_are_printed_not_raised(tmp_path, monkeypatch):
    model = FakeModel([call("list_files", call_id=f"c{i}") for i in range(25)])
    _use_model(monkeypatch, model)
    out = _run(tmp_path, ["--workspace", str(tmp_path)], ["loop forever"])
    assert "[turn stopped: no final answer after 20 steps]" in out


def test_missing_api_key_is_a_clean_exit(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY is not set"):
        _run(tmp_path, [], [])


def test_model_api_error_keeps_the_turn_and_resume_retries_it(tmp_path, monkeypatch):
    model = FlakyModel(fail_times=1, replies=[say("worked this time")])
    _use_model(monkeypatch, model)
    out = _run(tmp_path, [], ["hi", "something else", "/resume"])

    assert "[Anthropic API error: overloaded]" in out
    assert "type /resume to retry it" in out
    assert "the last turn didn't finish" in out  # "something else" was refused
    assert "assistant> worked this time" in out


@pytest.mark.parametrize("command", ["/exit", "/quit"])
def test_exit_commands(tmp_path, monkeypatch, command):
    _use_model(monkeypatch, FakeModel([]))
    _run(tmp_path, [], [command, "never read"])


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


async def test_approver_prompt_shortens_long_arguments():
    out = io.StringIO()
    approver = TerminalApprover(Terminal(lambda prompt: "y", out), out)
    decision = await approver.ask(ToolCall("c1", "write_file", {"path": "a.txt", "content": "x" * 1_000}))
    assert decision == Decision("approve")
    assert "(1000 characters)" in out.getvalue()


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


# ── helpers ──────────────────────────────────────────────────────────────────────────────


class FlakyModel(FakeModel):
    """Raises ModelError for the first `fail_times` requests, then follows its script."""

    def __init__(self, fail_times, replies):
        super().__init__(replies=replies)
        self.fail_times = fail_times

    async def complete(self, system, messages, tools, on_delta=None):
        if self.fail_times:
            self.fail_times -= 1
            raise ModelError("Anthropic API error: overloaded")
        return await super().complete(system, messages, tools, on_delta)


def _use_model(monkeypatch, model):
    monkeypatch.setattr(cli, "make_model", lambda provider, model_name=None: model)


def _run(tmp_path, args, inputs):
    """Run main() with scripted prompt answers. A KeyboardInterrupt entry is raised at that prompt."""
    answers = iter(inputs)

    def read(prompt):
        answer = next(answers, EOFError)
        if answer in (EOFError, KeyboardInterrupt):
            raise answer()
        return answer

    out = io.StringIO()
    main(["--db", str(tmp_path / "agent.db"), *args], read=read, out=out)
    return out.getvalue()
