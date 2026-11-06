"""Kill the process mid-turn with SIGKILL, restart, and check the turn finishes with no side
effect repeated.

Each test runs a turn in a child process that kills itself at one point: inside a tool after
its side effect, while waiting for approval, or during the next model call. The parent then
opens the session from the same database, resumes, and checks the side-effect file. The tool's
side effect is appending a line to that file, which is not idempotent: running it twice would
leave two lines.
"""

import asyncio
import multiprocessing as mp
import os
import signal

from runtime.approval import AutoApprove, Decision
from runtime.events import EventLog
from runtime.session import INTERRUPTED_RESULT, Session
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, call, say


async def test_kill_inside_a_tool_after_its_side_effect(db, tmp_path):
    effects = tmp_path / "effects.txt"
    sid = _run_child_until_killed(_child_dies_inside_tool, db, effects)
    assert _lines(effects) == ["hello"]

    model = FakeModel([say("I can't confirm it was sent")])
    with EventLog(db) as log:
        session = Session.open(log, sid, model, _tools(effects))
        result = await session.resume()

    assert _lines(effects) == ["hello"]  # not appended twice
    assert result.text == "I can't confirm it was sent"
    assert model.requests[0].messages[-1].text == INTERRUPTED_RESULT


async def test_kill_while_waiting_for_approval(db, tmp_path):
    effects = tmp_path / "effects.txt"
    sid = _run_child_until_killed(_child_dies_in_approver, db, effects)
    assert _lines(effects) == []

    with EventLog(db) as log:
        session = Session.open(log, sid, FakeModel([say("sent")]), _tools(effects),
                               approver=_Approve())
        assert (await session.resume()).text == "sent"

    assert _lines(effects) == ["hello"]


async def test_kill_during_the_model_call_after_a_tool_finished(db, tmp_path):
    effects = tmp_path / "effects.txt"
    sid = _run_child_until_killed(_child_dies_in_second_model_call, db, effects)
    assert _lines(effects) == ["hello"]

    with EventLog(db) as log:
        session = Session.open(log, sid, FakeModel([say("sent")]), _tools(effects))
        assert (await session.resume()).text == "sent"

    assert _lines(effects) == ["hello"]


# ── running the child ────────────────────────────────────────────────────────────────────


def _run_child_until_killed(turn, db, effects):
    """Start a session, run the coroutine function `turn` in a child process, check it died by
    SIGKILL."""
    with EventLog(db) as log:
        sid = log.new_session()
    child = mp.get_context("spawn").Process(target=_child_main, args=(turn, db, sid, str(effects)))
    child.start()
    child.join(timeout=60)
    assert child.exitcode == -signal.SIGKILL
    return sid


def _child_main(turn, db, sid, effects):
    asyncio.run(turn(db, sid, effects))


async def _child_dies_inside_tool(db, sid, effects):
    def append_then_die(text):
        _append(effects, text)
        _die()

    tools = ToolRegistry([_append_tool(append_then_die)])
    with EventLog(db) as log:
        session = Session.open(log, sid, FakeModel([call("append_line", text="hello")]), tools,
                               approver=AutoApprove())
        await session.run_turn("append hello")


async def _child_dies_in_approver(db, sid, effects):
    class DieApprover:
        async def ask(self, call):
            _die()

    with EventLog(db) as log:
        session = Session.open(log, sid, FakeModel([call("append_line", text="hello")]),
                               _tools(effects), approver=DieApprover())
        await session.run_turn("append hello")


async def _child_dies_in_second_model_call(db, sid, effects):
    class DieOnSecondCall(FakeModel):
        async def complete(self, system, messages, tools, on_delta=None):
            if self.requests:
                _die()
            return await super().complete(system, messages, tools, on_delta)

    with EventLog(db) as log:
        session = Session.open(log, sid, DieOnSecondCall([call("append_line", text="hello")]),
                               _tools(effects), approver=AutoApprove())
        await session.run_turn("append hello")


# ── tools and helpers (module-level so the spawned child can import them) ────────────────


def _tools(effects):
    return ToolRegistry([_append_tool(lambda text: _append(effects, text) or "appended")])


def _append_tool(fn):
    return Tool(
        "append_line",
        "Append a line to the effects file.",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        fn,
        requires_approval=True,
    )


def _append(path, text):
    with open(path, "a", encoding="utf-8") as f:
        f.write(text + "\n")
        f.flush()
        os.fsync(f.fileno())


def _lines(path):
    return path.read_text().splitlines() if path.exists() else []


def _die():
    os.kill(os.getpid(), signal.SIGKILL)


class _Approve:
    async def ask(self, call):
        return Decision("approve")
