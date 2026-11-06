"""Compaction in the agent loop: when it triggers, what the model sees afterwards, and the
100-step session that has to keep its original question through repeated compactions."""

import pytest

from runtime.compaction import REQUEST_HEADER, SUMMARY_SYSTEM_PROMPT, CompactionPolicy
from runtime.events import EventLog
from runtime.messages import ToolCall
from runtime.model import ModelReply
from runtime.session import CompactionFailed, Session
from runtime.state import project
from runtime.tools import Tool, ToolRegistry
from tests.fake_model import FakeModel, Request, call, say

SMALL = CompactionPolicy(context_window=4_000)  # triggers at 3,200 tokens, keeps ~1,000


@pytest.fixture
def log(db):
    with EventLog(db) as log:
        yield log


async def test_100_step_session_compacts_and_still_answers_the_original_question(log):
    question = "What is the sum of all the numbers the counter tool returns?"
    model = LongTaskModel(steps=100)
    session = Session.create(log, model, _tools(), compaction=SMALL, max_steps=150)

    result = await session.run_turn(question)

    compactions = [e for e in log.load(session.session_id) if e.type == "compaction"]
    assert len(compactions) >= 3
    assert model.tool_steps == 100
    assert result.text == f"Answering: {question}"
    for request in model.task_requests:  # the question was in every request the model saw
        assert any(question in m.text for m in request.messages)
    assert session.state == project(log.load(session.session_id))


async def test_no_compaction_below_the_trigger(log):
    model = FakeModel([call("counter"), say("done")])
    session = Session.create(log, model, _tools(), compaction=CompactionPolicy(context_window=100_000))
    await session.run_turn("count once")
    assert "compaction" not in _types(log, session)
    assert all(r.system != SUMMARY_SYSTEM_PROMPT for r in model.requests)


async def test_compaction_replaces_old_messages_with_summary_and_request(log):
    model = FakeModel([say("first answer " + "x" * 16_000), say("SUMMARY: talked about x"), say("second")])
    session = Session.create(log, model, _tools(), compaction=SMALL)
    await session.run_turn("tell me about x")
    await session.run_turn("and now y?")

    summary_request, final_request = model.requests[1], model.requests[2]
    assert summary_request.system == SUMMARY_SYSTEM_PROMPT and summary_request.tools == []
    assert "USER: tell me about x" in summary_request.messages[0].text
    first = final_request.messages[0]
    assert first.role == "user" and "SUMMARY: talked about x" in first.text
    assert first.text.endswith(f"{REQUEST_HEADER}\nand now y?")
    assert final_request.messages[-1].text == "and now y?"  # the newest message is kept verbatim


async def test_a_tool_call_and_its_result_are_never_separated(log):
    model = LongTaskModel(steps=30)
    session = Session.create(log, model, _tools(), compaction=SMALL, max_steps=50)
    await session.run_turn("count")

    for request in model.task_requests:
        call_ids = {c.id for m in request.messages for c in m.tool_calls}
        result_ids = {m.tool_call_id for m in request.messages if m.role == "tool"}
        assert result_ids <= call_ids  # every result has its call in the same request


async def test_truncated_summary_raises_and_records_nothing(log):
    model = FakeModel([say("x" * 16_000), ModelReply(text="half a summ", truncated=True)])
    session = Session.create(log, model, _tools(), compaction=SMALL)
    await session.run_turn("long answer please")
    before = _types(log, session)

    with pytest.raises(CompactionFailed):
        await session.run_turn("next")
    assert _types(log, session) == before + ["user_message"]


async def test_resume_after_compaction_continues_from_the_compacted_history(log):
    model = FakeModel([say("x" * 16_000), say("SUMMARY"), call("counter")])
    session = Session.create(log, model, _tools(), compaction=SMALL)
    await session.run_turn("long answer please")
    with pytest.raises(AssertionError):  # FakeModel runs out: stands in for a crash mid-turn
        await session.run_turn("count")

    resumed_model = FakeModel([say("done")])
    resumed = Session.open(log, session.session_id, resumed_model, _tools(), compaction=SMALL)
    assert (await resumed.resume()).text == "done"
    assert "SUMMARY" in resumed_model.requests[0].messages[0].text


# ── helpers ──────────────────────────────────────────────────────────────────────────────


class LongTaskModel(FakeModel):
    """Calls `counter` `steps` times, then answers. Answers summary requests with a summary.

    Each counter result is long, so the history passes the trigger every few steps.
    """

    def __init__(self, steps):
        super().__init__(replies=[])
        self.steps = steps
        self.tool_steps = 0
        self.task_requests: list[Request] = []

    async def complete(self, system, messages, tools):
        request = Request(system, list(messages), list(tools))
        self.requests.append(request)
        if system == SUMMARY_SYSTEM_PROMPT:
            return ModelReply(text=f"Called counter {self.tool_steps} times so far.")
        self.task_requests.append(request)
        if self.tool_steps < self.steps:
            self.tool_steps += 1
            return ModelReply(tool_calls=(ToolCall(f"c{self.tool_steps}", "counter", {}),))
        return ModelReply(text=f"Answering: {_request_in(messages)}")


def _request_in(messages):
    """The user's request as the model can see it: the newest plain user message, or, once that
    has been compacted away, the verbatim copy at the end of the summary."""
    for m in reversed(messages):
        if m.role != "user":
            continue
        if REQUEST_HEADER in m.text:
            return m.text.split(f"{REQUEST_HEADER}\n", 1)[1]
        return m.text
    raise AssertionError("no user request visible to the model")


def _tools():
    return ToolRegistry([Tool("counter", "Returns a number with a long log.", {"type": "object"},
                              lambda: "42 " + "log line " * 150, idempotent=True)])


def _types(log, session):
    return [e.type for e in log.load(session.session_id)]
