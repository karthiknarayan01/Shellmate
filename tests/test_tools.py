"""ToolRegistry: specs, and every way a call can fail coming back as an error result."""

import pytest

from runtime.messages import ToolCall
from runtime.tools import Tool, ToolRegistry, ToolResult

SCHEMA = {
    "type": "object",
    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
    "required": ["a", "b"],
    "additionalProperties": False,
}


@pytest.fixture
def registry():
    return ToolRegistry([Tool("add", "Add two numbers.", SCHEMA, lambda a, b: a + b),
                         Tool("boom", "Always fails.", {"type": "object"}, _raise)])


def test_specs_describe_every_tool(registry):
    assert [(s.name, s.description) for s in registry.specs()] == [
        ("add", "Add two numbers."), ("boom", "Always fails.")
    ]


def test_valid_call_returns_the_output_as_text(registry):
    assert registry.run(ToolCall("c", "add", {"a": 2, "b": 3})) == ToolResult("5")


@pytest.mark.parametrize(
    "call, message",
    [
        (ToolCall("c", "nope", {}), "unknown tool"),
        (ToolCall("c", "add", {"a": 1}), "invalid arguments"),
        (ToolCall("c", "add", {"a": "1", "b": 2}), "invalid arguments"),
        (ToolCall("c", "add", {"a": 1, "b": 2, "c": 3}), "invalid arguments"),
        (ToolCall("c", "boom", {}), "boom failed: ValueError: bad"),
    ],
    ids=["unknown", "missing-arg", "wrong-type", "extra-arg", "tool-raises"],
)
def test_failures_are_error_results_not_exceptions(registry, call, message):
    result = registry.run(call)
    assert result.is_error and message in result.output


def test_arguments_the_model_could_not_encode_are_an_error_result(registry):
    result = registry.run(ToolCall("c", "add", {}, arguments_error="not valid JSON: '{\"a\": '"))
    assert result.is_error and "not valid JSON" in result.output


def test_check_reports_problems_without_running(registry):
    assert registry.check(ToolCall("c", "add", {"a": 1, "b": 2})) is None
    assert registry.check(ToolCall("c", "add", {"a": 1})).is_error


def test_only_tools_marked_as_such_require_approval():
    registry = ToolRegistry([Tool("read", "", {}, str), Tool("write", "", {}, str, requires_approval=True)])
    assert registry.requires_approval("write")
    assert not registry.requires_approval("read")
    assert not registry.requires_approval("unknown")


def test_duplicate_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        ToolRegistry([Tool("x", "", {}, str), Tool("x", "", {}, str)])


def _raise():
    raise ValueError("bad")
