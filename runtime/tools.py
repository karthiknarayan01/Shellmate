"""Tool registry: the tools a session may use, and running one tool call safely.

A tool is a name, a description, a JSON schema for its arguments, and a Python function.
`ToolRegistry.run()` never raises for a bad call: an unknown tool, arguments that are not
valid JSON or fail the schema, or an exception inside the tool all come back as an error
result, which the model sees and can recover from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jsonschema

from runtime.messages import ToolCall
from runtime.model import ToolSpec


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]  # called with the validated arguments as keyword arguments

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


@dataclass(frozen=True, slots=True)
class ToolResult:
    output: str
    is_error: bool = False


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        names = [t.name for t in tools]
        duplicates = {n for n in names if names.count(n) > 1}
        if duplicates:
            raise ValueError(f"duplicate tool names: {sorted(duplicates)}")
        self._tools = {t.name: t for t in tools}

    def specs(self) -> list[ToolSpec]:
        """What the model is told about the available tools."""
        return [t.spec() for t in self._tools.values()]

    def run(self, call: ToolCall) -> ToolResult:
        """Validate and run one call. Every failure is returned as an error result."""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(f"unknown tool {call.name!r}", is_error=True)
        if call.arguments_error:
            message = f"invalid arguments for {call.name}: {call.arguments_error}"
            return ToolResult(message, is_error=True)
        try:
            jsonschema.validate(call.arguments, tool.input_schema)
        except jsonschema.ValidationError as e:
            return ToolResult(f"invalid arguments for {call.name}: {e.message}", is_error=True)
        try:
            return ToolResult(str(tool.fn(**call.arguments)))
        except Exception as e:
            return ToolResult(f"{call.name} failed: {type(e).__name__}: {e}", is_error=True)
