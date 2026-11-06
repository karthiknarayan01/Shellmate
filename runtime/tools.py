"""Tool registry: the tools a session may use, and running one tool call safely.

A tool is a name, a description, a JSON schema for its arguments, a Python function, and two
flags:
    requires_approval  ask before running it (any tool that changes something should)
    idempotent         running it twice with the same arguments leaves the same result as once,
                       so after a crash mid-run it is safe to run again

Running a call has two parts, so the session can ask for approval in between:
    check(call)          is the call well-formed? (known tool, valid JSON, matches the schema)
    await execute(call)  run the function
Neither raises for a bad call: problems come back as an error ToolResult, which the model sees
and can recover from.

A tool function may be `async def`, which is awaited on the event loop, or a plain function,
which runs in a worker thread (asyncio.to_thread) so it doesn't block the loop while it waits
on disk or network. Either way several calls can run at the same time.
"""

from __future__ import annotations

import asyncio
import inspect
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
    fn: Callable[..., Any]  # called with the validated arguments as keyword arguments
    requires_approval: bool = False
    idempotent: bool = False

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

    async def run(self, call: ToolCall) -> ToolResult:
        """check() then execute(), for callers that don't need approval in between."""
        return self.check(call) or await self.execute(call)

    def check(self, call: ToolCall) -> ToolResult | None:
        """An error result if the call can't run as given, otherwise None."""
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
        return None

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run a call that passed check(). An exception inside the tool becomes an error result."""
        fn = self._tools[call.name].fn
        try:
            if inspect.iscoroutinefunction(fn):
                output = await fn(**call.arguments)
            else:
                output = await asyncio.to_thread(fn, **call.arguments)
            return ToolResult(str(output))
        except Exception as e:
            return ToolResult(f"{call.name} failed: {type(e).__name__}: {e}", is_error=True)

    def requires_approval(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.requires_approval

    def is_idempotent(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool is not None and tool.idempotent
