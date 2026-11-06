"""Compaction: when the conversation nears the model's context window, summarize the oldest part.

Before each model step the session estimates the size of the next request. Past
`trigger_tokens` it compacts:

    messages:  [ m0 m1 ... m(k-1) | mk ... mn ]
                 summarized         kept verbatim (the most recent ~keep_tokens)
    becomes:   [ summary | mk ... mn ]

The summary is written by the same model in a separate request with no tools. The user's
current request is appended to it verbatim, so the goal of the turn survives any number of
compactions even if the summary leaves it out. The cut is never placed right before a tool
result, so a tool call and its result stay together.

Token estimate: the provider's own count from the latest reply, plus characters / 4 for
messages added since. Characters / 4 is a rough English average; it is only used for the
part the provider hasn't measured yet.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from runtime.messages import Message
from runtime.model import ToolSpec
from runtime.state import SessionState

CONTEXT_WINDOW_ENV = "AGENT_CONTEXT_WINDOW"
DEFAULT_CONTEXT_WINDOW = 128_000
CHARS_PER_TOKEN = 4
MAX_TOOL_RESULT_CHARS_IN_TRANSCRIPT = 2_000

SUMMARY_SYSTEM_PROMPT = (
    "You compress the earlier part of a conversation between a user and an AI assistant that "
    "uses tools. Write a summary the assistant can continue from without the original. Keep: "
    "what the user asked for and any constraints they gave; decisions made; facts learned from "
    "tool results (file names, values, errors) that are still relevant; what has been done and "
    "what remains. Drop pleasantries and anything superseded. Write plain prose or short bullets."
)

SUMMARY_HEADER = "Summary of the earlier conversation:"
REQUEST_HEADER = "The user's current request, verbatim:"


@dataclass(frozen=True, slots=True)
class CompactionPolicy:
    context_window: int = DEFAULT_CONTEXT_WINDOW
    trigger_ratio: float = 0.8  # compact when the next request would exceed this share
    keep_ratio: float = 0.25  # keep about this share of the window verbatim after compacting

    @property
    def trigger_tokens(self) -> int:
        return int(self.context_window * self.trigger_ratio)

    @property
    def keep_tokens(self) -> int:
        return int(self.context_window * self.keep_ratio)


def policy_from_env() -> CompactionPolicy:
    """The default policy, with the context window from AGENT_CONTEXT_WINDOW when set."""
    window = context_window_from_env()
    return CompactionPolicy() if window is None else CompactionPolicy(context_window=window)


def context_window_from_env() -> int | None:
    """AGENT_CONTEXT_WINDOW as an integer of at least 1000, or None when unset."""
    raw = os.environ.get(CONTEXT_WINDOW_ENV, "").strip()
    if not raw:
        return None
    try:
        window = int(raw)
    except ValueError:
        raise ValueError(f"{CONTEXT_WINDOW_ENV} must be an integer, got {raw!r}") from None
    if window < 1_000:
        raise ValueError(f"{CONTEXT_WINDOW_ENV} must be at least 1000, got {window}")
    return window


def estimate_tokens(state: SessionState, system: str, tools: list[ToolSpec]) -> int:
    """Estimated size of the next request: measured tokens plus an estimate for the rest."""
    if state.measured_tokens is not None:
        unmeasured = state.messages[state.measured_through :]
        return state.measured_tokens + sum(message_tokens(m) for m in unmeasured)
    fixed = len(system) + sum(
        len(t.name) + len(t.description) + len(json.dumps(t.input_schema)) for t in tools
    )
    return fixed // CHARS_PER_TOKEN + sum(message_tokens(m) for m in state.messages)


def message_tokens(message: Message) -> int:
    chars = len(message.text) + len(message.reasoning)
    for call in message.tool_calls:
        chars += len(call.name) + len(json.dumps(call.arguments))
    return chars // CHARS_PER_TOKEN + 1


def choose_cut(messages: list[Message], keep_tokens: int) -> int | None:
    """How many leading messages to summarize, or None if nothing useful can be compacted.

    Keeps the longest recent tail that fits in keep_tokens and doesn't start with a tool
    result. At least one message is always kept and at least one summarized.
    """
    cut = None
    kept = 0
    for index in range(len(messages) - 1, 0, -1):
        kept += message_tokens(messages[index])
        if kept > keep_tokens and cut is not None:
            break
        if messages[index].role != "tool":
            cut = index
    return cut


def render_transcript(messages: list[Message]) -> str:
    """The messages to summarize, as plain text for the summary request.

    Tool results are clipped: the summary needs what they showed, not every line.
    """
    lines = []
    for m in messages:
        if m.role == "user":
            lines.append(f"USER: {m.text}")
        elif m.role == "assistant":
            if m.text:
                lines.append(f"ASSISTANT: {m.text}")
            for call in m.tool_calls:
                lines.append(f"ASSISTANT CALLED {call.name}({json.dumps(call.arguments)})")
        else:
            text = m.text
            if len(text) > MAX_TOOL_RESULT_CHARS_IN_TRANSCRIPT:
                text = text[:MAX_TOOL_RESULT_CHARS_IN_TRANSCRIPT] + " ... [clipped]"
            label = "TOOL ERROR" if m.is_error else "TOOL RESULT"
            lines.append(f"{label}: {text}")
    return "\n\n".join(lines)


def summary_message_text(summary: str, current_request: str) -> str:
    """What replaces the summarized messages: the summary, then the current request verbatim."""
    return f"{SUMMARY_HEADER}\n{summary.strip()}\n\n{REQUEST_HEADER}\n{current_request}"
