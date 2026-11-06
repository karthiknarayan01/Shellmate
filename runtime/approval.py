"""Approval decisions for tools that change something.

Before a tool marked `requires_approval` runs, the session asks an Approver. The answer is
one of:

    approve               run this call
    approve_for_session   run this call, and every later call of the same tool in this session
    deny                  don't run it; the model is told it was denied, plus any feedback

A denial with feedback ("no, edit config.py instead") lets the model adjust and keeps the turn
going. A denial without feedback stops the turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from runtime.messages import ToolCall

DecisionKind = Literal["approve", "approve_for_session", "deny"]


@dataclass(frozen=True, slots=True)
class Decision:
    kind: DecisionKind
    feedback: str = ""

    @property
    def approved(self) -> bool:
        return self.kind != "deny"


class Approver(Protocol):
    def ask(self, call: ToolCall) -> Decision:
        """Decide one call. May block, e.g. while waiting for a person to answer."""
        ...


class AutoApprove:
    """Approves everything. For trusted, unattended runs only."""

    def ask(self, call: ToolCall) -> Decision:
        return Decision("approve")


class DenyAll:
    """Denies everything. The default, so a session with no approver never runs a risky tool."""

    def ask(self, call: ToolCall) -> Decision:
        return Decision("deny")
