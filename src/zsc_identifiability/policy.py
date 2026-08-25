"""Immutable conditional policy trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from zsc_identifiability.numeric import Number, serialize_number


@dataclass(frozen=True)
class PolicyBranch:
    next_state: str
    observation: str
    probability: Number
    expected_immediate_cost: Number
    child: PolicyNode

    def to_dict(self) -> dict[str, Any]:
        return {
            "next_state": self.next_state,
            "observation": self.observation,
            "probability": serialize_number(self.probability),
            "expected_immediate_cost": serialize_number(self.expected_immediate_cost),
            "child": self.child.to_dict(),
        }


@dataclass(frozen=True)
class PolicyNode:
    kind: Literal["commit", "act"]
    time: int
    state: str
    belief: tuple[Number, ...]
    decision: str | None = None
    action: str | None = None
    branches: tuple[PolicyBranch, ...] = ()

    def signature(self) -> str:
        if self.kind == "commit":
            return f"commit:{self.decision}"
        children = ",".join(
            f"{branch.next_state}/{branch.observation}:{branch.child.signature()}"
            for branch in self.branches
        )
        return f"act:{self.action}[{children}]"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "time": self.time,
            "state": self.state,
            "belief": [serialize_number(value) for value in self.belief],
        }
        if self.kind == "commit":
            result["decision"] = self.decision
        else:
            result["action"] = self.action
            result["branches"] = [branch.to_dict() for branch in self.branches]
        return result


def policy_from_dict(data: dict[str, Any]) -> PolicyNode:
    """Deserialize a policy tree; numeric strings remain exact fractions."""
    from fractions import Fraction

    belief = tuple(Fraction(str(value)) for value in data["belief"])
    if data["kind"] == "commit":
        return PolicyNode(
            kind="commit",
            time=int(data["time"]),
            state=str(data["state"]),
            belief=belief,
            decision=str(data["decision"]),
        )
    branches = tuple(
        PolicyBranch(
            next_state=str(item["next_state"]),
            observation=str(item["observation"]),
            probability=Fraction(str(item["probability"])),
            expected_immediate_cost=Fraction(str(item["expected_immediate_cost"])),
            child=policy_from_dict(item["child"]),
        )
        for item in data["branches"]
    )
    return PolicyNode(
        kind="act",
        time=int(data["time"]),
        state=str(data["state"]),
        belief=belief,
        action=str(data["action"]),
        branches=branches,
    )
