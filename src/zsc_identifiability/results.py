"""Typed public result objects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zsc_identifiability.numeric import Backend, Number, serialize_number
from zsc_identifiability.policy import PolicyNode


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    game_id: str | None
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicySolution:
    game_id: str
    action_class: str
    objective: str
    backend: Backend
    policy: PolicyNode
    expected_intervention_cost: Number
    residual_decision_risk: Number
    total_cost_plus_risk: Number
    expected_commitment_time: Number
    tie_breaking_record: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "action_class": self.action_class,
            "objective": self.objective,
            "backend": self.backend,
            "expected_intervention_cost": serialize_number(self.expected_intervention_cost),
            "residual_decision_risk": serialize_number(self.residual_decision_risk),
            "total_cost_plus_risk": serialize_number(self.total_cost_plus_risk),
            "expected_commitment_time": serialize_number(self.expected_commitment_time),
            "tie_breaking_record": list(self.tie_breaking_record),
            "policy": self.policy.to_dict(),
        }


@dataclass(frozen=True)
class HistoryDistributions:
    game_id: str
    by_mode: dict[str, dict[str, Number]]
    expected_cost_by_mode: dict[str, Number]
    decisions_by_history: dict[str, str]
    commitment_time_by_history: dict[str, int]
    posterior_by_history: dict[str, dict[str, Number]]


@dataclass(frozen=True)
class PolicyEvaluation:
    game_id: str
    backend: Backend
    prior_risk: Number
    residual_risk_precommitment: Number
    residual_risk_eventual: Number
    expected_intervention_cost: Number
    net_oracle_regret: Number
    dri_precommitment: Number | None
    dri_eventual: Number | None
    identification_required: bool
    decision_sufficient: bool
    identity_mutual_information_bits: float
    decision_signature_mutual_information_bits: float
    map_type_accuracy: Number
    decision_accuracy: Number | None
    expected_commitment_time: Number
    pairwise_total_variation: dict[str, Number]
    actual_policy_loss: Number

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "backend": self.backend,
            "prior_risk": serialize_number(self.prior_risk),
            "residual_risk_precommitment": serialize_number(self.residual_risk_precommitment),
            "residual_risk_eventual": serialize_number(self.residual_risk_eventual),
            "expected_intervention_cost": serialize_number(self.expected_intervention_cost),
            "net_oracle_regret": serialize_number(self.net_oracle_regret),
            "dri_precommitment": serialize_number(self.dri_precommitment),
            "dri_eventual": serialize_number(self.dri_eventual),
            "identification_required": self.identification_required,
            "decision_sufficient": self.decision_sufficient,
            "identity_mutual_information_bits": self.identity_mutual_information_bits,
            "decision_signature_mutual_information_bits": (
                self.decision_signature_mutual_information_bits
            ),
            "map_type_accuracy": serialize_number(self.map_type_accuracy),
            "decision_accuracy": serialize_number(self.decision_accuracy),
            "expected_commitment_time": serialize_number(self.expected_commitment_time),
            "pairwise_total_variation": {
                key: serialize_number(value) for key, value in self.pairwise_total_variation.items()
            },
            "actual_policy_loss": serialize_number(self.actual_policy_loss),
        }


@dataclass(frozen=True)
class FrontierPoint:
    expected_cost: Number
    residual_risk: Number
    dri: Number | None
    expected_commitment_time: Number
    policy: PolicyNode

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_cost": serialize_number(self.expected_cost),
            "residual_risk": serialize_number(self.residual_risk),
            "dri": serialize_number(self.dri),
            "expected_commitment_time": serialize_number(self.expected_commitment_time),
            "policy": self.policy.to_dict(),
        }


@dataclass(frozen=True)
class RemovedFrontierPoint:
    expected_cost: Number
    residual_risk: Number
    reason: str
    policy_signature: str


@dataclass(frozen=True)
class FrontierResult:
    game_id: str
    action_class: str
    backend: Backend
    deterministic_points: tuple[FrontierPoint, ...]
    removed_points: tuple[RemovedFrontierPoint, ...]
    convexified_envelope: tuple[FrontierPoint, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "action_class": self.action_class,
            "backend": self.backend,
            "deterministic_points": [point.to_dict() for point in self.deterministic_points],
            "removed_points": [
                {
                    "expected_cost": serialize_number(point.expected_cost),
                    "residual_risk": serialize_number(point.residual_risk),
                    "reason": point.reason,
                    "policy_signature": point.policy_signature,
                }
                for point in self.removed_points
            ],
            "convexified_envelope": [point.to_dict() for point in self.convexified_envelope],
        }


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    suite_id: str
    configuration_hash: str
    source_tree_hash: str
    python_version: str
    dependency_versions: dict[str, str]
    invoked_command: str
    generated_files: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "configuration_hash": self.configuration_hash,
            "source_tree_hash": self.source_tree_hash,
            "python_version": self.python_version,
            "dependency_versions": self.dependency_versions,
            "invoked_command": self.invoked_command,
            "generated_files": list(self.generated_files),
        }
