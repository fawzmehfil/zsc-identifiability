"""Schemas and typed results for matched Phase 3 benchmark populations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number, parse_rational

SignalTarget = Literal["response", "subtype", "null"]
EvidenceSlot = Literal["0", "1", "post_commitment", "never"]


class FrozenBenchmarkModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class BinaryCellSpec(FrozenBenchmarkModel):
    cell_id: str
    passive_evidence_slot: EvidenceSlot
    active_signal_target: Literal["response", "null"]
    matching_group: str | None = None
    intervention_cost: str | None = None
    shared_response: bool = False


class FactorizedCellSpec(FrozenBenchmarkModel):
    cell_id: str
    passive_signal_target: SignalTarget
    active_signal_target: SignalTarget
    matching_group: str | None = None
    intervention_cost: str | None = None
    distractor_steps: int | None = Field(default=None, ge=0, le=2)


class SweepSpec(FrozenBenchmarkModel):
    sweep_id: str
    cell_id: str
    parameter: Literal[
        "reliability",
        "intervention_cost",
        "evidence_slot",
        "distractor_steps",
        "active_signal_target",
    ]
    values: tuple[str, ...]


class BinaryFamilySpec(FrozenBenchmarkModel):
    kind: Literal["binary_role_allocation"]
    family_id: str
    reliability: str
    intervention_cost: str
    mismatch_loss: str
    cells: tuple[BinaryCellSpec, ...]
    sweeps: tuple[SweepSpec, ...] = ()
    generate_symmetries: bool = True

    @model_validator(mode="after")
    def validate_parameters(self) -> BinaryFamilySpec:
        _validate_probability(self.reliability)
        _validate_nonnegative(self.intervention_cost, "intervention_cost")
        _validate_nonnegative(self.mismatch_loss, "mismatch_loss")
        _require_unique("binary cell", tuple(cell.cell_id for cell in self.cells))
        _require_unique("binary sweep", tuple(sweep.sweep_id for sweep in self.sweeps))
        _validate_sweep_targets(self.cells, self.sweeps)
        _validate_sweep_values(self.sweeps, family_kind=self.kind)
        return self


class FactorizedFamilySpec(FrozenBenchmarkModel):
    kind: Literal["factorized_identity_memory"]
    family_id: str
    reliability: str
    intervention_cost: str
    mismatch_loss: str
    distractor_steps: int = Field(default=1, ge=0, le=2)
    cells: tuple[FactorizedCellSpec, ...]
    sweeps: tuple[SweepSpec, ...] = ()
    generate_symmetries: bool = True

    @model_validator(mode="after")
    def validate_parameters(self) -> FactorizedFamilySpec:
        _validate_probability(self.reliability)
        _validate_nonnegative(self.intervention_cost, "intervention_cost")
        _validate_nonnegative(self.mismatch_loss, "mismatch_loss")
        _require_unique("factorized cell", tuple(cell.cell_id for cell in self.cells))
        _require_unique("factorized sweep", tuple(sweep.sweep_id for sweep in self.sweeps))
        _validate_sweep_targets(self.cells, self.sweeps)
        _validate_sweep_values(self.sweeps, family_kind=self.kind)
        return self


FamilySpec = Annotated[BinaryFamilySpec | FactorizedFamilySpec, Field(discriminator="kind")]


class MetricRuleSpec(FrozenBenchmarkModel):
    metric: str
    relation: Literal["equal", "difference"]
    tolerance: str = "0"
    expected_difference: str | None = None
    role: Literal["control", "treatment"]

    @model_validator(mode="after")
    def validate_rule(self) -> MetricRuleSpec:
        _validate_nonnegative(self.tolerance, "metric tolerance")
        if self.relation == "difference" and self.expected_difference is None:
            raise ValueError("difference metric rule requires expected_difference")
        if self.expected_difference is not None:
            parse_rational(self.expected_difference)
        return self


class MatchingContractSpec(FrozenBenchmarkModel):
    contract_id: str
    left_population_id: str
    right_population_id: str
    require_structural_match: bool = True
    require_passive_history_match: bool = False
    require_divergence_profile_match: bool = False
    metric_rules: tuple[MetricRuleSpec, ...]
    sampled: bool = False


class SampleAuditSpec(FrozenBenchmarkModel):
    episodes_per_mode: int = Field(default=10_000, ge=100)
    bootstrap_resamples: int = Field(default=2_000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.0, lt=1.0)
    seed: int = 1729
    return_margin: str = "1"
    lobp_margin_nats: float = Field(default=0.01, ge=0.0)
    determinant_margin: float = Field(default=0.01, ge=0.0)
    divergence_time_margin: float = Field(default=0.02, ge=0.0)
    dri_margin: float = Field(default=0.02, ge=0.0)

    @model_validator(mode="after")
    def validate_margins(self) -> SampleAuditSpec:
        _validate_nonnegative(self.return_margin, "return_margin")
        return self


class MatchedBenchmarkSuite(FrozenBenchmarkModel):
    schema_version: Literal[1]
    suite_id: str
    base_team_return: str
    families: tuple[FamilySpec, ...]
    matching_contracts: tuple[MatchingContractSpec, ...]
    sample_audit: SampleAuditSpec = Field(default_factory=SampleAuditSpec)

    @model_validator(mode="after")
    def validate_suite(self) -> MatchedBenchmarkSuite:
        base = parse_rational(self.base_team_return)
        if base <= 0:
            raise ValueError("base_team_return must be positive")
        _require_unique("family", tuple(family.family_id for family in self.families))
        _require_unique(
            "matching contract", tuple(contract.contract_id for contract in self.matching_contracts)
        )
        maximum = Fraction(0)
        for family in self.families:
            maximum = max(
                maximum,
                parse_rational(family.intervention_cost) + parse_rational(family.mismatch_loss),
            )
            for cell in family.cells:
                if cell.intervention_cost is not None:
                    _validate_nonnegative(cell.intervention_cost, "cell intervention_cost")
                    maximum = max(
                        maximum,
                        parse_rational(cell.intervention_cost)
                        + parse_rational(family.mismatch_loss),
                    )
            for sweep in family.sweeps:
                if sweep.parameter == "intervention_cost":
                    maximum = max(
                        maximum,
                        max(parse_rational(value) for value in sweep.values)
                        + parse_rational(family.mismatch_loss),
                    )
        if base < maximum:
            raise ValueError("base_team_return is below a declared cost-plus-loss maximum")
        return self


class GeneratedPopulationDescriptor(FrozenBenchmarkModel):
    schema_version: Literal[1] = 1
    population_id: str
    family_id: str
    family_kind: Literal["binary_role_allocation", "factorized_identity_memory"]
    cell_id: str
    matching_group: str | None
    symmetry_id: str
    base_team_return: str
    response_signature_by_mode: dict[str, str]
    best_response_event_features: dict[str, tuple[str, ...]]
    passive_reference_actions: tuple[str, ...]
    commitment_states: tuple[str, ...]
    intended_treatments: dict[str, str]
    matched_nuisances: dict[str, str]
    runtime_visible_fields: tuple[str, ...]
    analytical_expectations: dict[str, str]
    game_hash: str
    suite_hash: str


@dataclass(frozen=True)
class GeneratedPopulation:
    descriptor: GeneratedPopulationDescriptor
    game: FiniteConventionGame


@dataclass(frozen=True)
class GeneratedBenchmarkSet:
    suite: MatchedBenchmarkSuite
    populations: tuple[GeneratedPopulation, ...]
    suite_hash: str

    def by_id(self) -> dict[str, GeneratedPopulation]:
        return {item.descriptor.population_id: item for item in self.populations}


@dataclass(frozen=True)
class PopulationMetrics:
    population_id: str
    backend: Backend
    metric_scope: str
    estimator_type: str
    applicability_flags: dict[str, bool]
    values: dict[str, Any]
    per_mode: dict[str, dict[str, Any]]
    response_confusion_matrix: dict[str, dict[str, Number]]
    brdiv_matrices: dict[str, Any]
    prefix_tv_curves: dict[str, tuple[Number, ...]]
    divergence_threshold_steps: dict[str, dict[str, int | None]]
    deterministic_edp: dict[str, int | None] | None
    passive_policy: dict[str, Any]
    active_policy: dict[str, Any]
    information_policy: dict[str, Any]
    reference_policy: dict[str, Any]
    active_frontier: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


@dataclass(frozen=True)
class AuditItem:
    metric: str
    left: Number | float | str | None
    right: Number | float | str | None
    difference: Number | float | None
    tolerance: Number | float | None
    status: Literal["pass", "fail", "not_applicable"]
    reason: str
    role: Literal["control", "treatment"]


@dataclass(frozen=True)
class MatchingAudit:
    contract_id: str
    left_population_id: str
    right_population_id: str
    passed: bool
    structural_checks: dict[str, bool]
    passive_history_match: bool | None
    divergence_profile_match: bool | None
    items: tuple[AuditItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


@dataclass(frozen=True)
class ShortcutAudit:
    population_id: str
    passed: bool
    best_fixed_risk: Number
    evidence_blind_risk: Number
    memoryless_risk: Number
    history_aware_risk: Number
    postcommit_leak_free: bool
    public_state_leak_free: bool
    identifier_leak_free: bool
    universal_response_blocked: bool
    valueless_probe_tie_break_ok: bool
    checks: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


@dataclass(frozen=True)
class BenchmarkRunManifest:
    schema_version: int
    suite_id: str
    scientific_verdict: Literal["continue", "redesign", "stop"]
    implementation_passed: bool
    scientific_audit_passed: bool
    configuration_hash: str
    source_tree_hash: str
    python_version: str
    dependency_versions: dict[str, str]
    invoked_command: str
    rng_configuration: dict[str, int | float | str]
    generated_files: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


def load_benchmark_suite_file(path: str | Path) -> MatchedBenchmarkSuite:
    with Path(path).open(encoding="utf-8") as handle:
        return MatchedBenchmarkSuite.model_validate(json.load(handle))


def _validate_probability(value: str) -> None:
    probability = parse_rational(value)
    if probability < Fraction(1, 2) or probability > 1:
        raise ValueError("signal reliability must lie in [1/2, 1]")


def _validate_nonnegative(value: str, label: str) -> None:
    if parse_rational(value) < 0:
        raise ValueError(f"{label} must be nonnegative")


def _require_unique(label: str, values: tuple[str, ...]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} identifier")


def _validate_sweep_targets(cells: tuple[Any, ...], sweeps: tuple[SweepSpec, ...]) -> None:
    cell_ids = {cell.cell_id for cell in cells}
    for sweep in sweeps:
        if sweep.cell_id not in cell_ids:
            raise ValueError(f"sweep {sweep.sweep_id!r} references an unknown cell")
        if not sweep.values:
            raise ValueError(f"sweep {sweep.sweep_id!r} has no values")


def _validate_sweep_values(
    sweeps: tuple[SweepSpec, ...],
    family_kind: Literal["binary_role_allocation", "factorized_identity_memory"],
) -> None:
    for sweep in sweeps:
        if sweep.parameter == "reliability":
            for value in sweep.values:
                _validate_probability(value)
        elif sweep.parameter == "intervention_cost":
            for value in sweep.values:
                _validate_nonnegative(value, "sweep intervention_cost")
        elif sweep.parameter == "evidence_slot":
            if family_kind != "binary_role_allocation":
                raise ValueError("evidence_slot sweep is invalid for factorized family")
            if any(value not in {"0", "1", "post_commitment", "never"} for value in sweep.values):
                raise ValueError("invalid evidence_slot sweep value")
        elif sweep.parameter == "distractor_steps":
            if family_kind != "factorized_identity_memory":
                raise ValueError("distractor_steps sweep is invalid for binary family")
            try:
                distractors = tuple(int(value) for value in sweep.values)
            except ValueError as exc:
                raise ValueError("distractor_steps sweep values must be integers") from exc
            if any(value < 0 or value > 2 for value in distractors):
                raise ValueError("distractor_steps sweep values must lie in [0, 2]")
        elif sweep.parameter == "active_signal_target":
            allowed = (
                {"response", "null"}
                if family_kind == "binary_role_allocation"
                else {"response", "subtype", "null"}
            )
            if any(value not in allowed for value in sweep.values):
                raise ValueError("invalid active_signal_target sweep value")


def _serialize(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    if isinstance(value, float | str | int | bool) or value is None:
        return value
    if isinstance(value, tuple | list):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {name: _serialize(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    raise TypeError(f"cannot serialize {type(value)!r}")
