"""Schemas and typed results for the Stage 4 learned-agent audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from zsc_identifiability.benchmark_models import GeneratedPopulation
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import parse_rational

LearningMethod = Literal[
    "mlp_ppo",
    "gru_ppo_passive",
    "gru_ppo_active",
    "odits_style",
    "pace_aux",
    "pace_style",
    "talents_style",
    "tom_selector_style",
    "csp_style_reconnaissance",
]
SplitName = Literal["train", "validation", "test"]
EvaluationMode = Literal["greedy", "stochastic"]


class FrozenLearningModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PartnerProfileSpec(FrozenLearningModel):
    profile_id: str
    reliability: str
    nuisance_probability: str

    @model_validator(mode="after")
    def validate_profile(self) -> PartnerProfileSpec:
        reliability = parse_rational(self.reliability)
        nuisance = parse_rational(self.nuisance_probability)
        if reliability < parse_rational("1/2") or reliability > 1:
            raise ValueError("profile reliability must lie in [1/2, 1]")
        if nuisance < 0 or nuisance > 1:
            raise ValueError("nuisance_probability must lie in [0, 1]")
        return self


class PartnerSplitSpec(FrozenLearningModel):
    train: tuple[PartnerProfileSpec, ...]
    validation: tuple[PartnerProfileSpec, ...]
    test: tuple[PartnerProfileSpec, ...]

    @model_validator(mode="after")
    def validate_splits(self) -> PartnerSplitSpec:
        if not self.train or not self.validation or not self.test:
            raise ValueError("every partner split must be nonempty")
        profiles = tuple(
            item for split in (self.train, self.validation, self.test) for item in split
        )
        identifiers = tuple(item.profile_id for item in profiles)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("partner profile identifiers must be unique across splits")
        dynamics = tuple((item.reliability, item.nuisance_probability) for item in profiles)
        if len(dynamics) != len(set(dynamics)):
            raise ValueError("partner profile dynamics must be disjoint across splits")
        return self


class PPOConfig(FrozenLearningModel):
    hidden_size: int = Field(default=64, ge=8)
    learning_rate: float = Field(default=3e-4, gt=0)
    clip_ratio: float = Field(default=0.2, gt=0, lt=1)
    value_coefficient: float = Field(default=0.5, ge=0)
    entropy_coefficient: float = Field(default=0.01, ge=0)
    gae_lambda: float = Field(default=0.95, ge=0, le=1)
    gamma: float = Field(default=1.0, ge=0, le=1)
    max_gradient_norm: float = Field(default=0.5, gt=0)
    optimization_epochs: int = Field(default=4, ge=1)
    transitions_per_update: int = Field(default=4096, ge=32)
    minibatch_size: int = Field(default=512, ge=8)
    auxiliary_coefficient: float = Field(default=1.0, ge=0)
    pace_bonus_initial: float = Field(default=0.5, ge=0)
    pace_bonus_decay_fraction: float = Field(default=0.8, gt=0, le=1)
    latent_dimension: int = Field(default=4, ge=2)
    kl_coefficient: float = Field(default=0.1, ge=0)


class LearningMethodSpec(FrozenLearningModel):
    method_id: LearningMethod
    enabled: bool = True
    config: PPOConfig = Field(default_factory=PPOConfig)


class TrainingBudgetSpec(FrozenLearningModel):
    smoke_transitions: int = Field(default=20_000, ge=1)
    development_transitions: int = Field(default=250_000, ge=1)
    confirmatory_transitions: int = Field(default=500_000, ge=1)
    rescue_transitions: int = Field(default=2_000_000, ge=1)
    checkpoint_interval: int = Field(default=50_000, ge=1)
    num_envs: int = Field(default=256, ge=1)
    development_seeds: tuple[int, ...] = (101, 102, 103)
    confirmatory_seeds: tuple[int, ...] = tuple(range(2001, 2011))
    device: Literal["cpu", "mps", "auto"] = "cpu"

    @model_validator(mode="after")
    def validate_budget(self) -> TrainingBudgetSpec:
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("duplicate development seed")
        if len(set(self.confirmatory_seeds)) != len(self.confirmatory_seeds):
            raise ValueError("duplicate confirmatory seed")
        if set(self.development_seeds) & set(self.confirmatory_seeds):
            raise ValueError("development and confirmatory seeds must be disjoint")
        return self


class StatisticalSpec(FrozenLearningModel):
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    bootstrap_seed: int = 1729


class SymmetryAuditSpec(FrozenLearningModel):
    cell_id: str
    symmetry_id: str


class LearningComparisonSpec(FrozenLearningModel):
    comparison_id: str
    left_cell: str
    right_cell: str
    intended_treatment: str


class LearningAuditSuite(FrozenLearningModel):
    schema_version: Literal[1]
    suite_id: str
    source_benchmark_suite: str
    cells: tuple[str, ...]
    specialist_cells: tuple[str, ...]
    reconnaissance_cells: tuple[str, ...]
    profiles: PartnerSplitSpec
    methods: tuple[LearningMethodSpec, ...]
    budget: TrainingBudgetSpec = Field(default_factory=TrainingBudgetSpec)
    statistics: StatisticalSpec = Field(default_factory=StatisticalSpec)
    symmetry_audits: tuple[SymmetryAuditSpec, ...] = ()
    comparisons: tuple[LearningComparisonSpec, ...] = ()
    base_team_return: str = "100"
    loss_scale: str = "40"

    @model_validator(mode="after")
    def validate_suite(self) -> LearningAuditSuite:
        if not self.cells:
            raise ValueError("learning audit must declare at least one cell")
        if len(self.cells) != len(set(self.cells)):
            raise ValueError("duplicate learning cell")
        if not set(self.specialist_cells).issubset(self.cells):
            raise ValueError("specialist cells must be a subset of cells")
        if not set(self.reconnaissance_cells).issubset(self.cells):
            raise ValueError("reconnaissance cells must be a subset of cells")
        if not {item.cell_id for item in self.symmetry_audits}.issubset(self.cells):
            raise ValueError("symmetry-audit cells must be a subset of cells")
        symmetry_keys = tuple((item.cell_id, item.symmetry_id) for item in self.symmetry_audits)
        if len(symmetry_keys) != len(set(symmetry_keys)):
            raise ValueError("duplicate symmetry audit")
        comparison_ids = tuple(item.comparison_id for item in self.comparisons)
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("duplicate learning comparison")
        comparison_cells = {
            cell
            for comparison in self.comparisons
            for cell in (comparison.left_cell, comparison.right_cell)
        }
        if not comparison_cells.issubset(self.cells):
            raise ValueError("learning-comparison cells must be declared cells")
        method_ids = tuple(item.method_id for item in self.methods)
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("duplicate learning method")
        if parse_rational(self.base_team_return) <= 0:
            raise ValueError("base_team_return must be positive")
        if parse_rational(self.loss_scale) <= 0:
            raise ValueError("loss_scale must be positive")
        return self


@dataclass(frozen=True)
class LearningGame:
    cell_id: str
    split: SplitName
    profile_id: str
    source_population_id: str
    partner_identity_prefix: str
    game: FiniteConventionGame
    commitment_states: frozenset[str]
    dynamics_hash: str


@dataclass(frozen=True)
class LearningCellPools:
    cell_id: str
    source_population: GeneratedPopulation
    train: tuple[LearningGame, ...]
    validation: tuple[LearningGame, ...]
    test: tuple[LearningGame, ...]


@dataclass(frozen=True)
class GeneratedLearningPools:
    suite: LearningAuditSuite
    cells: tuple[LearningCellPools, ...]
    source_suite_hash: str

    def by_cell(self) -> dict[str, LearningCellPools]:
        return {item.cell_id: item for item in self.cells}


@dataclass(frozen=True)
class TrainingRunManifest:
    schema_version: int
    run_id: str
    suite_id: str
    cell_id: str
    method_id: LearningMethod
    seed: int
    device: str
    requested_transitions: int
    completed_transitions: int
    configuration_hash: str
    training_pool_hashes: tuple[str, ...]
    validation_pool_hashes: tuple[str, ...]
    checkpoint_path: str
    metrics_path: str
    deterministic: bool
    checkpoint_hash: str
    source_tree_hash: str
    python_version: str
    dependency_versions: dict[str, str]
    rng_configuration: dict[str, Any]
    pretraining_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


@dataclass(frozen=True)
class LearnedPolicyEvaluation:
    population_id: str
    method_id: str
    mode: EvaluationMode
    evaluator: str
    team_return: float
    expected_intervention_cost: float
    actual_confusion_loss: float
    residual_bayes_risk: float
    decision_utilization_gap: float
    total_regret: float
    policy_dri: float | None
    probe_probability: float
    expected_commitment_time: float
    identity_mutual_information_bits: float
    decision_signature_mutual_information_bits: float
    active_frontier_distance: float | None
    oracle_normalized_return: float | None
    response_signature_accuracy: float | None
    belief_brier_score: float | None
    expected_calibration_error: float | None
    partner_response_prediction_loss: float | None
    commitment_time_distribution: dict[str, float]
    applicability_flags: dict[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


@dataclass(frozen=True)
class ReconnaissanceEvaluation:
    population_id: str
    method_id: str
    episodes: int
    scored_episode_return: float
    reconnaissance_episode_return: float
    combined_return_sum: float
    combined_return_mean: float
    reconnaissance_cost: float
    reconnaissance_loss: float
    extra_partner_interactions: float

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


@dataclass(frozen=True)
class LearningAuditManifest:
    schema_version: int
    suite_id: str
    status: Literal["complete", "incomplete", "invalid"]
    scientific_verdict: Literal[
        "continue_to_repair", "continue_without_repair", "redesign", "stop", "pending"
    ]
    implementation_passed: bool
    configuration_hash: str
    source_tree_hash: str
    generated_files: tuple[str, ...]
    missing_runs: tuple[str, ...]
    python_version: str
    dependency_versions: dict[str, str]
    invoked_command: tuple[str, ...]
    rng_configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], _serialize(self))


def load_learning_suite_file(path: str | Path) -> LearningAuditSuite:
    with Path(path).open(encoding="utf-8") as handle:
        return LearningAuditSuite.model_validate(json.load(handle))


def serialize_learning(value: Any) -> Any:
    return _serialize(value)


def _serialize(value: Any) -> Any:
    if isinstance(value, Fraction):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except ImportError:  # pragma: no cover - NumPy is a required runtime dependency
        pass
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, tuple | list | frozenset):
        return [_serialize(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return {name: _serialize(getattr(value, name)) for name in value.__dataclass_fields__}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot serialize {type(value)!r}")
