"""Versioned models for the inference-only official-checkpoint Stage 6 audit."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

OfficialLayoutId = Literal["random3_m", "small_corridor"]
OfficialMethodId = Literal["fcp", "mep", "trajedi", "hsp", "cole", "e3t"]
OfficialSplit = Literal["calibration", "validation", "confirmatory"]
OfficialShardKind = Literal["parity", "response", "trace", "method"]
OfficialShardStatus = Literal["pending", "running", "complete", "failed"]
OfficialVerdict = Literal[
    "pending",
    "continue_top_paper_package",
    "complete_evaluation_only",
    "complete_measurement_only",
    "redesign",
    "stop",
]


class FrozenOfficialModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class OfficialUpstreamSpec(FrozenOfficialModel):
    repository_url: str
    repository_commit: str
    policy_pool_url: str
    policy_pool_revision: str
    license: Literal["MIT"] = "MIT"

    @model_validator(mode="after")
    def validate_pins(self) -> OfficialUpstreamSpec:
        for value in (self.repository_commit, self.policy_pool_revision):
            if re.fullmatch(r"[0-9a-f]{40}", value) is None:
                raise ValueError("official repository and model revisions must be full hashes")
        return self


class OfficialLayoutSpec(FrozenOfficialModel):
    layout_id: OfficialLayoutId
    role: Literal["primary", "robustness"]
    benchmark_yaml_path: str
    expected_partner_count: int = Field(ge=1)
    expected_scheme_count: int = Field(ge=1)
    response_episodes_per_pair: int = Field(ge=1)
    trace_episodes: dict[OfficialSplit, int]
    method_episodes: int = Field(ge=1)
    diagnostic_options: tuple[str, ...]
    max_episode_steps: Literal[400] = 400

    @model_validator(mode="after")
    def validate_layout(self) -> OfficialLayoutSpec:
        if self.expected_partner_count != 2 * self.expected_scheme_count:
            raise ValueError("official layouts require one mid and one final partner per scheme")
        if set(self.trace_episodes) != {"calibration", "validation", "confirmatory"}:
            raise ValueError("trace counts must declare three disjoint evaluation splits")
        if not self.diagnostic_options or self.diagnostic_options[0] != "ordinary_progress":
            raise ValueError("every layout must include ordinary_progress as its comparator")
        path = Path(self.benchmark_yaml_path)
        if path.is_absolute() or ".." in path.parts or path.suffix not in {".yml", ".yaml"}:
            raise ValueError("benchmark YAML must be a safe relative path")
        return self


class OfficialMethodSpec(FrozenOfficialModel):
    method_id: OfficialMethodId
    asset_path_template: str
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5)
    architecture_by_layout: dict[OfficialLayoutId, Literal["mlp", "rnn"]]

    @model_validator(mode="after")
    def validate_method(self) -> OfficialMethodSpec:
        if tuple(sorted(set(self.seeds))) != self.seeds:
            raise ValueError("method seeds must be sorted and unique")
        if "{seed}" not in self.asset_path_template:
            raise ValueError("method asset templates must contain {seed}")
        return self


class OfficialEvidenceSpec(FrozenOfficialModel):
    passive_reference_method: Literal["fcp"] = "fcp"
    passive_reference_seed: Literal[1] = 1
    passive_reference_greedy: Literal[True] = True
    prefix_steps: tuple[int, ...] = (0, 8, 16, 32)
    commitment_event: Literal["first_successful_pot_ingredient_placement"] = (
        "first_successful_pot_ingredient_placement"
    )
    post_commitment_endpoint: Literal["first_delivery_feedback"] = "first_delivery_feedback"
    maximum_option_steps: int = Field(default=16, ge=1, le=16)


class OfficialEstimatorSpec(FrozenOfficialModel):
    event_laplace_alpha: float = Field(default=1.0, gt=0)
    gru_hidden_size: Literal[64] = 64
    gru_seeds: tuple[int, ...] = (6173, 6174, 6175, 6176, 6177)
    temperature_calibration: Literal[True] = True
    direct_pairwise_refit_count: Literal[10] = 10
    synthetic_tolerance: float = Field(default=0.03, ge=0)
    treatment_agreement_tolerance: float = Field(default=0.05, ge=0)
    label_shuffle_repeats: int = Field(default=100, ge=1)


class OfficialStatisticsSpec(FrozenOfficialModel):
    ridge_strengths: tuple[float, ...] = (0.0, 0.01, 0.1, 1.0, 10.0)
    bootstrap_resamples: Literal[10000] = 10000
    correction: Literal["holm"] = "holm"
    held_out_unit: Literal["hsp_scheme"] = "hsp_scheme"
    adequacy_margins: tuple[float, ...] = (0.01, 0.02, 0.05)
    primary_adequacy_margin: float = Field(default=0.02, ge=0, le=1)

    @model_validator(mode="after")
    def validate_primary_margin(self) -> OfficialStatisticsSpec:
        if abs(self.primary_adequacy_margin - 0.02) > 1e-12:
            raise ValueError("the preregistered primary adequacy margin is 0.02")
        if self.primary_adequacy_margin not in self.adequacy_margins:
            raise ValueError("primary adequacy margin must be included in sensitivity margins")
        return self


class OfficialRuntimeSpec(FrozenOfficialModel):
    runtime_project: str = "phase-6-established-validation/runtime-zsceval"
    upstream_directory: str = ".external/zsc-eval"
    asset_directory: str = ".external/zsc-eval-policy-pool"
    python_version: Literal["3.9"] = "3.9"
    device: Literal["cpu"] = "cpu"
    default_workers: int = Field(default=2, ge=1, le=4)
    maximum_workers: Literal[4] = 4


class OfficialCheckpointAuditSuiteV2(FrozenOfficialModel):
    schema_version: Literal[2] = 2
    audit_mode: Literal["official_checkpoint_evaluation"] = "official_checkpoint_evaluation"
    suite_id: str
    state_directory: str
    policy_training_allowed: Literal[False] = False
    partner_selection_criterion: Literal["official_benchmark_yaml_only"] = (
        "official_benchmark_yaml_only"
    )
    upstream: OfficialUpstreamSpec
    runtime: OfficialRuntimeSpec
    layouts: tuple[OfficialLayoutSpec, ...]
    methods: tuple[OfficialMethodSpec, ...]
    evidence: OfficialEvidenceSpec
    estimator: OfficialEstimatorSpec
    statistics: OfficialStatisticsSpec
    split_key_salts: dict[OfficialSplit, str]
    required_policy_configs: tuple[str, ...]

    @model_validator(mode="before")
    @classmethod
    def reject_training_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            forbidden = {
                "training",
                "training_budget",
                "transition_budget",
                "transitions",
                "partner_generation",
                "train_partners",
            }

            def visit(value: Any, path: tuple[str, ...] = ()) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if key in forbidden:
                            raise ValueError(
                                "policy training is forbidden in an official-checkpoint suite: "
                                + ".".join((*path, key))
                            )
                        visit(child, (*path, key))
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        visit(child, (*path, str(index)))

            visit(data)
        return data

    @model_validator(mode="after")
    def validate_contract(self) -> OfficialCheckpointAuditSuiteV2:
        if len(self.layouts) < 2 or {item.role for item in self.layouts} != {
            "primary",
            "robustness",
        }:
            raise ValueError("official audit requires primary and robustness layouts")
        if len(self.methods) < 4:
            raise ValueError("official audit requires at least four published ZSC methods")
        if len({item.layout_id for item in self.layouts}) != len(self.layouts):
            raise ValueError("official layout identifiers must be unique")
        if len({item.method_id for item in self.methods}) != len(self.methods):
            raise ValueError("official method identifiers must be unique")
        layout_ids = {item.layout_id for item in self.layouts}
        if any(set(item.architecture_by_layout) != layout_ids for item in self.methods):
            raise ValueError("every official method must declare its architecture per layout")
        if set(self.split_key_salts) != {"calibration", "validation", "confirmatory"}:
            raise ValueError("split-key salts must cover calibration, validation, confirmatory")
        if len(set(self.split_key_salts.values())) != 3:
            raise ValueError("split-key salts must be disjoint")
        if not self.required_policy_configs:
            raise ValueError("official response counterparts require policy configurations")
        return self


class OfficialAssetLockEntry(FrozenOfficialModel):
    asset_id: str
    relative_path: str
    source: Literal["repository", "policy_pool"]
    source_revision: str
    expected_size: int | None = Field(default=None, ge=0)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    role: Literal["benchmark", "config", "partner", "response", "method", "source"]


class OfficialAssetLock(FrozenOfficialModel):
    schema_version: Literal[1] = 1
    suite_id: str
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    repository_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    policy_pool_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    workspace: str
    entries: tuple[OfficialAssetLockEntry, ...]
    lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OfficialPartnerAsset(FrozenOfficialModel):
    partner_id: str
    layout_id: OfficialLayoutId
    scheme_id: str
    training_stage: Literal["mid", "final"]
    partner_checkpoint_path: str
    response_checkpoint_path: str
    partner_asset_id: str
    response_asset_id: str


class OfficialMethodAsset(FrozenOfficialModel):
    method_id: OfficialMethodId
    layout_id: OfficialLayoutId
    seed: int
    checkpoint_path: str
    asset_id: str
    policy_architecture: Literal["mlp", "rnn"]
    recurrent: bool


class OfficialAssetRecord(FrozenOfficialModel):
    asset_id: str
    local_path: str
    repository_path: str | None = None
    revision: str | None = None
    size: int = Field(ge=0)
    file_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_tensor_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    layout_id: OfficialLayoutId | None = None
    algorithm: str | None = None
    seed: int | None = None
    policy_architecture: Literal["mlp", "rnn", "configuration", "metadata"] | None = None
    provenance: (
        Literal[
            "official_benchmark_yaml",
            "official_response_counterpart",
            "official_method_spec",
            "required_policy_config",
            "official_metadata",
        ]
        | None
    ) = None


class OfficialAssetInventory(FrozenOfficialModel):
    schema_version: Literal[1] = 1
    suite_id: str
    lock_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    partners: tuple[OfficialPartnerAsset, ...]
    methods: tuple[OfficialMethodAsset, ...]
    assets: tuple[OfficialAssetRecord, ...]
    duplicate_tensor_groups: tuple[tuple[str, ...], ...] = ()
    complete: bool
    missing_asset_ids: tuple[str, ...] = ()
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OfficialRolloutShard(FrozenOfficialModel):
    shard_id: str
    kind: OfficialShardKind
    layout_id: OfficialLayoutId
    request_path: str
    result_path: str
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    partner_id: str | None = None
    response_id: str | None = None
    method_id: OfficialMethodId | None = None
    method_seed: int | None = None
    deployment: Literal["greedy", "stochastic"] | None = None
    evidence_policy: str | None = None
    split: OfficialSplit | None = None
    episode_keys: tuple[int, ...] = ()
    deterministic: bool = False


class OfficialRolloutPlan(FrozenOfficialModel):
    schema_version: Literal[1] = 1
    suite_id: str
    suite_path: str
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_path: str
    inventory_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace: str
    shards: tuple[OfficialRolloutShard, ...]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class OfficialRolloutLedgerEntry(FrozenOfficialModel):
    shard_id: str
    status: OfficialShardStatus
    attempts: int = Field(default=0, ge=0)
    result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    error: str | None = None


class OfficialRolloutLedger(FrozenOfficialModel):
    schema_version: Literal[1] = 1
    suite_id: str
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[OfficialRolloutLedgerEntry, ...]
    complete: bool
    failed_shards: tuple[str, ...] = ()


class OfficialResponseValueMatrix(FrozenOfficialModel):
    schema_version: Literal[1] = 1
    suite_id: str
    layout_id: OfficialLayoutId
    partner_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    raw_values: tuple[tuple[float, ...], ...]
    raw_value_intervals_95: tuple[tuple[tuple[float, float], ...], ...]
    normalized_losses: tuple[tuple[float, ...], ...]
    adequate_response_sets: dict[str, tuple[str, ...]]
    conflicting_pairs_by_margin: dict[str, tuple[tuple[str, str], ...]]
    conflict_coefficients: dict[str, float]
    best_response_event_features: dict[str, tuple[float, ...]]
    rahman_brdiv_return: float
    zsceval_br_div_raw: float
    zsceval_br_div_code: float


class OfficialTraceIndexEntry(FrozenOfficialModel):
    trace_id: str
    layout_id: OfficialLayoutId
    partner_id: str
    evidence_policy: str
    split: OfficialSplit
    path: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    episodes: int = Field(ge=1)


class OfficialTraceIndex(FrozenOfficialModel):
    schema_version: Literal[1] = 1
    suite_id: str
    entries: tuple[OfficialTraceIndexEntry, ...]


class PairwiseIdentifiabilityRow(FrozenOfficialModel):
    layout_id: OfficialLayoutId
    left_partner_id: str
    right_partner_id: str
    left_scheme_id: str
    right_scheme_id: str
    evidence_policy: str
    estimator: Literal["event", "gru"]
    prefix: str
    prior_risk: float = Field(ge=0)
    residual_risk: float = Field(ge=0)
    dri: float | None
    identity_mi_nats: float = Field(ge=0)
    decision_mi_nats: float = Field(ge=0)
    prefix_tv: float = Field(ge=0, le=1)
    commitment_rate: float = Field(ge=0, le=1)


class OfficialCheckpointAuditManifest(FrozenOfficialModel):
    schema_version: Literal[1] = 1
    suite_id: str
    status: Literal["complete", "incomplete", "invalid"]
    verdict: OfficialVerdict
    asset_integrity_passed: bool
    runtime_parity_passed: bool
    scientific_gates: dict[str, bool | None]
    generated_files: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    total_episodes: int = Field(ge=0)
    total_environment_steps: int = Field(ge=0)
    peak_workers: int = Field(ge=0, le=4)
    peak_memory_bytes: int | None = Field(default=None, ge=0)
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    inventory_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    invoked_command: tuple[str, ...]


def load_official_checkpoint_suite(
    path: str | Path,
) -> OfficialCheckpointAuditSuiteV2:
    with Path(path).open(encoding="utf-8") as handle:
        return OfficialCheckpointAuditSuiteV2.model_validate(json.load(handle))
