"""Versioned schemas for Stage 6 established-environment validation.

The main package deliberately models manifests and traces only.  The pinned
OvercookedV2 runtime executes in a separate Python 3.10 environment and
communicates with this module through JSON files.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RuntimeKind = Literal["overcookedv2_py310", "zsceval_py39", "tomzsc_py310"]
SplitName = Literal["train", "validation", "evaluation"]
PartnerPoolStage = Literal["screen", "finalist"]
PartnerCandidateStatus = Literal[
    "inactive",
    "pending_screen",
    "screen_running",
    "screen_rejected",
    "pending_finalist",
    "finalist_running",
    "finalist_rejected",
    "eligible",
    "failed",
]
EstablishedMethod = Literal[
    "rnn_ippo",
    "fcp",
    "other_play",
    "tbs_style",
    "pace_aux",
    "pace_style",
    "csp_style_reconnaissance",
]
EstablishedPolicyKind = Literal[
    "ppo",
    "pace",
    "tbs_selector",
    "csp_reconnaissance",
]
ComputeAllocation = Literal["per-specialist", "split-total"]
EvidencePolicy = Literal[
    "ordinary_progress",
    "stage_candidate_ingredient",
    "temporary_role_takeover",
    "corridor_yield",
    "recipe_button_control",
]
Stage6Verdict = Literal["complete_evaluation_only", "reopen_phase5", "redesign", "stop", "pending"]


class FrozenEstablishedModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class UpstreamRepositorySpec(FrozenEstablishedModel):
    repository_id: Literal["overcookedv2", "zsceval", "tomzsc"]
    url: str
    commit: str
    runtime: RuntimeKind
    local_directory: str
    required: bool = True

    @model_validator(mode="after")
    def validate_pin(self) -> UpstreamRepositorySpec:
        if re.fullmatch(r"[0-9a-f]{40}", self.commit) is None:
            raise ValueError("upstream commits must be full lowercase 40-character SHA-1 hashes")
        path = Path(self.local_directory)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("upstream local_directory must be a safe relative path")
        return self


class RuntimeBoundarySpec(FrozenEstablishedModel):
    runtime_id: RuntimeKind
    python_version: Literal["3.10", "3.9"]
    project_directory: str
    request_schema_version: Literal[1] = 1
    trace_schema_version: Literal[1] = 1


class LayoutSpec(FrozenEstablishedModel):
    layout_id: Literal[
        "demo_cook_simple",
        "test_time_simple",
        "grounded_coord_simple",
        "demo_cook_wide",
    ]
    role: Literal[
        "primary",
        "timing",
        "environment_information_control",
        "geometry_replication",
    ]
    max_steps: int = Field(default=400, ge=1)
    agent_view_size: int = Field(default=2, ge=1)
    random_agent_positions: bool = True
    negative_incorrect_delivery_reward: bool = True
    sample_recipe_on_delivery: bool = True
    launch_condition: Literal["always", "after_primary_feasibility"] = "always"


class CommitmentRuleSpec(FrozenEstablishedModel):
    event_name: Literal["successful_pot_ingredient_placement"] = (
        "successful_pot_ingredient_placement"
    )
    work_unit_reset_events: tuple[str, ...] = ("episode_reset", "successful_delivery")
    no_commitment_treatment: Literal["censored_prior_risk"] = "censored_prior_risk"
    primary_work_unit: int = Field(default=0, ge=0)


class DRIEstimatorSpec(FrozenEstablishedModel):
    prefix_steps: tuple[int, ...] = (0, 8, 16, 32)
    estimators: tuple[Literal["gru", "event"], ...] = ("gru", "event")
    temperature_calibration: bool = True
    folds: int = Field(default=5, ge=2)
    random_seed: int = 6173
    synthetic_absolute_tolerance: float = Field(default=0.03, ge=0)
    treatment_agreement_tolerance: float = Field(default=0.05, ge=0)
    label_shuffle_repeats: int = Field(default=100, ge=1)
    gru_hidden_size: int = Field(default=64, ge=8)
    gru_learning_rate: float = Field(default=1e-3, gt=0)
    gru_max_epochs: int = Field(default=100, ge=1)
    gru_patience: int = Field(default=12, ge=1)

    @model_validator(mode="after")
    def validate_prefixes(self) -> DRIEstimatorSpec:
        if not self.prefix_steps or self.prefix_steps[0] != 0:
            raise ValueError("DRI prefix_steps must begin at zero")
        if tuple(sorted(set(self.prefix_steps))) != self.prefix_steps:
            raise ValueError("DRI prefix_steps must be unique and sorted")
        if set(self.estimators) != {"gru", "event"}:
            raise ValueError("Stage 6 requires both GRU and high-level-event estimators")
        return self


class DiagnosticOptionSpec(FrozenEstablishedModel):
    option_id: EvidencePolicy
    max_low_level_steps: int = Field(default=16, ge=1, le=16)
    partner_diagnostic_candidate: bool = True
    environment_information_control: bool = False

    @model_validator(mode="after")
    def validate_role(self) -> DiagnosticOptionSpec:
        if self.option_id == "recipe_button_control":
            if not self.environment_information_control or self.partner_diagnostic_candidate:
                raise ValueError("the recipe button must be an environment-information control")
        elif self.environment_information_control:
            raise ValueError("only the recipe button may be the environment-information control")
        return self


class PartnerGenerationSpec(FrozenEstablishedModel):
    behavior_events: tuple[str, ...]
    preference_values: tuple[float, ...] = (-1.0, 1.0)
    maximum_nonzero_behavior_preferences: int = Field(default=3, ge=1, le=3)
    screen_transitions: int = Field(default=5_000_000, ge=1)
    finalist_transitions: int = Field(default=30_000_000, ge=1)
    seeds_per_reward_vector: Literal[2] = 2
    validation_rollouts: int = Field(default=100, ge=1)
    minimum_correct_delivery_rate: float = Field(default=0.8, ge=0, le=1)
    training_partner_quota: int = Field(default=24, ge=1)
    training_candidate_cap: int = Field(default=48, ge=1)
    validation_partner_quota: int = Field(default=8, ge=1)
    validation_candidate_cap: int = Field(default=16, ge=1)
    evaluation_candidate_quota: int = Field(default=32, ge=1)
    evaluation_candidate_cap: int = Field(default=64, ge=32)
    expansion_block_size: int = Field(default=8, ge=1)
    split_hash_salt: str = "zsc-identifiability-stage6-v1"
    split_proportions: tuple[int, int, int] = (3, 1, 4)

    @model_validator(mode="after")
    def validate_generation(self) -> PartnerGenerationSpec:
        if not self.behavior_events or len(self.behavior_events) != len(set(self.behavior_events)):
            raise ValueError("behavior event identifiers must be nonempty and unique")
        if not self.preference_values or any(value == 0 for value in self.preference_values):
            raise ValueError("preference_values must be nonempty and non-zero")
        if self.finalist_transitions < self.screen_transitions:
            raise ValueError("finalist budget cannot be below the screen budget")
        quotas_and_caps = (
            (self.training_partner_quota, self.training_candidate_cap),
            (self.validation_partner_quota, self.validation_candidate_cap),
            (self.evaluation_candidate_quota, self.evaluation_candidate_cap),
        )
        if any(cap < quota for quota, cap in quotas_and_caps):
            raise ValueError("partner candidate caps cannot be below their quotas")
        grouping_values = (
            *(value for pair in quotas_and_caps for value in pair),
            self.expansion_block_size,
        )
        if any(value % self.seeds_per_reward_vector for value in grouping_values):
            raise ValueError(
                "partner quotas, caps, and expansion size must preserve complete seed groups"
            )
        if sum(self.split_proportions) <= 0 or any(value <= 0 for value in self.split_proportions):
            raise ValueError("partner split proportions must be positive")
        return self


class ResponseLibrarySpec(FrozenEstablishedModel):
    adequacy_margin: float = Field(default=0.02, ge=0, le=1)
    maximum_response_clusters: int = Field(default=6, ge=2, le=6)
    include_cotrained_counterparts: bool = True
    include_cluster_best_responses: bool = True
    include_fcp: bool = True
    include_other_play: bool = True


class MatchingSpec(FrozenEstablishedModel):
    subset_size: int = Field(default=8, ge=2)
    competence_margin: float = Field(default=0.02, ge=0)
    fixed_response_margin: float = Field(default=0.02, ge=0)
    br_prox_margin: float = Field(default=0.02, ge=0)
    br_div_logdet_margin: float = Field(default=0.05, ge=0)
    predictability_margin_nats: float = Field(default=0.02, ge=0)
    trajectory_divergence_margin: float = Field(default=0.05, ge=0)
    discovery_commitment_rate_margin: float = Field(default=0.05, ge=0)
    active_contrast_passive_dri_margin: float = Field(default=0.02, ge=0)
    minimum_dri_separation: float = Field(default=0.15, gt=0)
    require_equal_cluster_counts: bool = True
    require_commitment_rate_nonsignificant: bool = True
    contrasts: tuple[Literal["passive_dri", "active_dri"], ...] = (
        "passive_dri",
        "active_dri",
    )


class EstablishedMethodSpec(FrozenEstablishedModel):
    method_id: EstablishedMethod
    development_transitions: int = Field(default=5_000_000, ge=1)
    confirmatory_transitions: int = Field(ge=1)
    reconnaissance_episodes: int = Field(default=0, ge=0, le=1)
    included_in_central_ranking: bool = True

    @model_validator(mode="after")
    def validate_protocol(self) -> EstablishedMethodSpec:
        is_csp = self.method_id == "csp_style_reconnaissance"
        if is_csp != (self.reconnaissance_episodes == 1):
            raise ValueError("only CSP-style must receive one reconnaissance episode")
        if is_csp and self.included_in_central_ranking:
            raise ValueError("CSP-style reconnaissance is excluded from central rankings")
        return self


class EstablishedMethodAssetsManifest(FrozenEstablishedModel):
    """Content-addressed inputs and intermediate products for a method port."""

    schema_version: Literal[1] = 1
    method_id: EstablishedMethod
    train_pool_path: str
    train_pool_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    validation_pool_path: str
    validation_pool_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cross_play_values_path: str | None = None
    cross_play_values_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    cluster_assignments: dict[str, int] = Field(default_factory=dict)
    concept_schema: tuple[str, ...] = ()
    component_paths: dict[str, str] = Field(default_factory=dict)
    component_hashes: dict[str, str] = Field(default_factory=dict)
    compute_allocation: ComputeAllocation = "per-specialist"

    @model_validator(mode="after")
    def validate_method_assets(self) -> EstablishedMethodAssetsManifest:
        has_values = self.cross_play_values_path is not None
        if has_values != (self.cross_play_values_hash is not None):
            raise ValueError("cross-play values path and hash must be supplied together")
        if self.method_id == "tbs_style" and not has_values:
            raise ValueError("TBS-style assets require training cross-play values")
        if any(value < 0 or value >= 6 for value in self.cluster_assignments.values()):
            raise ValueError("method cluster identifiers must be between zero and five")
        if set(self.component_paths) != set(self.component_hashes):
            raise ValueError("method component paths and hashes must use identical identifiers")
        for key, value in self.component_hashes.items():
            if re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"component {key!r} has an invalid content hash")
        return self


class EstablishedPolicyComponent(FrozenEstablishedModel):
    component_id: str
    role: Literal[
        "task_policy",
        "specialist",
        "probe_policy",
        "trajectory_encoder",
        "response_decoder",
        "global_tom",
        "cluster_tom",
        "centroids",
    ]
    path: str
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    cluster_id: int | None = Field(default=None, ge=0, le=5)


class EstablishedPolicyArtifact(FrozenEstablishedModel):
    """Compact deployment artifact; full optimizer state is stored separately."""

    schema_version: Literal[1] = 1
    policy_kind: EstablishedPolicyKind
    method_id: EstablishedMethod
    layout_id: str
    seed: int
    backbone_config: dict[str, Any]
    components: tuple[EstablishedPolicyComponent, ...]
    partner_ids: tuple[str, ...]
    cluster_assignments: dict[str, int] = Field(default_factory=dict)
    concept_schema: tuple[str, ...] = ()
    centroids: tuple[tuple[float, ...], ...] = ()
    reconnaissance_episodes: int = Field(default=0, ge=0, le=1)
    source_configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    aggregate_training_transitions: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_policy_protocol(self) -> EstablishedPolicyArtifact:
        if not self.components:
            raise ValueError("policy artifacts require at least one component")
        if len(self.partner_ids) != len(set(self.partner_ids)):
            raise ValueError("policy artifact partner identifiers must be unique")
        is_csp = self.policy_kind == "csp_reconnaissance"
        if is_csp != (self.reconnaissance_episodes == 1):
            raise ValueError("only CSP artifacts use one reconnaissance episode")
        roles = [component.role for component in self.components]
        if self.policy_kind == "pace" and roles.count("task_policy") != 1:
            raise ValueError("PACE artifacts require exactly one task policy")
        if self.policy_kind == "tbs_selector":
            specialists = roles.count("specialist")
            if specialists < 1 or roles.count("global_tom") != 1:
                raise ValueError("TBS artifacts require specialists and one global ToM")
            if roles.count("cluster_tom") != specialists:
                raise ValueError("TBS artifacts require one cluster ToM per specialist")
            if not self.concept_schema:
                raise ValueError("TBS artifacts require a visible concept schema")
        if is_csp:
            if (
                roles.count("probe_policy") != 1
                or roles.count("trajectory_encoder") != 1
                or roles.count("response_decoder") != 1
            ):
                raise ValueError("CSP artifacts require probe, encoder, and decoder components")
            specialist_count = roles.count("specialist")
            if not self.centroids or specialist_count != len(self.centroids):
                raise ValueError("CSP artifacts require one specialist per centroid")
        return self


class TrainingCheckpointMetadata(FrozenEstablishedModel):
    """Metadata next to an ignored Orbax training-state checkpoint."""

    schema_version: Literal[2] = 2
    suite_id: str
    method_id: EstablishedMethod | Literal["partner_ippo"]
    layout_id: str
    seed: int
    component_id: str
    completed_transitions: int = Field(ge=0)
    target_transitions: int = Field(ge=1)
    configuration_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_hashes: tuple[str, ...]
    upstream_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    upstream_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: str
    state_tree_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_checkpoint_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    exact_continuation: bool
    validation_metric: float | None = None
    best_validation_metric: float | None = None


class EstablishedTrainingSpec(FrozenEstablishedModel):
    smoke_transitions: int = Field(default=100_000, ge=1)
    development_seeds: tuple[int, ...] = (5101, 5102, 5103)
    confirmatory_seeds: tuple[int, ...] = tuple(range(6001, 6011))
    learning_rates: tuple[float, ...] = (1.25e-4, 2.5e-4, 5e-4)
    entropy_coefficients: tuple[float, ...] = (0.01, 0.02)
    methods: tuple[EstablishedMethodSpec, ...]

    @model_validator(mode="after")
    def validate_training(self) -> EstablishedTrainingSpec:
        method_ids = tuple(method.method_id for method in self.methods)
        if len(method_ids) != len(set(method_ids)):
            raise ValueError("duplicate established method")
        if len(set(self.development_seeds)) != len(self.development_seeds):
            raise ValueError("duplicate development seed")
        if len(set(self.confirmatory_seeds)) != len(self.confirmatory_seeds):
            raise ValueError("duplicate confirmatory seed")
        if set(self.development_seeds) & set(self.confirmatory_seeds):
            raise ValueError("development and confirmatory seeds must be disjoint")
        return self


class EstablishedStatisticsSpec(FrozenEstablishedModel):
    episodes_per_method_partner_seed_layout: int = Field(default=500, ge=1)
    bootstrap_resamples: int = Field(default=10_000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0, lt=1)
    correction: Literal["holm"] = "holm"
    held_out_unit: Literal["reward_vector"] = "reward_vector"


class SecondaryAuditSpec(FrozenEstablishedModel):
    enabled: bool = True
    minimum_partners: int = Field(default=8, ge=1)
    minimum_algorithms: int = Field(default=4, ge=1)
    unavailable_is_blocking: bool = False


class EstablishedValidationSuite(FrozenEstablishedModel):
    schema_version: Literal[1]
    suite_id: str
    state_directory: str = "phase-6-established-validation/runs"
    upstreams: tuple[UpstreamRepositorySpec, ...]
    runtimes: tuple[RuntimeBoundarySpec, ...]
    layouts: tuple[LayoutSpec, ...]
    commitment: CommitmentRuleSpec = Field(default_factory=CommitmentRuleSpec)
    dri_estimator: DRIEstimatorSpec = Field(default_factory=DRIEstimatorSpec)
    diagnostics: tuple[DiagnosticOptionSpec, ...]
    partner_generation: PartnerGenerationSpec
    response_library: ResponseLibrarySpec = Field(default_factory=ResponseLibrarySpec)
    matching: MatchingSpec = Field(default_factory=MatchingSpec)
    training: EstablishedTrainingSpec
    statistics: EstablishedStatisticsSpec = Field(default_factory=EstablishedStatisticsSpec)
    secondary_zsceval: SecondaryAuditSpec = Field(default_factory=SecondaryAuditSpec)

    @model_validator(mode="after")
    def validate_suite(self) -> EstablishedValidationSuite:
        state_path = Path(self.state_directory)
        if state_path.is_absolute() or ".." in state_path.parts:
            raise ValueError("state_directory must be a safe project-relative path")
        upstream_ids = tuple(item.repository_id for item in self.upstreams)
        if len(upstream_ids) != len(set(upstream_ids)) or set(upstream_ids) != {
            "overcookedv2",
            "zsceval",
            "tomzsc",
        }:
            raise ValueError("suite must pin exactly OvercookedV2, ZSC-Eval, and ToMZSC")
        fixed_upstreams = {
            "overcookedv2": (
                "https://github.com/overcookedv2/experiments",
                "5ce1707cf31c1c115e6f6ba96db7bc9cc80a850e",
                "overcookedv2_py310",
            ),
            "zsceval": (
                "https://github.com/SJTU-MARL/ZSC-Eval",
                "f940869afc42b688332a385892d8dbb57a190f95",
                "zsceval_py39",
            ),
            "tomzsc": (
                "https://github.com/andrewni2002/ToMZSC",
                "a4b41d53fc77e452cdfca8edc95fd153d51d13cd",
                "tomzsc_py310",
            ),
        }
        for upstream in self.upstreams:
            expected_url, expected_commit, expected_runtime = fixed_upstreams[
                upstream.repository_id
            ]
            if (
                upstream.url.rstrip("/").removesuffix(".git") != expected_url
                or upstream.commit != expected_commit
                or upstream.runtime != expected_runtime
            ):
                raise ValueError(f"canonical upstream pin changed: {upstream.repository_id}")
        runtime_ids = tuple(item.runtime_id for item in self.runtimes)
        if len(runtime_ids) != len(set(runtime_ids)):
            raise ValueError("duplicate runtime boundary")
        expected_runtime_versions = {
            "overcookedv2_py310": "3.10",
            "zsceval_py39": "3.9",
            "tomzsc_py310": "3.10",
        }
        if {
            item.runtime_id: item.python_version for item in self.runtimes
        } != expected_runtime_versions:
            raise ValueError("canonical isolated runtime versions changed")
        required_runtimes = {item.runtime for item in self.upstreams}
        if not required_runtimes.issubset(runtime_ids):
            raise ValueError("every upstream runtime must have a declared boundary")
        layout_ids = tuple(item.layout_id for item in self.layouts)
        if len(layout_ids) != len(set(layout_ids)):
            raise ValueError("duplicate established layout")
        if set(layout_ids) != {
            "demo_cook_simple",
            "test_time_simple",
            "grounded_coord_simple",
            "demo_cook_wide",
        }:
            raise ValueError("canonical Stage 6 layout set is incomplete")
        expected_layout_roles = {
            "demo_cook_simple": ("primary", "always"),
            "test_time_simple": ("timing", "always"),
            "grounded_coord_simple": ("environment_information_control", "always"),
            "demo_cook_wide": ("geometry_replication", "after_primary_feasibility"),
        }
        for layout in self.layouts:
            if (
                layout.max_steps != 400
                or layout.agent_view_size != 2
                or not layout.random_agent_positions
                or not layout.negative_incorrect_delivery_reward
                or not layout.sample_recipe_on_delivery
            ):
                raise ValueError("layouts must preserve the fixed official Stage 6 settings")
            if (layout.role, layout.launch_condition) != expected_layout_roles[layout.layout_id]:
                raise ValueError(f"canonical layout role changed: {layout.layout_id}")
        option_ids = tuple(item.option_id for item in self.diagnostics)
        if len(option_ids) != len(set(option_ids)) or set(option_ids) != {
            "ordinary_progress",
            "stage_candidate_ingredient",
            "temporary_role_takeover",
            "corridor_yield",
            "recipe_button_control",
        }:
            raise ValueError("canonical diagnostic-action audit is incomplete")
        required_methods = {
            "rnn_ippo",
            "fcp",
            "other_play",
            "tbs_style",
            "pace_aux",
            "pace_style",
            "csp_style_reconnaissance",
        }
        if {item.method_id for item in self.training.methods} != required_methods:
            raise ValueError("canonical established-method set is incomplete")
        expected_budgets = {
            "rnn_ippo": 30_000_000,
            "fcp": 30_000_000,
            "other_play": 50_000_000,
            "tbs_style": 40_000_000,
            "pace_aux": 30_000_000,
            "pace_style": 30_000_000,
            "csp_style_reconnaissance": 30_000_000,
        }
        if any(
            item.development_transitions != 5_000_000
            or item.confirmatory_transitions != expected_budgets[item.method_id]
            for item in self.training.methods
        ):
            raise ValueError("canonical established-method budgets changed")
        if (
            self.training.smoke_transitions != 100_000
            or self.training.development_seeds != (5101, 5102, 5103)
            or self.training.confirmatory_seeds != tuple(range(6001, 6011))
            or self.training.learning_rates != (1.25e-4, 2.5e-4, 5e-4)
            or self.training.entropy_coefficients != (0.01, 0.02)
        ):
            raise ValueError("canonical Stage 6 training protocol changed")
        if (
            self.statistics.episodes_per_method_partner_seed_layout != 500
            or self.statistics.bootstrap_resamples != 10_000
            or self.statistics.correction != "holm"
            or self.statistics.held_out_unit != "reward_vector"
        ):
            raise ValueError("canonical Stage 6 evaluation protocol changed")
        return self


class TaskEvent(FrozenEstablishedModel):
    name: str
    actor: Literal["ego", "partner", "environment"]
    success: bool = True
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)


class CommitmentTraceStep(FrozenEstablishedModel):
    episode_id: str
    partner_id: str
    reward_vector_id: str
    layout_id: str
    environment_key: str
    work_unit: int = Field(ge=0)
    step: int = Field(ge=0)
    ego_observation: tuple[float, ...] = ()
    ego_action: int | str
    visible_partner_action: int | str | None = None
    reward: float = 0.0
    events: tuple[TaskEvent, ...] = ()
    high_level_events: tuple[str, ...] = ()


class UpstreamRepositoryAudit(FrozenEstablishedModel):
    repository_id: str
    path: str
    expected_commit: str
    observed_commit: str | None
    remote_url: str | None
    exists: bool
    commit_matches: bool
    remote_matches: bool
    passed: bool
    issues: tuple[str, ...] = ()


class UpstreamAudit(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    repositories: tuple[UpstreamRepositoryAudit, ...]
    runtime_projects_present: dict[str, bool]
    passed: bool
    missing_required_assets: tuple[str, ...] = ()


class PartnerCheckpoint(FrozenEstablishedModel):
    partner_id: str
    reward_vector_id: str
    reward_vector_hash: str
    split: SplitName
    seed: int
    layout_id: str
    checkpoint_path: str
    normalized_checkpoint_hash: str
    checkpoint_content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_state_checkpoint_path: str | None = None
    training_state_checkpoint_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stage: PartnerPoolStage | None = None
    requested_transitions: int | None = Field(default=None, ge=1)
    training_request_path: str | None = None
    training_request_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    training_result_path: str | None = None
    training_result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    competence_request_path: str | None = None
    competence_request_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    competence_result_path: str | None = None
    competence_result_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_plan_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    transitions: int = Field(ge=0)
    validation_correct_delivery_rate: float = Field(ge=0, le=1)
    competent: bool

    @model_validator(mode="after")
    def validate_training_state_pair(self) -> PartnerCheckpoint:
        has_path = self.training_state_checkpoint_path is not None
        if has_path != (self.training_state_checkpoint_hash is not None):
            raise ValueError("partner training-state checkpoint path and hash must be paired")
        if (self.training_request_path is None) != (self.training_request_hash is None):
            raise ValueError("partner training request path and hash must be paired")
        if (self.training_result_path is None) != (self.training_result_hash is None):
            raise ValueError("partner training result path and hash must be paired")
        has_competence_path = self.competence_result_path is not None
        if has_competence_path != (self.competence_result_hash is not None):
            raise ValueError("partner competence result path and hash must be paired")
        if (self.competence_request_path is None) != (self.competence_request_hash is None):
            raise ValueError("partner competence request path and hash must be paired")
        return self


class PartnerPoolCandidatePlan(FrozenEstablishedModel):
    candidate_id: str
    split: SplitName
    candidate_index: int = Field(ge=0)
    reward_vector_index: int = Field(ge=0)
    replicate: int = Field(ge=0)
    seed: int = Field(ge=0)
    reward_vector: dict[str, float]
    reward_vector_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    initially_active: bool


class PartnerPoolBuildPlan(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    suite_path: str
    suite_id: str
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout_id: str
    workspace: str
    project_root: str
    upstream_commits: dict[str, str]
    orchestrator_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_source_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    quotas: dict[SplitName, int]
    caps: dict[SplitName, int]
    expansion_block_size: int = Field(ge=1)
    screen_transitions: int = Field(ge=1)
    finalist_transitions: int = Field(ge=1)
    validation_rollouts: int = Field(ge=1)
    minimum_correct_delivery_rate: float = Field(ge=0, le=1)
    competence_environment_keys: tuple[int, ...]
    candidates: tuple[PartnerPoolCandidatePlan, ...]
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_pool_build_plan(self) -> PartnerPoolBuildPlan:
        expected_splits: tuple[SplitName, ...] = ("train", "validation", "evaluation")
        expected_keys = set(expected_splits)
        if set(self.quotas) != expected_keys or set(self.caps) != expected_keys:
            raise ValueError("partner plan quotas and caps must cover exactly all three splits")
        if any(self.quotas[item] <= 0 for item in expected_splits) or any(
            self.caps[item] < self.quotas[item] for item in expected_splits
        ):
            raise ValueError("partner plan quotas and caps are inconsistent")
        if len(self.competence_environment_keys) != self.validation_rollouts:
            raise ValueError("partner plan competence keys do not match the rollout count")
        identifiers = tuple(item.candidate_id for item in self.candidates)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("partner build candidate identifiers must be unique")
        seeds = tuple(item.seed for item in self.candidates)
        if len(seeds) != len(set(seeds)):
            raise ValueError("partner build seeds must be globally unique")
        for split in ("train", "validation", "evaluation"):
            items = tuple(item for item in self.candidates if item.split == split)
            if len(items) != self.caps[split]:
                raise ValueError(f"partner plan does not materialize the {split} cap")
            if tuple(item.candidate_index for item in items) != tuple(range(self.caps[split])):
                raise ValueError(f"partner plan {split} candidate order is not contiguous")
            expected_activation = tuple(
                index < self.quotas[split] for index in range(self.caps[split])
            )
            if tuple(item.initially_active for item in items) != expected_activation:
                raise ValueError(f"partner plan does not activate the first {split} quota")
            if any(
                item.reward_vector_index != item.candidate_index // 2
                or item.replicate != item.candidate_index % 2
                for item in items
            ):
                raise ValueError(f"partner plan {split} candidates break two-seed grouping")
            pairs = tuple(zip(items[::2], items[1::2], strict=True))
            if any(
                left.reward_vector_hash != right.reward_vector_hash
                or left.reward_vector != right.reward_vector
                for left, right in pairs
            ):
                raise ValueError(f"partner plan {split} seed pairs use different reward vectors")
            vector_hashes = tuple(left.reward_vector_hash for left, _right in pairs)
            if len(vector_hashes) != len(set(vector_hashes)):
                raise ValueError(f"partner plan {split} repeats a reward-vector pair")
        return self


class PartnerCandidateLedgerEntry(FrozenEstablishedModel):
    candidate_id: str
    status: PartnerCandidateStatus
    active: bool
    attempts: int = Field(default=0, ge=0)
    last_error: str | None = None
    screen_checkpoint: PartnerCheckpoint | None = None
    finalist_checkpoint: PartnerCheckpoint | None = None


class PartnerPoolBuildLedger(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    entries: tuple[PartnerCandidateLedgerEntry, ...]
    frozen_bundle_path: str | None = None
    frozen_bundle_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    updated_at_utc: str

    @model_validator(mode="after")
    def validate_frozen_pair(self) -> PartnerPoolBuildLedger:
        has_path = self.frozen_bundle_path is not None
        if has_path != (self.frozen_bundle_hash is not None):
            raise ValueError("frozen bundle path and hash must be supplied together")
        identifiers = tuple(item.candidate_id for item in self.entries)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("partner ledger candidate identifiers must be unique")
        return self


class PartnerPoolSplitStatus(FrozenEstablishedModel):
    split: SplitName
    quota: int
    cap: int
    active: int
    eligible: int
    rejected: int
    failed: int
    pending: int
    quota_met: bool
    cap_exhausted: bool


class PartnerPoolBuildStatus(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    splits: tuple[PartnerPoolSplitStatus, ...]
    complete: bool
    frozen: bool
    unresolved_failures: int = Field(ge=0)


class FrozenPartnerPoolManifest(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    suite_id: str
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout_id: str
    split: SplitName
    selection_policy: Literal["exact_quota", "all_processed_eligible"]
    quota: int = Field(ge=1)
    checkpoints: tuple[PartnerCheckpoint, ...]
    checkpoint_hashes: tuple[str, ...]
    frozen_at_utc: str

    @model_validator(mode="after")
    def validate_frozen_pool(self) -> FrozenPartnerPoolManifest:
        if len(self.checkpoints) < self.quota:
            raise ValueError("frozen partner pool does not meet its quota")
        if len(self.checkpoints) != len(self.checkpoint_hashes):
            raise ValueError("frozen checkpoint hashes do not align with checkpoints")
        if any(
            len(item) != 64 or any(character not in "0123456789abcdef" for character in item)
            for item in self.checkpoint_hashes
        ):
            raise ValueError("frozen checkpoint hashes must be lowercase SHA-256 values")
        if self.selection_policy == "exact_quota" and len(self.checkpoints) != self.quota:
            raise ValueError("exact-quota frozen pools must contain exactly their quota")
        if any(item.split != self.split for item in self.checkpoints):
            raise ValueError("frozen pool checkpoint split mismatch")
        if any(
            checkpoint.checkpoint_content_hash != checkpoint_hash
            for checkpoint, checkpoint_hash in zip(
                self.checkpoints, self.checkpoint_hashes, strict=True
            )
        ):
            raise ValueError("frozen pool checkpoint hash does not match its checkpoint")
        return self


class FrozenPartnerPoolBundle(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    suite_id: str
    suite_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    plan_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    layout_id: str
    pool_paths: dict[SplitName, str]
    pool_hashes: dict[SplitName, str]
    leakage_audit_path: str
    leakage_audit_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    publication_summary_path: str
    publication_summary_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_at_utc: str

    @model_validator(mode="after")
    def validate_split_maps(self) -> FrozenPartnerPoolBundle:
        expected = {"train", "validation", "evaluation"}
        if set(self.pool_paths) != expected or set(self.pool_hashes) != expected:
            raise ValueError("frozen partner bundle must contain all three split pools")
        return self


class PartnerPoolManifest(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    suite_id: str
    split: SplitName
    checkpoints: tuple[PartnerCheckpoint, ...]
    quota: int
    quota_met: bool
    reward_vector_hashes: tuple[str, ...]
    source_request_hash: str
    generated_at_utc: str


class ResponseLibrary(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    partner_ids: tuple[str, ...]
    response_ids: tuple[str, ...]
    normalized_values: tuple[tuple[float, ...], ...]
    loss_matrix: tuple[tuple[float, ...], ...]
    adequate_responses: dict[str, tuple[str, ...]]
    response_clusters: dict[str, int]
    response_conflicts: tuple[tuple[str, str], ...]
    adequacy_margin: float
    approximate_oracle_label: Literal["response_library_oracle"] = "response_library_oracle"


class TraceManifest(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    suite_id: str
    trace_path: str
    trace_hash: str
    evidence_policy: EvidencePolicy
    split: SplitName
    partner_ids: tuple[str, ...]
    episode_count: int
    first_work_unit_count: int
    commitment_reached_rate: float = Field(ge=0, le=1)
    post_commitment_excluded: bool
    source_checkpoint_hashes: tuple[str, ...]


class DRIPoint(FrozenEstablishedModel):
    prefix: str
    prior_risk: float
    residual_risk: float
    dri: float | None
    identity_mutual_information_nats: float
    response_signature_mutual_information_nats: float | None
    episode_count: int
    censored_count: int


class DRIEstimate(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    estimator: Literal["gru", "event", "analytical"]
    split: SplitName | Literal["synthetic"]
    evidence_policy: EvidencePolicy | Literal["synthetic_control"]
    points: tuple[DRIPoint, ...]
    temperature: float | None
    cross_fitted: bool
    calibration_trajectory_hashes: tuple[str, ...]
    confirmatory_trajectory_hashes: tuple[str, ...]
    leakage_checks_passed: bool
    treatment_effect: float | None = None


class CandidatePartnerMetrics(FrozenEstablishedModel):
    partner_id: str
    reward_vector_id: str
    response_cluster: int
    competence: float
    best_fixed_response_value: float
    br_prox: float
    lobp_score_nats: float
    commitment_reached_rate: float
    passive_dri: float
    active_dri: float
    trajectory_divergence: float
    br_event_features: tuple[float, ...]

    @model_validator(mode="after")
    def validate_features(self) -> CandidatePartnerMetrics:
        if not self.br_event_features:
            raise ValueError("candidate partner requires best-response event features")
        return self


class MatchingMetricAudit(FrozenEstablishedModel):
    metric: str
    left_value: float | str
    right_value: float | str
    difference: float | None
    margin: float | None
    role: Literal["control", "treatment"]
    passed: bool
    reason: str


class MatchedPopulationAudit(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    contrast: Literal["passive_dri", "active_dri"]
    left_partner_ids: tuple[str, ...]
    right_partner_ids: tuple[str, ...]
    frozen: bool
    discovery_metrics: tuple[MatchingMetricAudit, ...]
    confirmatory_metrics: tuple[MatchingMetricAudit, ...] = ()
    discovery_passed: bool
    confirmatory_passed: bool | None = None
    solver_status: str
    selection_hash: str


class EstablishedTrainingManifest(FrozenEstablishedModel):
    schema_version: Literal[1, 2] = 2
    suite_id: str
    method_id: EstablishedMethod
    layout_id: str
    split: Literal["smoke", "development", "confirmatory"]
    seed: int
    requested_transitions: int
    completed_transitions: int
    checkpoint_path: str
    checkpoint_hash: str
    upstream_commit: str
    configuration_hash: str
    dataset_hashes: tuple[str, ...]
    python_version: str
    jax_version: str
    xla_version: str
    device: str
    resumed: bool
    policy_kind: EstablishedPolicyKind = "ppo"
    deployment_artifact_path: str | None = None
    deployment_artifact_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    resume_checkpoint_path: str | None = None
    parent_checkpoint_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    component_transitions: dict[str, int] = Field(default_factory=dict)
    aggregate_training_transitions: int | None = Field(default=None, ge=0)
    best_validation_checkpoint_path: str | None = None
    best_validation_checkpoint_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    best_validation_metric: float | None = None

    @model_validator(mode="after")
    def validate_training_manifest_version(self) -> EstablishedTrainingManifest:
        if self.schema_version == 1:
            return self
        has_artifact = self.deployment_artifact_path is not None
        if has_artifact != (self.deployment_artifact_hash is not None):
            raise ValueError("deployment artifact path and hash must be supplied together")
        if self.resumed != (self.parent_checkpoint_hash is not None):
            raise ValueError("resumed manifests require checkpoint lineage")
        best_fields = (
            self.best_validation_checkpoint_path,
            self.best_validation_checkpoint_hash,
            self.best_validation_metric,
        )
        if any(value is not None for value in best_fields) and any(
            value is None for value in best_fields
        ):
            raise ValueError("best validation checkpoint fields must be supplied together")
        return self


class EstablishedPolicyEvaluation(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    method_id: EstablishedMethod
    layout_id: str
    policy_seed: int
    partner_ids: tuple[str, ...]
    sparse_return: float
    normalized_return: float
    br_prox: float
    precommitment_dri: float | None
    eventual_dri: float | None
    identity_mutual_information_nats: float
    response_signature_mutual_information_nats: float
    commitment_reached_rate: float
    diagnostic_action_frequency: float
    diagnostic_action_cost: float
    empirical_frontier_distance: float | None
    history_use_effect: float
    response_prediction_calibration_error: float | None
    episodes: int
    protocol: Literal["single_encounter", "reconnaissance"]


class DiagnosticOptionResult(FrozenEstablishedModel):
    option_id: EvidencePolicy
    completes_before_commitment: bool
    conflicting_mode_response_tv: float = Field(ge=0, le=1)
    passive_dri: float | None
    option_dri: float | None
    recipe_prediction_only: bool
    expected_cost: float = Field(ge=0)
    expected_risk_reduction: float = Field(ge=0)
    universal_response_succeeds: bool
    qualifying_partner_diagnostic: bool
    net_value: float


class EmpiricalFrontierPoint(FrozenEstablishedModel):
    source: str
    expected_cost: float
    dri: float
    deterministic: bool
    mixture: tuple[str, str, float] | None = None


class DiagnosticActionAudit(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    layout_id: str
    results: tuple[DiagnosticOptionResult, ...]
    selected_option: EvidencePolicy | None
    deterministic_frontier: tuple[EmpiricalFrontierPoint, ...]
    convexified_frontier: tuple[EmpiricalFrontierPoint, ...]
    passed: bool
    verdict: Literal["qualifying_option_found", "redesign"]


class EstablishedAuditManifest(FrozenEstablishedModel):
    schema_version: Literal[1] = 1
    suite_id: str
    status: Literal["complete", "incomplete", "invalid"]
    scientific_verdict: Stage6Verdict
    upstream_audit_passed: bool
    metric_calibration_passed: bool
    matched_population_passed: bool | None
    diagnostic_action_passed: bool | None
    established_replication_passed: bool | None
    incremental_dri_value_passed: bool | None
    secondary_status: Literal["complete", "secondary_unavailable", "pending"]
    generated_files: tuple[str, ...]
    missing_assets: tuple[str, ...]
    configuration_hash: str
    source_tree_hash: str
    invoked_command: tuple[str, ...]


def load_established_suite_file(path: str | Path) -> EstablishedValidationSuite:
    with Path(path).open(encoding="utf-8") as handle:
        return EstablishedValidationSuite.model_validate(json.load(handle))
