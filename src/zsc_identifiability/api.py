"""Stable public API for the exact-model package."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import ValidationError

from zsc_identifiability.benchmark_audit import (
    audit_pair as _audit_population_pair,
)
from zsc_identifiability.benchmark_audit import (
    audit_shortcuts as _audit_shortcuts,
)
from zsc_identifiability.benchmark_generator import generate as _generate_benchmark_suite
from zsc_identifiability.benchmark_models import (
    BenchmarkRunManifest,
    GeneratedBenchmarkSet,
    GeneratedPopulation,
    MatchedBenchmarkSuite,
    MatchingAudit,
    MatchingContractSpec,
    PopulationMetrics,
    ShortcutAudit,
    load_benchmark_suite_file,
)
from zsc_identifiability.benchmark_runner import execute_benchmark_suite
from zsc_identifiability.established_divergence import estimate_prefix_tv_curves
from zsc_identifiability.established_dri import estimate_event_dri
from zsc_identifiability.established_matching import select_matched_population_pair
from zsc_identifiability.established_models import (
    CandidatePartnerMetrics,
    DRIEstimate,
    EstablishedAuditManifest,
    EstablishedPolicyEvaluation,
    EstablishedTrainingManifest,
    EstablishedValidationSuite,
    FrozenPartnerPoolBundle,
    MatchedPopulationAudit,
    MatchingSpec,
    PartnerPoolBuildLedger,
    PartnerPoolBuildPlan,
    PartnerPoolBuildStatus,
    PartnerPoolManifest,
    ResponseLibrary,
    SplitName,
    UpstreamAudit,
    load_established_suite_file,
)
from zsc_identifiability.established_official_analysis import (
    analyze_official_checkpoint_audit,
)
from zsc_identifiability.established_official_analysis import (
    audit_official_estimator_calibration as _audit_official_estimator_calibration,
)
from zsc_identifiability.established_official_analysis import (
    build_official_response_library as _build_official_response_library,
)
from zsc_identifiability.established_official_analysis import (
    estimate_official_pairwise_dri as _estimate_official_pairwise_dri,
)
from zsc_identifiability.established_official_assets import (
    prepare_official_asset_lock as _prepare_official_asset_lock,
)
from zsc_identifiability.established_official_assets import (
    sync_official_assets as _sync_official_assets,
)
from zsc_identifiability.established_official_models import (
    OfficialAssetInventory,
    OfficialAssetLock,
    OfficialCheckpointAuditManifest,
    OfficialCheckpointAuditSuiteV2,
    OfficialResponseValueMatrix,
    OfficialRolloutLedger,
    OfficialRolloutPlan,
    OfficialTraceIndex,
    PairwiseIdentifiabilityRow,
)
from zsc_identifiability.established_official_models import (
    load_official_checkpoint_suite as _load_official_checkpoint_suite,
)
from zsc_identifiability.established_official_reporting import (
    run_complete_official_checkpoint_analysis as _run_complete_official_checkpoint_analysis,
)
from zsc_identifiability.established_official_rollouts import (
    prepare_official_rollouts as _prepare_official_rollouts,
)
from zsc_identifiability.established_official_rollouts import (
    run_official_rollouts as _run_official_rollouts,
)
from zsc_identifiability.established_partner_pools import (
    freeze_partner_pools as _freeze_partner_pools,
)
from zsc_identifiability.established_partner_pools import (
    get_partner_pool_status as _get_partner_pool_status,
)
from zsc_identifiability.established_partner_pools import (
    prepare_partner_pool_build as _prepare_partner_pool_build,
)
from zsc_identifiability.established_partner_pools import (
    run_partner_pool_build as _run_partner_pool_build,
)
from zsc_identifiability.established_partners import (
    generate_partner_pool_manifest,
    load_partner_checkpoints,
)
from zsc_identifiability.established_predictability import (
    estimate_lobp_action_oracle_from_trace_files,
)
from zsc_identifiability.established_response import build_response_library_from_values
from zsc_identifiability.established_runner import execute_established_audit
from zsc_identifiability.established_runtime import validate_upstreams
from zsc_identifiability.established_trace_estimation import (
    estimate_dri_curve_from_trace_files,
)
from zsc_identifiability.frontier import compute as _compute_frontier
from zsc_identifiability.learning_env import VectorConventionEnvironment
from zsc_identifiability.learning_models import (
    GeneratedLearningPools,
    LearnedPolicyEvaluation,
    LearningAuditManifest,
    LearningAuditSuite,
    LearningCellPools,
    LearningGame,
    LearningMethodSpec,
    TrainingRunManifest,
    load_learning_suite_file,
)
from zsc_identifiability.learning_pools import generate_learning_pools as _generate_learning_pools
from zsc_identifiability.metrics import compute_distributions, evaluate
from zsc_identifiability.models import FiniteConventionGame, load_game_file
from zsc_identifiability.numeric import Backend
from zsc_identifiability.policy import PolicyNode
from zsc_identifiability.population_metrics import compute as _compute_population_metrics
from zsc_identifiability.results import (
    FrontierResult,
    HistoryDistributions,
    PolicyEvaluation,
    PolicySolution,
    RunManifest,
    ValidationReport,
)
from zsc_identifiability.runner import execute_suite
from zsc_identifiability.solver import ActionClass, Objective, solve


def load_game(path: str | Path) -> FiniteConventionGame:
    return load_game_file(path)


def validate_game(game: FiniteConventionGame) -> ValidationReport:
    try:
        validated = FiniteConventionGame.model_validate(game.model_dump())
    except ValidationError as exc:
        return ValidationReport(False, getattr(game, "game_id", None), (str(exc),))
    return ValidationReport(True, validated.game_id)


def solve_bayes(
    game: FiniteConventionGame,
    action_class: ActionClass = "task",
    objective: Objective = "net_regret",
    backend: Backend = "fraction",
) -> PolicySolution:
    return solve(game, action_class, objective, backend)


def evaluate_policy(
    game: FiniteConventionGame, policy: PolicyNode, backend: Backend = "fraction"
) -> PolicyEvaluation:
    return evaluate(game, policy, backend)


def compute_frontier(
    game: FiniteConventionGame,
    action_class: ActionClass = "task",
    backend: Backend = "fraction",
) -> FrontierResult:
    return _compute_frontier(game, action_class, backend)


def compute_history_distributions(
    game: FiniteConventionGame, policy: PolicyNode, backend: Backend = "fraction"
) -> HistoryDistributions:
    return compute_distributions(game, policy, backend)


def run_suite(suite_config: str | Path, output_dir: str | Path) -> RunManifest:
    return execute_suite(suite_config, output_dir)


def load_benchmark_suite(path: str | Path) -> MatchedBenchmarkSuite:
    return load_benchmark_suite_file(path)


def generate_benchmark_suite(
    spec: MatchedBenchmarkSuite,
    backend: Backend = "fraction",
) -> GeneratedBenchmarkSet:
    return _generate_benchmark_suite(spec, backend)


def compute_population_metrics(
    population: GeneratedPopulation,
    backend: Backend = "fraction",
) -> PopulationMetrics:
    return _compute_population_metrics(population, backend)


def audit_population_pair(
    left: GeneratedPopulation,
    right: GeneratedPopulation,
    contract: MatchingContractSpec,
    backend: Backend = "fraction",
) -> MatchingAudit:
    return _audit_population_pair(left, right, contract, backend)


def audit_shortcuts(
    population: GeneratedPopulation,
    backend: Backend = "fraction",
) -> ShortcutAudit:
    return _audit_shortcuts(population, backend)


def run_benchmark_suite(
    suite_config: str | Path,
    output_dir: str | Path,
) -> BenchmarkRunManifest:
    return execute_benchmark_suite(suite_config, output_dir)


def load_learning_suite(path: str | Path) -> LearningAuditSuite:
    return load_learning_suite_file(path)


def generate_learning_pools(
    spec: LearningAuditSuite,
    *,
    suite_path: str | Path | None = None,
) -> GeneratedLearningPools:
    return _generate_learning_pools(spec, suite_path=suite_path)


def make_vector_environment(
    games: tuple[LearningGame, ...],
    seed: int,
    num_envs: int,
    *,
    action_class: str = "task",
    loss_scale: float = 40.0,
) -> VectorConventionEnvironment:

    if action_class not in {"passive", "task"}:
        raise ValueError(f"unsupported learning action class: {action_class!r}")
    return VectorConventionEnvironment(
        games,
        seed,
        num_envs,
        action_class=action_class,  # type: ignore[arg-type]
        loss_scale=loss_scale,
    )


def train_learning_method(
    suite: LearningAuditSuite,
    method: LearningMethodSpec,
    cell: LearningCellPools,
    output_dir: str | Path,
    *,
    seed: int,
    transitions: int,
    resume_checkpoint: str | Path | None = None,
) -> TrainingRunManifest:
    from zsc_identifiability.learning_trainer import train_method

    return train_method(
        suite,
        method,
        cell,
        output_dir,
        seed=seed,
        transitions=transitions,
        resume_checkpoint=resume_checkpoint,
    )


def evaluate_learned_policy(
    checkpoint: str | Path,
    population: LearningGame,
    mode: str = "greedy",
) -> LearnedPolicyEvaluation:
    from zsc_identifiability.learning_evaluation import evaluate_neural_policy_exact
    from zsc_identifiability.learning_trainer import load_checkpoint

    if mode not in {"greedy", "stochastic"}:
        raise ValueError(f"unsupported learned-policy evaluation mode: {mode!r}")
    model, payload = load_checkpoint(checkpoint)
    method_id = str(payload["method"]["method_id"])
    action_class = "passive" if method_id == "gru_ppo_passive" else "task"
    return evaluate_neural_policy_exact(
        model,
        population,
        method_id=method_id,
        mode=mode,  # type: ignore[arg-type]
        action_class=action_class,
        identity_label_response_classes=tuple(
            int(item) for item in payload["partner_response_classes"]
        ),
    )


def run_learning_audit(
    suite_config: str | Path,
    output_dir: str | Path,
    *,
    runs_dir: str | Path | None = None,
    rescue_runs_dir: str | Path | None = None,
) -> LearningAuditManifest:
    from zsc_identifiability.learning_runner import execute_learning_audit

    return execute_learning_audit(
        suite_config,
        output_dir,
        runs_dir=runs_dir,
        rescue_runs_dir=rescue_runs_dir,
    )


def load_established_suite(path: str | Path) -> EstablishedValidationSuite:
    return load_established_suite_file(path)


def validate_upstream_runtime(
    spec: EstablishedValidationSuite,
    project_root: str | Path,
) -> UpstreamAudit:
    return validate_upstreams(spec, project_root)


def generate_partner_population(
    spec: EstablishedValidationSuite,
    split: str,
    *,
    checkpoint_index: str | Path | None = None,
) -> PartnerPoolManifest:
    if split not in {"train", "validation", "evaluation"}:
        raise ValueError(f"invalid established partner split: {split!r}")
    checkpoints = () if checkpoint_index is None else load_partner_checkpoints(checkpoint_index)
    return generate_partner_pool_manifest(
        spec,
        split,  # type: ignore[arg-type]
        checkpoints,
    )


def prepare_partner_pool_build(
    suite: EstablishedValidationSuite,
    layout: str,
    workspace: str | Path,
    *,
    suite_path: str | Path,
    project_root: str | Path | None = None,
) -> PartnerPoolBuildPlan:
    return _prepare_partner_pool_build(
        suite,
        suite_path=suite_path,
        layout=layout,
        workspace=workspace,
        project_root=project_root,
    )


def run_partner_pool_build(
    plan: PartnerPoolBuildPlan | str | Path,
    splits: tuple[str, ...] = ("train", "validation", "evaluation"),
    workers: int = 1,
) -> PartnerPoolBuildLedger:
    if any(item not in {"train", "validation", "evaluation"} for item in splits):
        raise ValueError("partner-pool run contains an invalid split")
    return _run_partner_pool_build(
        plan,
        splits=tuple(cast(SplitName, item) for item in splits),
        workers=workers,
    )


def get_partner_pool_status(
    plan: PartnerPoolBuildPlan | str | Path,
) -> PartnerPoolBuildStatus:
    return _get_partner_pool_status(plan)


def freeze_partner_pools(
    plan: PartnerPoolBuildPlan | str | Path,
) -> FrozenPartnerPoolBundle:
    return _freeze_partner_pools(plan)


def build_response_library(
    values: dict[str, dict[str, float]],
    *,
    adequacy_margin: float = 0.02,
    response_clusters: dict[str, int] | None = None,
) -> ResponseLibrary:
    return build_response_library_from_values(
        values,
        adequacy_margin=adequacy_margin,
        response_clusters=response_clusters,
    )


def estimate_precommitment_dri(
    calibration_histories: tuple[tuple[str, ...], ...],
    calibration_labels: tuple[int, ...],
    confirmatory_histories: tuple[tuple[str, ...], ...],
    prior: tuple[float, ...],
    loss_matrix: tuple[tuple[float, ...], ...],
    *,
    response_signatures: tuple[str, ...] | None = None,
    confirmatory_labels: tuple[int, ...] | None = None,
) -> DRIEstimate:
    return estimate_event_dri(
        calibration_histories,
        calibration_labels,
        confirmatory_histories,
        prior,
        loss_matrix,
        response_signatures=response_signatures,
        confirmatory_labels=confirmatory_labels,
    )


def collect_commitment_traces(
    calibration_trace: str | Path,
    validation_trace: str | Path,
    confirmatory_trace: str | Path,
    response_library: ResponseLibrary,
    suite: EstablishedValidationSuite,
    *,
    estimator: str = "event",
) -> DRIEstimate:
    """Analyze collected trace files without importing the JAX runtime."""

    return estimate_dri_curve_from_trace_files(
        calibration_trace,
        validation_trace,
        confirmatory_trace,
        response_library,
        suite.dri_estimator,
        estimator=estimator,
    )


def select_matched_populations(
    metrics: tuple[CandidatePartnerMetrics, ...],
    contract: MatchingSpec,
    *,
    contrast: str = "passive_dri",
) -> MatchedPopulationAudit:
    return select_matched_population_pair(metrics, contract, contrast)


def estimate_established_predictability(
    calibration_trace: str | Path,
    confirmatory_trace: str | Path,
) -> dict[str, float | int | str]:
    return estimate_lobp_action_oracle_from_trace_files(
        calibration_trace,
        confirmatory_trace,
    )


def estimate_established_divergence(
    trace_path: str | Path,
    *,
    response_signatures: dict[str, str | int] | None = None,
) -> dict[str, object]:
    return estimate_prefix_tv_curves(
        trace_path,
        response_signatures=response_signatures,
    )


def train_established_method(
    manifest_path: str | Path,
) -> EstablishedTrainingManifest:
    """Load a completed isolated-runtime training manifest.

    Training itself is dispatched through the `established train-method` CLI so
    the main Python 3.12 process never imports JAX or upstream packages.
    """

    import json

    return EstablishedTrainingManifest.model_validate(
        json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    )


def evaluate_established_policy(
    evaluation_path: str | Path,
) -> EstablishedPolicyEvaluation:
    import json

    return EstablishedPolicyEvaluation.model_validate(
        json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    )


def run_established_audit(
    suite_config: str | Path,
    output_dir: str | Path,
    *,
    state_dir: str | Path | None = None,
) -> EstablishedAuditManifest:
    return execute_established_audit(suite_config, output_dir, state_dir=state_dir)


def audit_learning_smoke_matrix(
    suite_config: str | Path,
    runs_dir: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    from zsc_identifiability.learning_runner import audit_smoke_matrix

    return audit_smoke_matrix(suite_config, runs_dir, output_path)


def load_official_checkpoint_suite(path: str | Path) -> OfficialCheckpointAuditSuiteV2:
    return _load_official_checkpoint_suite(path)


def prepare_official_asset_lock(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    workspace: str | Path,
) -> OfficialAssetLock:
    return _prepare_official_asset_lock(suite, workspace)


def sync_official_assets(
    lock: OfficialAssetLock | str | Path,
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> OfficialAssetInventory:
    return _sync_official_assets(lock, suite)


def prepare_official_rollouts(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    assets: OfficialAssetInventory | str | Path,
    workspace: str | Path,
) -> OfficialRolloutPlan:
    return _prepare_official_rollouts(suite, assets, workspace)


def run_official_rollouts(
    plan: OfficialRolloutPlan | str | Path,
    workers: int = 2,
    resume: bool = True,
) -> OfficialRolloutLedger:
    return _run_official_rollouts(plan, workers=workers, resume=resume)


def build_official_response_library(
    response_results: str | Path | tuple[str | Path, ...],
    config: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> tuple[OfficialResponseValueMatrix, ...]:
    return _build_official_response_library(response_results, config)


def estimate_official_pairwise_dri(
    trace_index: OfficialTraceIndex | str | Path,
    response_library: OfficialResponseValueMatrix,
    config: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> tuple[PairwiseIdentifiabilityRow, ...]:
    return _estimate_official_pairwise_dri(trace_index, response_library, config)


def audit_official_estimator_calibration(
    trace_index: OfficialTraceIndex | str | Path,
    response_libraries: tuple[OfficialResponseValueMatrix, ...],
    pairwise_rows: tuple[PairwiseIdentifiabilityRow, ...],
    config: OfficialCheckpointAuditSuiteV2 | str | Path,
) -> dict[str, object]:
    return _audit_official_estimator_calibration(
        trace_index, response_libraries, pairwise_rows, config
    )


def run_official_checkpoint_analysis(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    plan: OfficialRolloutPlan | str | Path,
    ledger: OfficialRolloutLedger | str | Path,
    output_dir: str | Path,
) -> OfficialCheckpointAuditManifest:
    return _run_complete_official_checkpoint_analysis(suite, plan, ledger, output_dir)


def analyze_official_audit(
    suite: OfficialCheckpointAuditSuiteV2 | str | Path,
    results: dict[str, object],
) -> OfficialCheckpointAuditManifest:
    return analyze_official_checkpoint_audit(suite, results)
