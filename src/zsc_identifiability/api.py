"""Stable public API for the exact-model package."""

from __future__ import annotations

from pathlib import Path

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
