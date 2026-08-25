"""Stable public API for the exact-model package."""

from __future__ import annotations

from pathlib import Path

from pydantic import ValidationError

from zsc_identifiability.frontier import compute as _compute_frontier
from zsc_identifiability.metrics import compute_distributions, evaluate
from zsc_identifiability.models import FiniteConventionGame, load_game_file
from zsc_identifiability.numeric import Backend
from zsc_identifiability.policy import PolicyNode
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
