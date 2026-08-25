"""Generate disjoint Stage 4 partner pools from Phase 3 populations."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from zsc_identifiability.benchmark_generator import generate as generate_benchmarks
from zsc_identifiability.benchmark_models import (
    GeneratedPopulation,
    load_benchmark_suite_file,
)
from zsc_identifiability.learning_models import (
    GeneratedLearningPools,
    LearningAuditSuite,
    LearningCellPools,
    LearningGame,
    PartnerProfileSpec,
    SplitName,
)
from zsc_identifiability.models import FiniteConventionGame, KernelRow, OutcomeSpec
from zsc_identifiability.numeric import parse_rational


def generate_learning_pools(
    spec: LearningAuditSuite,
    *,
    suite_path: str | Path | None = None,
) -> GeneratedLearningPools:
    """Instantiate train/validation/test mechanisms without changing v1 games."""
    source_path = _resolve_source_path(spec, suite_path)
    source_spec = load_benchmark_suite_file(source_path)
    benchmark = generate_benchmarks(source_spec)
    canonical = {
        item.descriptor.cell_id: item
        for item in benchmark.populations
        if item.descriptor.symmetry_id == "identity"
    }
    missing = set(spec.cells) - set(canonical)
    if missing:
        raise ValueError(f"learning suite references unknown canonical cells: {sorted(missing)}")
    cells: list[LearningCellPools] = []
    for cell_id in spec.cells:
        source = canonical[cell_id]
        cells.append(
            LearningCellPools(
                cell_id=cell_id,
                source_population=source,
                train=_profile_games(source, "train", spec.profiles.train),
                validation=_profile_games(source, "validation", spec.profiles.validation),
                test=_profile_games(source, "test", spec.profiles.test, canonical_test=True),
            )
        )
    result = GeneratedLearningPools(spec, tuple(cells), benchmark.suite_hash)
    audit_learning_pool_leakage(result)
    audit_learning_pool_matching(result)
    return result


def generate_symmetry_pool(
    spec: LearningAuditSuite,
    cell_id: str,
    symmetry_id: str,
    *,
    suite_path: str | Path | None = None,
) -> LearningCellPools:
    """Build independently trainable pools for one structural relabeling."""
    source_path = _resolve_source_path(spec, suite_path)
    benchmark = generate_benchmarks(load_benchmark_suite_file(source_path))
    source = next(
        (
            population
            for population in benchmark.populations
            if population.descriptor.cell_id == cell_id
            and population.descriptor.symmetry_id == symmetry_id
        ),
        None,
    )
    if source is None:
        raise ValueError(f"unknown symmetry population: {cell_id!r}/{symmetry_id!r}")
    return LearningCellPools(
        cell_id=f"{cell_id}--symmetry-{symmetry_id}",
        source_population=source,
        train=_profile_games(source, "train", spec.profiles.train),
        validation=_profile_games(source, "validation", spec.profiles.validation),
        test=_profile_games(source, "test", spec.profiles.test, canonical_test=True),
    )


def generate_evaluation_variant(
    spec: LearningAuditSuite,
    cell_id: str,
    variant_id: str,
    *,
    suite_path: str | Path | None = None,
) -> LearningGame:
    """Load one Phase 3 sweep variant for frozen-policy evaluation only."""
    source_path = _resolve_source_path(spec, suite_path)
    benchmark = generate_benchmarks(load_benchmark_suite_file(source_path))
    source = next(
        (
            population
            for population in benchmark.populations
            if population.descriptor.cell_id == cell_id
            and population.descriptor.symmetry_id == variant_id
        ),
        None,
    )
    if source is None:
        raise ValueError(f"unknown evaluation variant: {cell_id!r}/{variant_id!r}")
    return _learning_game(
        source,
        "test",
        spec.profiles.test[0],
        canonical_test=True,
    )


def make_smoke_pool(cell: LearningCellPools) -> LearningCellPools:
    """Create the deterministic q=1, zero-intervention-cost implementation gate."""
    profile = PartnerProfileSpec(
        profile_id="smoke_q1_p1",
        reliability="1",
        nuisance_probability="1",
    )
    base = _learning_game(cell.source_population, "train", profile, canonical_test=False)
    game = base.game.model_copy(
        update={
            "game_id": f"{base.game.game_id}--zero-cost",
            "kernels": tuple(
                row.model_copy(
                    update={
                        "outcomes": tuple(
                            outcome.model_copy(update={"cost": "0"}) for outcome in row.outcomes
                        )
                    }
                )
                for row in base.game.kernels
            ),
            "metadata": {**base.game.metadata, "learning_gate": "smoke"},
        }
    )
    smoke = LearningGame(
        cell_id=cell.cell_id,
        split="train",
        profile_id=profile.profile_id,
        source_population_id=cell.source_population.descriptor.population_id,
        partner_identity_prefix=f"smoke:{profile.profile_id}",
        game=game,
        commitment_states=base.commitment_states,
        dynamics_hash=normalized_dynamics_hash(game),
    )
    validation = LearningGame(
        cell_id=smoke.cell_id,
        split="validation",
        profile_id="smoke_validation_q1_p1",
        source_population_id=smoke.source_population_id,
        partner_identity_prefix="smoke-validation:q1-p1",
        game=game,
        commitment_states=smoke.commitment_states,
        dynamics_hash=smoke.dynamics_hash,
    )
    return LearningCellPools(
        cell_id=cell.cell_id,
        source_population=cell.source_population,
        train=(smoke,),
        validation=(validation,),
        test=(validation,),
    )


def audit_learning_pool_leakage(pools: GeneratedLearningPools) -> dict[str, object]:
    """Fail on exact dynamics reuse or runtime metadata leakage."""
    cell_reports: dict[str, object] = {}
    for cell in pools.cells:
        train_hashes = {item.dynamics_hash for item in cell.train}
        validation_hashes = {item.dynamics_hash for item in cell.validation}
        test_hashes = {item.dynamics_hash for item in cell.test}
        if train_hashes & validation_hashes:
            raise ValueError(f"train/validation dynamics leak in cell {cell.cell_id!r}")
        if train_hashes & test_hashes:
            raise ValueError(f"train/test dynamics leak in cell {cell.cell_id!r}")
        if validation_hashes & test_hashes:
            raise ValueError(f"validation/test dynamics leak in cell {cell.cell_id!r}")
        forbidden = {
            "mode",
            "profile",
            "response_signature",
            "cell",
            "population_id",
            "dri",
        }
        visible = set(cell.source_population.descriptor.runtime_visible_fields)
        overlap = forbidden & visible
        if overlap:
            raise ValueError(
                f"runtime-visible metadata leak in {cell.cell_id!r}: {sorted(overlap)}"
            )
        cell_reports[cell.cell_id] = {
            "train_hashes": sorted(train_hashes),
            "validation_hashes": sorted(validation_hashes),
            "test_hashes": sorted(test_hashes),
            "passed": True,
        }
    return {"passed": True, "cells": cell_reports}


def audit_learning_pool_matching(pools: GeneratedLearningPools) -> dict[str, object]:
    """Verify that declared treatment pairs share all learning nuisance structure."""
    by_cell = pools.by_cell()
    reports: list[dict[str, object]] = []
    for comparison in pools.suite.comparisons:
        left = by_cell[comparison.left_cell]
        right = by_cell[comparison.right_cell]
        checks: dict[str, bool] = {
            "profile_counts": tuple(map(len, (left.train, left.validation, left.test)))
            == tuple(map(len, (right.train, right.validation, right.test))),
            "base_team_return": (
                left.source_population.descriptor.base_team_return
                == right.source_population.descriptor.base_team_return
            ),
            "best_response_features": (
                left.source_population.descriptor.best_response_event_features
                == right.source_population.descriptor.best_response_event_features
            ),
        }
        for split in ("train", "validation", "test"):
            left_games = getattr(left, split)
            right_games = getattr(right, split)
            for index, (left_item, right_item) in enumerate(
                zip(left_games, right_games, strict=True)
            ):
                left_game, right_game = left_item.game, right_item.game
                prefix = f"{split}_{index}"
                checks[f"{prefix}_prior"] = left_game.prior_exact() == right_game.prior_exact()
                checks[f"{prefix}_spaces"] = (
                    left_game.states == right_game.states
                    and left_game.observations == right_game.observations
                    and left_game.actions == right_game.actions
                    and left_game.decisions == right_game.decisions
                    and left_game.horizon == right_game.horizon
                )
                checks[f"{prefix}_losses"] = left_game.decision_losses == right_game.decision_losses
                checks[f"{prefix}_cost_geometry"] = _cost_geometry(left_game) == _cost_geometry(
                    right_game
                )
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError(
                f"learning matching contract {comparison.comparison_id!r} failed: {failed}"
            )
        reports.append(
            {
                "comparison_id": comparison.comparison_id,
                "left_cell": comparison.left_cell,
                "right_cell": comparison.right_cell,
                "intended_treatment": comparison.intended_treatment,
                "checks": checks,
                "passed": True,
            }
        )
    return {"passed": all(report["passed"] for report in reports), "comparisons": reports}


def normalized_dynamics_hash(game: FiniteConventionGame) -> str:
    """Hash only behaviorally relevant fields, excluding IDs and metadata."""
    payload = {
        "horizon": game.horizon,
        "prior": [item.probability for item in game.modes],
        "states": game.states,
        "initial_state": game.initial_state,
        "observations": game.observations,
        "actions": [item.model_dump(mode="json") for item in game.actions],
        "decisions": game.decisions,
        "kernels": [
            {
                "time": row.time,
                "state": row.state,
                "action": row.action,
                "mode_index": game.mode_ids.index(row.mode),
                "outcomes": [item.model_dump(mode="json") for item in row.outcomes],
            }
            for row in game.kernels
        ],
        "losses": [
            {
                "mode_index": game.mode_ids.index(item.mode),
                "decision": item.decision,
                "loss": item.loss,
            }
            for item in game.decision_losses
        ],
        "post": [
            {
                "mode_index": game.mode_ids.index(item.mode),
                "observations": [value.model_dump(mode="json") for value in item.observations],
            }
            for item in game.post_commitment_observations
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cost_geometry(game: FiniteConventionGame) -> tuple[tuple[int, str, str, tuple[str, ...]], ...]:
    return tuple(
        (
            row.time,
            row.state,
            row.action,
            tuple(sorted(outcome.cost for outcome in row.outcomes)),
        )
        for row in game.kernels
    )


def _resolve_source_path(spec: LearningAuditSuite, suite_path: str | Path | None) -> Path:
    candidate = Path(spec.source_benchmark_suite)
    if candidate.is_absolute():
        return candidate
    if suite_path is not None:
        project_candidate = Path(suite_path).resolve().parents[2] / candidate
        if project_candidate.exists():
            return project_candidate
        local_candidate = Path(suite_path).resolve().parent / candidate
        if local_candidate.exists():
            return local_candidate
    return candidate.resolve()


def _profile_games(
    source: GeneratedPopulation,
    split: SplitName,
    profiles: tuple[PartnerProfileSpec, ...],
    *,
    canonical_test: bool = False,
) -> tuple[LearningGame, ...]:
    return tuple(
        _learning_game(source, split, profile, canonical_test=canonical_test)
        for profile in profiles
    )


def _learning_game(
    source: GeneratedPopulation,
    split: SplitName,
    profile: PartnerProfileSpec,
    *,
    canonical_test: bool,
) -> LearningGame:
    game = (
        source.game if canonical_test else _replace_signal_parameters(source.game, split, profile)
    )
    return LearningGame(
        cell_id=source.descriptor.cell_id,
        split=split,
        profile_id=profile.profile_id,
        source_population_id=source.descriptor.population_id,
        partner_identity_prefix=f"{split}:{profile.profile_id}",
        game=game,
        commitment_states=frozenset(source.descriptor.commitment_states),
        dynamics_hash=normalized_dynamics_hash(game),
    )


def _replace_signal_parameters(
    game: FiniteConventionGame,
    split: SplitName,
    profile: PartnerProfileSpec,
) -> FiniteConventionGame:
    q = parse_rational(profile.reliability)
    nuisance = parse_rational(profile.nuisance_probability)
    grouped: dict[tuple[int, str, str], list[KernelRow]] = {}
    for row in game.kernels:
        grouped.setdefault((row.time, row.state, row.action), []).append(row)
    replacement: dict[tuple[int, str, str, str], KernelRow] = {}
    for rows in grouped.values():
        distributions = [_distribution(row) for row in rows]
        informative = len({tuple(sorted(item.items())) for item in distributions}) > 1
        for row in rows:
            probabilities = _adjust_distribution(
                _distribution(row), q if informative else nuisance, informative
            )
            outcomes = tuple(
                OutcomeSpec(
                    next_state=outcome.next_state,
                    observation=outcome.observation,
                    probability=str(probabilities[outcome.observation]),
                    cost=outcome.cost,
                )
                for outcome in row.outcomes
            )
            replacement[(row.time, row.state, row.action, row.mode)] = row.model_copy(
                update={"outcomes": outcomes}
            )
    profile_slug = profile.profile_id.replace("_", "-")
    return game.model_copy(
        update={
            "game_id": f"{game.game_id}--learn-{split}-{profile_slug}",
            "kernels": tuple(
                replacement[(row.time, row.state, row.action, row.mode)] for row in game.kernels
            ),
            "metadata": {
                **game.metadata,
                "learning_split": split,
                "learning_profile": profile.profile_id,
            },
        }
    )


def _distribution(row: KernelRow) -> dict[str, Fraction]:
    return {outcome.observation: parse_rational(outcome.probability) for outcome in row.outcomes}


def _adjust_distribution(
    distribution: dict[str, Fraction], target: Fraction, informative: bool
) -> dict[str, Fraction]:
    observations = sorted(distribution)
    if len(observations) != 2:
        return distribution
    if informative:
        preferred = max(observations, key=lambda item: (distribution[item], item))
    else:
        preferred = observations[0]
    other = observations[1] if observations[0] == preferred else observations[0]
    return {preferred: target, other: 1 - target}
