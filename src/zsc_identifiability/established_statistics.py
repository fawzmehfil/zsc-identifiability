"""Cross-fitted incremental-value analysis for established environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def leave_one_reward_vector_out_regression(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Compare nuisance-only and DRI models on held-out reward-vector families."""

    required = {
        "reward_vector_id",
        "method_id",
        "regret",
        "competence",
        "br_div",
        "br_prox",
        "predictability",
        "trajectory_divergence",
        "dri",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("regression rows omit required Stage 6 fields")
    groups = sorted({str(row["reward_vector_id"]) for row in rows})
    methods = sorted({str(row["method_id"]) for row in rows})
    if len(groups) < 2:
        raise ValueError("leave-one-reward-vector-out analysis needs at least two vectors")
    baseline_predictions: list[float] = []
    full_predictions: list[float] = []
    outcomes: list[float] = []
    dri_coefficients: list[float] = []
    fold_rows: list[dict[str, Any]] = []
    for group in groups:
        train = [row for row in rows if str(row["reward_vector_id"]) != group]
        test = [row for row in rows if str(row["reward_vector_id"]) == group]
        x_base_train = _design(train, methods, include_dri=False)
        x_full_train = _design(train, methods, include_dri=True)
        y_train = np.asarray([float(row["regret"]) for row in train])
        base_beta = np.linalg.lstsq(x_base_train, y_train, rcond=None)[0]
        full_beta = np.linalg.lstsq(x_full_train, y_train, rcond=None)[0]
        base_test = _design(test, methods, include_dri=False) @ base_beta
        full_test = _design(test, methods, include_dri=True) @ full_beta
        y_test = np.asarray([float(row["regret"]) for row in test])
        baseline_predictions.extend(base_test.tolist())
        full_predictions.extend(full_test.tolist())
        outcomes.extend(y_test.tolist())
        dri_coefficients.append(float(full_beta[-len(methods)]))
        fold_rows.append(
            {
                "held_out_reward_vector": group,
                "count": len(test),
                "baseline_mse": float(np.mean(np.square(y_test - base_test))),
                "full_mse": float(np.mean(np.square(y_test - full_test))),
                "dri_main_coefficient": dri_coefficients[-1],
            }
        )
    actual = np.asarray(outcomes)
    base = np.asarray(baseline_predictions)
    full = np.asarray(full_predictions)
    denominator = float(np.sum(np.square(actual - actual.mean())))
    base_sse = float(np.sum(np.square(actual - base)))
    full_sse = float(np.sum(np.square(actual - full)))
    base_r2 = 0.0 if denominator <= 1e-15 else 1.0 - base_sse / denominator
    full_r2 = 0.0 if denominator <= 1e-15 else 1.0 - full_sse / denominator
    return {
        "held_out_unit": "reward_vector",
        "folds": fold_rows,
        "baseline_mse": float(np.mean(np.square(actual - base))),
        "full_mse": float(np.mean(np.square(actual - full))),
        "baseline_r2": base_r2,
        "full_r2": full_r2,
        "delta_r2": full_r2 - base_r2,
        "delta_mse": float(np.mean(np.square(actual - full)) - np.mean(np.square(actual - base))),
        "dri_coefficient_min": min(dri_coefficients),
        "dri_coefficient_max": max(dri_coefficients),
        "incremental_value": full_sse < base_sse,
    }


def hierarchical_dri_coefficient_interval(
    rows: Sequence[dict[str, Any]],
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 6173,
) -> dict[str, float]:
    """Bootstrap seed, then partner, then episode for the DRI main coefficient."""

    required = {
        "training_seed",
        "partner_id",
        "episode_id",
        "method_id",
        "dri",
        "regret",
    }
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("hierarchical bootstrap rows omit seed/partner/episode fields")
    methods = sorted({str(row["method_id"]) for row in rows})
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["training_seed"]), []).append(row)
    seeds = sorted(by_seed)
    if len(seeds) < 2:
        raise ValueError("hierarchical bootstrap requires at least two training seeds")
    rng = np.random.Generator(np.random.PCG64(seed))
    coefficients = np.empty(resamples, dtype=np.float64)
    for draw in range(resamples):
        sampled_rows: list[dict[str, Any]] = []
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        for sampled_seed in sampled_seeds:
            seed_rows = by_seed[int(sampled_seed)]
            partners = sorted({str(row["partner_id"]) for row in seed_rows})
            sampled_partners = rng.choice(partners, size=len(partners), replace=True)
            for partner in sampled_partners:
                partner_rows = [row for row in seed_rows if str(row["partner_id"]) == partner]
                episodes = sorted({str(row["episode_id"]) for row in partner_rows})
                sampled_episodes = rng.choice(episodes, size=len(episodes), replace=True)
                for episode in sampled_episodes:
                    sampled_rows.extend(
                        row
                        for row in partner_rows
                        if str(row["episode_id"]) == episode
                    )
        design = _design(sampled_rows, methods, include_dri=True)
        outcomes = np.asarray([float(row["regret"]) for row in sampled_rows])
        coefficient = np.linalg.lstsq(design, outcomes, rcond=None)[0]
        coefficients[draw] = coefficient[-len(methods)]
    point_design = _design(rows, methods, include_dri=True)
    point_outcomes = np.asarray([float(row["regret"]) for row in rows])
    point = float(
        np.linalg.lstsq(point_design, point_outcomes, rcond=None)[0][-len(methods)]
    )
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(coefficients, (alpha, 1.0 - alpha))
    return {
        "estimate": point,
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": confidence_level,
        "resamples": float(resamples),
    }


def hierarchical_paired_bootstrap_difference(
    rows: Sequence[dict[str, Any]],
    left_method: str,
    right_method: str,
    metric: str,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 6173,
) -> dict[str, float]:
    """Paired method contrast with seed/partner/episode resampling."""

    indexed = {
        (
            int(row["training_seed"]),
            str(row["partner_id"]),
            str(row["episode_id"]),
            str(row["method_id"]),
        ): float(row[metric])
        for row in rows
        if str(row["method_id"]) in {left_method, right_method}
    }
    units = sorted({key[:3] for key in indexed})
    paired = [
        unit
        for unit in units
        if (*unit, left_method) in indexed and (*unit, right_method) in indexed
    ]
    if not paired:
        raise ValueError("paired hierarchical bootstrap has no aligned evaluation units")
    differences = {
        unit: indexed[(*unit, left_method)] - indexed[(*unit, right_method)]
        for unit in paired
    }
    seeds = sorted({unit[0] for unit in paired})
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.empty(resamples)
    for draw in range(resamples):
        values: list[float] = []
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            partners = sorted({unit[1] for unit in paired if unit[0] == sampled_seed})
            for partner in rng.choice(partners, size=len(partners), replace=True):
                episodes = [
                    unit
                    for unit in paired
                    if unit[0] == sampled_seed and unit[1] == partner
                ]
                indices = rng.integers(0, len(episodes), size=len(episodes))
                values.extend(differences[episodes[index]] for index in indices)
        draws[draw] = float(np.mean(values))
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(draws, (alpha, 1.0 - alpha))
    return {
        "estimate": float(np.mean(tuple(differences.values()))),
        "lower": float(lower),
        "upper": float(upper),
        "confidence_level": confidence_level,
        "resamples": float(resamples),
    }


def _design(
    rows: Sequence[dict[str, Any]], methods: Sequence[str], *, include_dri: bool
) -> np.ndarray:
    matrix: list[list[float]] = []
    for row in rows:
        method = str(row["method_id"])
        method_indicators = [1.0 if method == item else 0.0 for item in methods[1:]]
        values = [
            1.0,
            float(row["competence"]),
            float(row["br_div"]),
            float(row["br_prox"]),
            float(row["predictability"]),
            float(row["trajectory_divergence"]),
            *method_indicators,
        ]
        if include_dri:
            dri = float(row["dri"])
            values.extend((dri, *[dri * item for item in method_indicators]))
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64)
