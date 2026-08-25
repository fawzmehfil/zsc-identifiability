"""Paired seed inference and strict ranking-reversal checks."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float

    def to_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper}


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> ConfidenceInterval:
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.ndim != 1:
        raise ValueError("paired bootstrap inputs must be equal-length vectors")
    if len(left_values) < 2:
        raise ValueError("paired bootstrap requires at least two seeds")
    differences = left_values - right_values
    rng = np.random.Generator(np.random.PCG64(seed))
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    draws = differences[indices].mean(axis=1)
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(draws, [alpha, 1 - alpha]).tolist()
    return ConfidenceInterval(float(differences.mean()), float(lower), float(upper))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=lambda key: p_values[key])
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, key in enumerate(ordered):
        candidate = min(1.0, (count - index) * p_values[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def paired_sign_flip_p_value(
    left: np.ndarray,
    right: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 1729,
) -> float:
    differences = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    observed = abs(float(differences.mean()))
    rng = np.random.Generator(np.random.PCG64(seed))
    signs = rng.choice((-1.0, 1.0), size=(resamples, len(differences)))
    null = np.abs((signs * differences).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (resamples + 1))


def strict_ranking_reversals(
    rows: list[dict[str, Any]],
    left_cell: str,
    right_cell: str,
    *,
    metric: str = "team_return",
    resamples: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 1729,
) -> dict[str, Any]:
    methods = sorted({str(row["method_id"]) for row in rows})
    indexed = {
        (str(row["cell_id"]), str(row["method_id"]), int(row["seed"])): float(row[metric])
        for row in rows
    }
    comparisons: list[dict[str, Any]] = []
    raw_p_values: dict[str, float] = {}
    intervals: dict[str, tuple[ConfidenceInterval, ConfidenceInterval]] = {}
    for first, second in combinations(methods, 2):
        seeds = sorted(
            {key[2] for key in indexed if key[0] == left_cell and key[1] == first}
            & {key[2] for key in indexed if key[0] == left_cell and key[1] == second}
            & {key[2] for key in indexed if key[0] == right_cell and key[1] == first}
            & {key[2] for key in indexed if key[0] == right_cell and key[1] == second}
        )
        if len(seeds) < 2:
            continue
        left_first = np.asarray([indexed[(left_cell, first, item)] for item in seeds])
        left_second = np.asarray([indexed[(left_cell, second, item)] for item in seeds])
        right_first = np.asarray([indexed[(right_cell, first, item)] for item in seeds])
        right_second = np.asarray([indexed[(right_cell, second, item)] for item in seeds])
        pair_id = f"{first}__vs__{second}"
        intervals[pair_id] = (
            paired_bootstrap_interval(
                left_first,
                left_second,
                resamples=resamples,
                confidence_level=confidence_level,
                seed=seed,
            ),
            paired_bootstrap_interval(
                right_first,
                right_second,
                resamples=resamples,
                confidence_level=confidence_level,
                seed=seed + 1,
            ),
        )
        raw_p_values[f"{pair_id}:left"] = paired_sign_flip_p_value(
            left_first, left_second, resamples=resamples, seed=seed
        )
        raw_p_values[f"{pair_id}:right"] = paired_sign_flip_p_value(
            right_first, right_second, resamples=resamples, seed=seed + 1
        )
    adjusted = holm_adjust(raw_p_values)
    for pair_id, (left_interval, right_interval) in intervals.items():
        opposite = left_interval.estimate * right_interval.estimate < 0
        excludes_zero = (left_interval.lower > 0 or left_interval.upper < 0) and (
            right_interval.lower > 0 or right_interval.upper < 0
        )
        significant = adjusted[f"{pair_id}:left"] < 0.05 and adjusted[f"{pair_id}:right"] < 0.05
        comparisons.append(
            {
                "pair": pair_id,
                "left_cell": left_cell,
                "right_cell": right_cell,
                "left_difference": left_interval.to_dict(),
                "right_difference": right_interval.to_dict(),
                "holm_p_left": adjusted[f"{pair_id}:left"],
                "holm_p_right": adjusted[f"{pair_id}:right"],
                "strict_reversal": opposite and excludes_zero and significant,
            }
        )
    return {
        "left_cell": left_cell,
        "right_cell": right_cell,
        "metric": metric,
        "comparisons": comparisons,
        "strict_reversal_count": sum(item["strict_reversal"] for item in comparisons),
    }


def kendall_rank_correlation(left: dict[str, float], right: dict[str, float]) -> float:
    methods = sorted(set(left) & set(right))
    if len(methods) < 2:
        raise ValueError("Kendall correlation requires at least two common methods")
    concordant = 0
    discordant = 0
    for first, second in combinations(methods, 2):
        product = (left[first] - left[second]) * (right[first] - right[second])
        if product > 0:
            concordant += 1
        elif product < 0:
            discordant += 1
    denominator = concordant + discordant
    return 0.0 if denominator == 0 else (concordant - discordant) / denominator
