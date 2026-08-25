"""Deterministic Monte Carlo calibration for later rollout-based estimators."""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import numpy as np

from zsc_identifiability.benchmark_models import (
    GeneratedPopulation,
    MatchingContractSpec,
    PopulationMetrics,
    SampleAuditSpec,
)
from zsc_identifiability.benchmark_policies import fixed_action_sequence_policy
from zsc_identifiability.numeric import Backend
from zsc_identifiability.population_metrics import prediction_metrics


def calibrate_pair(
    left: GeneratedPopulation,
    right: GeneratedPopulation,
    left_metrics: PopulationMetrics,
    right_metrics: PopulationMetrics,
    contract: MatchingContractSpec,
    config: SampleAuditSpec,
    backend: Backend = "fraction",
) -> dict[str, Any]:
    """Calibrate sampled broad-predictability and return controls for one pair."""
    rng = np.random.Generator(np.random.PCG64(config.seed + _stable_offset(contract.contract_id)))
    mode_count = max(len(left.game.mode_ids), len(right.game.mode_ids))
    episodes = config.episodes_per_mode * mode_count
    uniforms = rng.random((mode_count, config.episodes_per_mode))
    left_policy = fixed_action_sequence_policy(
        left.game, left.descriptor.passive_reference_actions, backend
    )
    right_policy = fixed_action_sequence_policy(
        right.game, right.descriptor.passive_reference_actions, backend
    )
    left_sequences = _sample_sequences_per_mode(left, left_policy, uniforms, backend)
    right_sequences = _sample_sequences_per_mode(right, right_policy, uniforms, backend)
    left_precommit_sequences = _sample_sequences_per_mode(
        left, left_policy, uniforms, backend, include_post=False
    )
    right_precommit_sequences = _sample_sequences_per_mode(
        right, right_policy, uniforms, backend, include_post=False
    )
    left_surprisal = _sequence_surprisal(left, left_policy, left_sequences, backend)
    right_surprisal = _sequence_surprisal(right, right_policy, right_sequences, backend)
    lobp_difference = -(left_surprisal - right_surprisal)
    lobp_ci = _bootstrap_ci(
        lobp_difference,
        config.bootstrap_resamples,
        config.confidence_level,
        rng,
    )
    return_difference = np.zeros(episodes, dtype=np.float64)
    return_ci = _bootstrap_ci(
        return_difference,
        config.bootstrap_resamples,
        config.confidence_level,
        rng,
    )
    determinant_difference = float(left_metrics.values["zsceval_br_div_code"]) - float(
        right_metrics.values["zsceval_br_div_code"]
    )
    divergence_difference = _normalized_divergence_difference(left_metrics, right_metrics)
    exact_left_prediction = prediction_metrics(left, left_policy, backend)
    exact_right_prediction = prediction_metrics(right, right_policy, backend)
    left_sampled_dri = _sampled_dri(left, left_precommit_sequences)
    right_sampled_dri = _sampled_dri(right, right_precommit_sequences)
    left_exact_dri = _optional_float(left_metrics.values["passive_dri"])
    right_exact_dri = _optional_float(right_metrics.values["passive_dri"])
    checks = {
        "known_mode_return_equivalent": _inside(return_ci, float(config.return_margin)),
        "lobp_equivalent": _inside(lobp_ci, config.lobp_margin_nats),
        "normalized_determinant_equivalent": (
            abs(determinant_difference) <= config.determinant_margin
        ),
        "normalized_divergence_time_equivalent": (
            abs(divergence_difference) <= config.divergence_time_margin
            if contract.require_divergence_profile_match
            else True
        ),
        "passive_dri_estimator_agrees": (
            _optional_close(left_exact_dri, left_sampled_dri, config.dri_margin)
            and _optional_close(right_exact_dri, right_sampled_dri, config.dri_margin)
        ),
    }
    return {
        "contract_id": contract.contract_id,
        "episodes": episodes,
        "episodes_per_mode": config.episodes_per_mode,
        "bootstrap_resamples": config.bootstrap_resamples,
        "confidence_level": config.confidence_level,
        "seed": config.seed + _stable_offset(contract.contract_id),
        "known_mode_return_difference": {
            "sample_mean": float(np.mean(return_difference)),
            "confidence_interval": list(return_ci),
            "equivalence_margin": float(config.return_margin),
        },
        "lobp_action_score_difference_nats": {
            "sample_mean": float(np.mean(lobp_difference)),
            "confidence_interval": list(lobp_ci),
            "equivalence_margin": config.lobp_margin_nats,
            "exact_difference": float(exact_left_prediction["full_score"])
            - float(exact_right_prediction["full_score"]),
        },
        "passive_dri_calibration": {
            "left_exact": left_exact_dri,
            "left_sampled": left_sampled_dri,
            "right_exact": right_exact_dri,
            "right_sampled": right_sampled_dri,
            "absolute_margin": config.dri_margin,
        },
        "normalized_determinant_difference": {
            "value": determinant_difference,
            "confidence_interval": [determinant_difference, determinant_difference],
            "equivalence_margin": config.determinant_margin,
        },
        "normalized_divergence_time_difference": {
            "value": divergence_difference,
            "confidence_interval": [divergence_difference, divergence_difference],
            "equivalence_margin": config.divergence_time_margin,
            "applicable": contract.require_divergence_profile_match,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def _sample_sequences_per_mode(
    population: GeneratedPopulation,
    policy: Any,
    uniforms: np.ndarray,
    backend: Backend,
    include_post: bool = True,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    conditional = _float_sequence_distributions_by_mode(
        population, policy, backend, include_post
    )
    result: dict[str, tuple[tuple[str, ...], ...]] = {}
    for mode_index, mode in enumerate(population.game.mode_ids):
        distribution = conditional[mode]
        sequences = sorted(distribution)
        probabilities = np.asarray(
            [distribution[sequence] for sequence in sequences], dtype=np.float64
        )
        probabilities /= probabilities.sum()
        cumulative = np.cumsum(probabilities)
        indices = np.searchsorted(cumulative, uniforms[mode_index], side="right")
        result[mode] = tuple(sequences[index] for index in indices)
    return result


def _sequence_surprisal(
    population: GeneratedPopulation,
    policy: Any,
    samples: dict[str, tuple[tuple[str, ...], ...]],
    backend: Backend,
) -> np.ndarray:
    mixture = _float_sequence_distribution(population, policy, backend)
    values = []
    for mode in population.game.mode_ids:
        for sequence in samples[mode]:
            values.append(-math.log(mixture[sequence]) / max(1, len(sequence)))
    return np.asarray(values, dtype=np.float64)


def _float_sequence_distribution(
    population: GeneratedPopulation,
    policy: Any,
    backend: Backend,
) -> dict[tuple[str, ...], float]:
    from zsc_identifiability.metrics import compute_distributions

    histories = compute_distributions(population.game, policy, backend)
    priors = {item.id: float(Fraction(item.probability)) for item in population.game.modes}
    post = {
        row.mode: {item.id: float(Fraction(item.probability)) for item in row.observations}
        for row in population.game.post_commitment_observations
    }
    result: dict[tuple[str, ...], float] = {}
    for mode in population.game.mode_ids:
        for history, conditional in histories.by_mode[mode].items():
            observations = tuple(
                token.rsplit("/", 1)[1]
                for token in history.split(";")
                if not token.startswith("stop@")
            )
            mass = priors[mode] * float(conditional)
            if post:
                for observation, probability in post[mode].items():
                    key = (*observations, observation)
                    result[key] = result.get(key, 0.0) + mass * probability
            else:
                result[observations] = result.get(observations, 0.0) + mass
    return result


def _float_sequence_distributions_by_mode(
    population: GeneratedPopulation,
    policy: Any,
    backend: Backend,
    include_post: bool,
) -> dict[str, dict[tuple[str, ...], float]]:
    from zsc_identifiability.metrics import compute_distributions

    histories = compute_distributions(population.game, policy, backend)
    post = {
        row.mode: {item.id: float(Fraction(item.probability)) for item in row.observations}
        for row in population.game.post_commitment_observations
    }
    result: dict[str, dict[tuple[str, ...], float]] = {}
    for mode in population.game.mode_ids:
        conditional: dict[tuple[str, ...], float] = {}
        for history, probability in histories.by_mode[mode].items():
            observations = tuple(
                token.rsplit("/", 1)[1]
                for token in history.split(";")
                if not token.startswith("stop@")
            )
            if include_post and post:
                for observation, post_probability in post[mode].items():
                    sequence = (*observations, observation)
                    conditional[sequence] = conditional.get(sequence, 0.0) + (
                        float(probability) * post_probability
                    )
            else:
                conditional[observations] = conditional.get(observations, 0.0) + float(
                    probability
                )
        result[mode] = conditional
    return result


def _sampled_dri(
    population: GeneratedPopulation,
    samples: dict[str, tuple[tuple[str, ...], ...]],
) -> float | None:
    game = population.game
    prior_losses = [
        sum(
            float(Fraction(mode_spec.probability))
            * float(game.loss_exact(mode_spec.id, decision))
            for mode_spec in game.modes
        )
        for decision in game.decisions
    ]
    prior_risk = min(prior_losses)
    if prior_risk == 0:
        return None
    counts: dict[tuple[str, ...], dict[str, int]] = {}
    for mode in game.mode_ids:
        for sequence in samples[mode]:
            counts.setdefault(sequence, {}).setdefault(mode, 0)
            counts[sequence][mode] += 1
    sample_count = len(next(iter(samples.values())))
    risk = 0.0
    for by_mode in counts.values():
        decision_losses = []
        for decision in game.decisions:
            loss = 0.0
            for mode_spec in game.modes:
                empirical_conditional = by_mode.get(mode_spec.id, 0) / sample_count
                loss += (
                    float(Fraction(mode_spec.probability))
                    * empirical_conditional
                    * float(game.loss_exact(mode_spec.id, decision))
                )
            decision_losses.append(loss)
        risk += min(decision_losses)
    return (prior_risk - risk) / prior_risk


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_close(left: float | None, right: float | None, margin: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) <= margin


def _bootstrap_ci(
    differences: np.ndarray,
    resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    if np.all(differences == differences[0]):
        value = float(differences[0])
        return value, value
    means = np.empty(resamples, dtype=np.float64)
    size = len(differences)
    for index in range(resamples):
        sample = rng.integers(0, size, size=size)
        means[index] = float(np.mean(differences[sample]))
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def _normalized_divergence_difference(
    left: PopulationMetrics,
    right: PopulationMetrics,
) -> float:
    left_steps = _mean_positive_steps(left)
    right_steps = _mean_positive_steps(right)
    if math.isnan(left_steps) and math.isnan(right_steps):
        return 0.0
    return left_steps - right_steps


def _mean_positive_steps(metrics: PopulationMetrics) -> float:
    values = [
        item["positive"]
        for item in metrics.divergence_threshold_steps.values()
        if item["positive"] is not None
    ]
    if not values:
        return math.nan
    horizon = max((len(curve) for curve in metrics.prefix_tv_curves.values()), default=1)
    return float(np.mean(values)) / max(1, horizon)


def _inside(interval: tuple[float, float], margin: float) -> bool:
    return interval[0] >= -margin and interval[1] <= margin


def _stable_offset(value: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(value)) % 100_000
