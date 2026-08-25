"""Exact population-level metrics for matched identifiability benchmarks."""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

import numpy as np

from zsc_identifiability.belief import initial_belief
from zsc_identifiability.benchmark_models import GeneratedPopulation, PopulationMetrics
from zsc_identifiability.benchmark_policies import fixed_action_sequence_policy
from zsc_identifiability.frontier import compute as compute_frontier
from zsc_identifiability.metrics import compute_distributions, evaluate
from zsc_identifiability.numeric import Backend, Number, number, zero
from zsc_identifiability.policy import PolicyNode
from zsc_identifiability.solver import solve


def compute(population: GeneratedPopulation, backend: Backend = "fraction") -> PopulationMetrics:
    game = population.game
    descriptor = population.descriptor
    commitment_states = frozenset(descriptor.commitment_states)
    base_return = number(descriptor.base_team_return, backend)
    passive_solution = solve(
        game,
        "passive",
        "information",
        backend,
        commitment_states=commitment_states,
    )
    active_solution = solve(
        game,
        "task",
        "net_regret",
        backend,
        commitment_states=commitment_states,
    )
    active_information = solve(
        game,
        "task",
        "information",
        backend,
        commitment_states=commitment_states,
    )
    reference_policy = fixed_action_sequence_policy(
        game,
        descriptor.passive_reference_actions,
        backend,
    )
    passive_evaluation = evaluate(game, passive_solution.policy, backend)
    active_evaluation = evaluate(game, active_solution.policy, backend)
    active_information_evaluation = evaluate(game, active_information.policy, backend)
    reference_evaluation = evaluate(game, reference_policy, backend)
    response_matrix, best_decisions = _response_confusion_matrix(population, backend)
    fixed_decision = _best_fixed_decision(population, response_matrix, backend)
    cross_play = _cross_play_returns(response_matrix, best_decisions, base_return, backend)
    rahman = _rahman_brdiv(cross_play, backend)
    raw_features = _feature_matrix(population)
    raw_gram = _gram(raw_features)
    raw_determinant = _determinant(raw_gram)
    raw_float_sign, raw_float_logabs = float_determinant(raw_gram)
    code_features = _code_normalize_features(raw_features)
    code_gram = _gram(code_features)
    code_determinant = _determinant(code_gram)
    code_float_sign, code_float_logabs = float_determinant(code_gram)
    representative_indices = _response_representatives(population)
    representative_gram = _gram(tuple(raw_features[index] for index in representative_indices))
    representative_determinant = _determinant(representative_gram)
    active_returns = _per_mode_returns(population, active_solution.policy, base_return, backend)
    known_mode_returns = {mode: base_return for mode in game.mode_ids}
    brprox = {mode: active_returns[mode] / known_mode_returns[mode] for mode in game.mode_ids}
    brprox_values = tuple(brprox[mode] for mode in game.mode_ids)
    prediction = prediction_metrics(population, reference_policy, backend)
    prefix_curves, threshold_steps, deterministic_edp = prefix_tv_metrics(
        population, reference_policy, backend
    )
    conflicting_pairs = _conflicting_pairs(population)
    final_tvs = [curve[-1] for curve in prefix_curves.values() if curve]
    conflicting_final = [
        prefix_curves[key][-1]
        for key in conflicting_pairs
        if key in prefix_curves and prefix_curves[key]
    ]
    frontier = compute_frontier(
        game,
        "task",
        backend,
        commitment_states=commitment_states,
    )
    prior_risk = active_evaluation.prior_risk
    values: dict[str, Number | float | str | None] = {
        "base_team_return": base_return,
        "known_mode_return_mean": _mean(tuple(known_mode_returns.values()), backend),
        "population_competence": _mean(tuple(known_mode_returns.values()), backend),
        "best_fixed_response_value": base_return - prior_risk,
        "best_fixed_response_decision": fixed_decision,
        "passive_oracle_return": (
            base_return
            - passive_evaluation.expected_intervention_cost
            - passive_evaluation.actual_policy_loss
        ),
        "task_active_oracle_return": (
            base_return
            - active_evaluation.expected_intervention_cost
            - active_evaluation.actual_policy_loss
        ),
        "information_only_return": (
            base_return
            - active_information_evaluation.expected_intervention_cost
            - active_information_evaluation.actual_policy_loss
        ),
        "information_only_intervention_cost": (
            active_information_evaluation.expected_intervention_cost
        ),
        "information_only_residual_risk": (
            active_information_evaluation.residual_risk_precommitment
        ),
        "prior_risk": prior_risk,
        "passive_residual_risk": passive_evaluation.residual_risk_precommitment,
        "active_residual_risk": active_information_evaluation.residual_risk_precommitment,
        "active_net_residual_risk": active_evaluation.residual_risk_precommitment,
        "passive_dri": passive_evaluation.dri_precommitment,
        "active_dri": active_information_evaluation.dri_precommitment,
        "active_net_dri": active_evaluation.dri_precommitment,
        "eventual_dri": reference_evaluation.dri_eventual,
        "active_intervention_cost": active_evaluation.expected_intervention_cost,
        "active_net_regret": active_evaluation.net_oracle_regret,
        "identity_mutual_information_bits": (reference_evaluation.identity_mutual_information_bits),
        "decision_signature_mutual_information_bits": (
            reference_evaluation.decision_signature_mutual_information_bits
        ),
        "rahman_brdiv_return": rahman,
        "zsceval_br_div_raw": raw_determinant,
        "zsceval_br_div_code": code_determinant,
        "zsceval_br_div_response_representatives": representative_determinant,
        "brprox_mean": _mean(brprox_values, backend),
        "brprox_median": _median(brprox_values, backend),
        "brprox_iqm": _interquartile_mean(brprox_values, backend),
        "lobp_action_oracle_score_nats": prediction["full_score"],
        "lobp_action_precommit_score_nats": prediction["precommit_score"],
        "lobp_prediction_targets_precommit": prediction["precommit_targets"],
        "lobp_prediction_targets_full": prediction["full_targets"],
        "mean_final_prefix_tv": _mean(tuple(final_tvs), backend) if final_tvs else None,
        "conflicting_pair_mean_final_prefix_tv": (
            _mean(tuple(conflicting_final), backend) if conflicting_final else None
        ),
    }
    per_mode: dict[str, dict[str, Number | float | str | None]] = {}
    for mode in game.mode_ids:
        per_mode[mode] = {
            "known_mode_return": known_mode_returns[mode],
            "competence": known_mode_returns[mode],
            "active_policy_return": active_returns[mode],
            "brprox": brprox[mode],
            "response_signature": descriptor.response_signature_by_mode[mode],
        }
    matrices = {
        "cross_play_returns": _matrix_dict(game.mode_ids, game.mode_ids, cross_play),
        "zsceval_feature_matrix": [list(row) for row in raw_features],
        "zsceval_gram_raw": [list(row) for row in raw_gram],
        "zsceval_float_slogdet_raw": {
            "sign": raw_float_sign,
            "log_abs_determinant": (raw_float_logabs if math.isfinite(raw_float_logabs) else None),
        },
        "zsceval_feature_matrix_code_normalized": [list(row) for row in code_features],
        "zsceval_gram_code_normalized": [list(row) for row in code_gram],
        "zsceval_float_slogdet_code_normalized": {
            "sign": code_float_sign,
            "log_abs_determinant": (
                code_float_logabs if math.isfinite(code_float_logabs) else None
            ),
        },
        "response_representative_indices": list(representative_indices),
        "conflicting_pairs": list(conflicting_pairs),
        "prediction": prediction,
    }
    return PopulationMetrics(
        population_id=descriptor.population_id,
        backend=backend,
        metric_scope="full_population",
        estimator_type="exact" if backend == "fraction" else "numerical",
        applicability_flags={
            "passive_dri": prior_risk > 0,
            "active_dri": prior_risk > 0,
            "eventual_dri": prior_risk > 0,
            "identity_mutual_information": True,
            "decision_signature_mutual_information": True,
            "deterministic_edp": deterministic_edp is not None,
            "post_commitment_evidence": bool(game.post_commitment_observations),
            "response_representative_determinant_is_diagnostic": True,
        },
        values=values,
        per_mode=per_mode,
        response_confusion_matrix=response_matrix,
        brdiv_matrices=matrices,
        prefix_tv_curves=prefix_curves,
        divergence_threshold_steps=threshold_steps,
        deterministic_edp=deterministic_edp,
        passive_policy=passive_solution.policy.to_dict(),
        active_policy=active_solution.policy.to_dict(),
        information_policy=active_information.policy.to_dict(),
        reference_policy=reference_policy.to_dict(),
        active_frontier=frontier.to_dict(),
    )


def prediction_metrics(
    population: GeneratedPopulation,
    policy: PolicyNode,
    backend: Backend = "fraction",
) -> dict[str, float | int]:
    pre = _sequence_distribution(population, policy, backend, include_post=False)
    full = _sequence_distribution(population, policy, backend, include_post=True)
    pre_targets = max((len(sequence) for sequence in pre), default=0)
    full_targets = max((len(sequence) for sequence in full), default=0)
    pre_entropy = _entropy_nats(pre)
    full_entropy = _entropy_nats(full)
    return {
        "precommit_entropy_nats": pre_entropy,
        "full_entropy_nats": full_entropy,
        "precommit_score": -pre_entropy / pre_targets if pre_targets else 0.0,
        "full_score": -full_entropy / full_targets if full_targets else 0.0,
        "precommit_targets": pre_targets,
        "full_targets": full_targets,
    }


def prefix_tv_metrics(
    population: GeneratedPopulation,
    policy: PolicyNode,
    backend: Backend = "fraction",
) -> tuple[
    dict[str, tuple[Number, ...]],
    dict[str, dict[str, int | None]],
    dict[str, int | None] | None,
]:
    game = population.game
    histories = compute_distributions(game, policy, backend)
    sequence_by_mode = {
        mode: _mode_sequence_distribution(histories.by_mode[mode], backend)
        for mode in game.mode_ids
    }
    maximum = max(
        (len(sequence) for values in sequence_by_mode.values() for sequence in values),
        default=0,
    )
    curves: dict[str, tuple[Number, ...]] = {}
    thresholds: dict[str, dict[str, int | None]] = {}
    all_deterministic = all(
        len([probability for probability in values.values() if probability > 0]) == 1
        for values in sequence_by_mode.values()
    )
    edp: dict[str, int | None] | None = {} if all_deterministic else None
    for left_index, left in enumerate(game.mode_ids):
        for right in game.mode_ids[left_index + 1 :]:
            key = f"{left}|{right}"
            values: list[Number] = []
            for length in range(1, maximum + 1):
                left_prefix = _prefix_distribution(sequence_by_mode[left], length, backend)
                right_prefix = _prefix_distribution(sequence_by_mode[right], length, backend)
                values.append(_tv(left_prefix, right_prefix, backend))
            curves[key] = tuple(values)
            thresholds[key] = {
                "positive": _first_step(values, Fraction(0), strict=True),
                "one_half": _first_step(values, Fraction(1, 2)),
                "nine_tenths": _first_step(values, Fraction(9, 10)),
            }
            if edp is not None:
                edp[key] = _first_step(values, Fraction(0), strict=True)
    return curves, thresholds, edp


def _response_confusion_matrix(
    population: GeneratedPopulation,
    backend: Backend,
) -> tuple[dict[str, dict[str, Number]], dict[str, str]]:
    game = population.game
    result: dict[str, dict[str, Number]] = {}
    best: dict[str, str] = {}
    for mode in game.mode_ids:
        row = {
            decision: number(game.loss_exact(mode, decision), backend)
            for decision in game.decisions
        }
        result[mode] = row
        best[mode] = min(row, key=lambda decision: (float(row[decision]), decision))
    return result, best


def _best_fixed_decision(
    population: GeneratedPopulation,
    response_matrix: dict[str, dict[str, Number]],
    backend: Backend,
) -> str:
    priors = dict(
        zip(population.game.mode_ids, initial_belief(population.game, backend), strict=True)
    )
    return min(
        population.game.decisions,
        key=lambda decision: (
            float(
                sum(
                    (
                        priors[mode] * response_matrix[mode][decision]
                        for mode in population.game.mode_ids
                    ),
                    start=zero(backend),
                )
            ),
            decision,
        ),
    )


def _cross_play_returns(
    response_matrix: dict[str, dict[str, Number]],
    best_decisions: dict[str, str],
    base_return: Number,
    backend: Backend,
) -> tuple[tuple[Number, ...], ...]:
    modes = tuple(response_matrix)
    return tuple(
        tuple(
            base_return - response_matrix[partner][best_decisions[response_mode]]
            for partner in modes
        )
        for response_mode in modes
    )


def _rahman_brdiv(matrix: tuple[tuple[Number, ...], ...], backend: Backend) -> Number:
    size = len(matrix)
    if size < 2:
        return zero(backend)
    total = zero(backend)
    for left in range(size):
        for right in range(size):
            if left == right:
                continue
            total += (matrix[left][left] - matrix[left][right]) + (
                matrix[left][left] - matrix[right][left]
            )
    return total / number(str(2 * size * (size - 1)), backend)


def _feature_matrix(population: GeneratedPopulation) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(Fraction(value) for value in population.descriptor.best_response_event_features[mode])
        for mode in population.game.mode_ids
    )


def _code_normalize_features(
    matrix: tuple[tuple[Fraction, ...], ...],
) -> tuple[tuple[Fraction, ...], ...]:
    if not matrix:
        return ()
    maxima = tuple(max(row[column] for row in matrix) for column in range(len(matrix[0])))
    epsilon = Fraction(1, 1000)
    return tuple(
        tuple(value / (maxima[column] + epsilon) for column, value in enumerate(row))
        for row in matrix
    )


def _gram(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[tuple[Fraction, ...], ...]:
    return tuple(
        tuple(
            sum(
                (a * b for a, b in zip(left, right, strict=True)),
                start=Fraction(0),
            )
            for right in matrix
        )
        for left in matrix
    )


def _determinant(matrix: tuple[tuple[Fraction, ...], ...]) -> Fraction:
    size = len(matrix)
    if size == 0:
        return Fraction(1)
    work = [list(row) for row in matrix]
    sign = Fraction(1)
    determinant = Fraction(1)
    for column in range(size):
        pivot = next((row for row in range(column, size) if work[row][column] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            sign *= -1
        pivot_value = work[column][column]
        determinant *= pivot_value
        for row in range(column + 1, size):
            factor = work[row][column] / pivot_value
            for index in range(column, size):
                work[row][index] -= factor * work[column][index]
    return sign * determinant


def _response_representatives(population: GeneratedPopulation) -> tuple[int, ...]:
    seen: set[str] = set()
    result = []
    for index, mode in enumerate(population.game.mode_ids):
        signature = population.descriptor.response_signature_by_mode[mode]
        if signature not in seen:
            seen.add(signature)
            result.append(index)
    return tuple(result)


def _per_mode_returns(
    population: GeneratedPopulation,
    policy: PolicyNode,
    base_return: Number,
    backend: Backend,
) -> dict[str, Number]:
    game = population.game
    histories = compute_distributions(game, policy, backend)
    result: dict[str, Number] = {}
    for mode in game.mode_ids:
        loss = zero(backend)
        for history, probability in histories.by_mode[mode].items():
            decision = histories.decisions_by_history[history]
            loss += probability * number(game.loss_exact(mode, decision), backend)
        result[mode] = base_return - histories.expected_cost_by_mode[mode] - loss
    return result


def _sequence_distribution(
    population: GeneratedPopulation,
    policy: PolicyNode,
    backend: Backend,
    include_post: bool,
) -> dict[tuple[str, ...], float]:
    game = population.game
    histories = compute_distributions(game, policy, backend)
    priors = dict(zip(game.mode_ids, initial_belief(game, backend), strict=True))
    result: dict[tuple[str, ...], float] = defaultdict(float)
    post = {
        row.mode: {item.id: float(Fraction(item.probability)) for item in row.observations}
        for row in game.post_commitment_observations
    }
    for mode in game.mode_ids:
        for history, conditional in histories.by_mode[mode].items():
            sequence = _observations_from_history(history)
            mass = float(priors[mode] * conditional)
            if include_post and post:
                for observation, probability in post[mode].items():
                    result[(*sequence, observation)] += mass * probability
            else:
                result[sequence] += mass
    return dict(result)


def _mode_sequence_distribution(
    histories: dict[str, Number], backend: Backend
) -> dict[tuple[str, ...], Number]:
    result: dict[tuple[str, ...], Number] = defaultdict(lambda: zero(backend))
    for history, probability in histories.items():
        result[_observations_from_history(history)] += probability
    return dict(result)


def _observations_from_history(history: str) -> tuple[str, ...]:
    result = []
    for token in history.split(";"):
        if token.startswith("stop@"):
            continue
        result.append(token.rsplit("/", 1)[1])
    return tuple(result)


def _entropy_nats(distribution: dict[tuple[str, ...], float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0
    )


def _prefix_distribution(
    distribution: dict[tuple[str, ...], Number],
    length: int,
    backend: Backend,
) -> dict[tuple[str, ...], Number]:
    result: dict[tuple[str, ...], Number] = defaultdict(lambda: zero(backend))
    for sequence, probability in distribution.items():
        result[sequence[:length]] += probability
    return dict(result)


def _tv(
    left: dict[tuple[str, ...], Number],
    right: dict[tuple[str, ...], Number],
    backend: Backend,
) -> Number:
    total = zero(backend)
    for key in set(left) | set(right):
        difference = left.get(key, zero(backend)) - right.get(key, zero(backend))
        total += difference if difference >= 0 else -difference
    return total / number("2", backend)


def _first_step(
    values: list[Number],
    threshold: Fraction,
    strict: bool = False,
) -> int | None:
    for index, value in enumerate(values):
        if (value > threshold) if strict else (value >= threshold):
            return index
    return None


def _conflicting_pairs(population: GeneratedPopulation) -> tuple[str, ...]:
    signatures = population.descriptor.response_signature_by_mode
    modes = population.game.mode_ids
    return tuple(
        f"{left}|{right}"
        for index, left in enumerate(modes)
        for right in modes[index + 1 :]
        if signatures[left] != signatures[right]
    )


def _mean(values: tuple[Number, ...], backend: Backend) -> Number:
    if not values:
        raise ValueError("cannot average an empty collection")
    return sum(values, start=zero(backend)) / number(str(len(values)), backend)


def _median(values: tuple[Number, ...], backend: Backend) -> Number:
    ordered = sorted(values, key=float)
    if len(ordered) % 2:
        return ordered[len(ordered) // 2]
    middle = len(ordered) // 2
    return (ordered[middle - 1] + ordered[middle]) / number("2", backend)


def _interquartile_mean(values: tuple[Number, ...], backend: Backend) -> Number:
    """Exact 25%-trimmed mean with fractional weights at finite-sample boundaries."""
    ordered = sorted(values, key=float)
    size = len(ordered)
    lower = Fraction(size, 4)
    upper = Fraction(3 * size, 4)
    weighted = zero(backend)
    mass = zero(backend)
    for index, value in enumerate(ordered):
        interval_left = Fraction(index)
        interval_right = Fraction(index + 1)
        overlap = max(Fraction(0), min(interval_right, upper) - max(interval_left, lower))
        if overlap:
            weight = number(overlap, backend)
            weighted += weight * value
            mass += weight
    return weighted / mass


def _matrix_dict(
    rows: tuple[str, ...],
    columns: tuple[str, ...],
    matrix: tuple[tuple[Number, ...], ...],
) -> dict[str, dict[str, Number]]:
    return {
        row: {column: matrix[i][j] for j, column in enumerate(columns)}
        for i, row in enumerate(rows)
    }


def float_determinant(matrix: tuple[tuple[Fraction, ...], ...]) -> tuple[float, float]:
    """Return sign and log-absolute-determinant for numerical diagnostics."""
    if not matrix:
        return 1.0, 0.0
    sign, logabs = np.linalg.slogdet(np.asarray(matrix, dtype=np.float64))
    return float(sign), float(logabs)
