"""Policy-induced histories and decision-relevant identifiability metrics."""

from __future__ import annotations

import math
from collections import defaultdict

from zsc_identifiability.belief import best_decision, initial_belief, zero_loss_sets
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number, number, one, zero
from zsc_identifiability.policy import PolicyNode
from zsc_identifiability.results import HistoryDistributions, PolicyEvaluation


def _child(node: PolicyNode, next_state: str, observation: str) -> PolicyNode:
    for branch in node.branches:
        if (branch.next_state, branch.observation) == (next_state, observation):
            return branch.child
    raise ValueError(f"policy has no branch for {(next_state, observation)}")


def compute_distributions(
    game: FiniteConventionGame, policy: PolicyNode, backend: Backend
) -> HistoryDistributions:
    by_mode: dict[str, dict[str, Number]] = {}
    expected_cost_by_mode: dict[str, Number] = {}
    decisions: dict[str, str] = {}
    commitment_times: dict[str, int] = {}

    def recurse(
        mode: str, node: PolicyNode, path: tuple[str, ...]
    ) -> tuple[dict[str, Number], Number]:
        if node.kind == "commit":
            history = ";".join((*path, f"stop@{node.time}"))
            if node.decision is None:
                raise ValueError("commit node lacks a decision")
            prior_decision = decisions.setdefault(history, node.decision)
            if prior_decision != node.decision:
                raise ValueError("one evidence history maps to inconsistent decisions")
            commitment_times[history] = node.time
            return {history: one(backend)}, zero(backend)
        if node.action is None:
            raise ValueError("action node lacks an action")
        row = game.kernel(node.time, node.state, node.action, mode)
        distributions: dict[str, Number] = defaultdict(lambda: zero(backend))
        expected_cost = zero(backend)
        grouped: dict[tuple[str, str], tuple[Number, Number]] = {}
        for outcome in row.outcomes:
            key = (outcome.next_state, outcome.observation)
            probability = number(outcome.probability, backend)
            cost_weight = probability * number(outcome.cost, backend)
            old_probability, old_cost = grouped.get(key, (zero(backend), zero(backend)))
            grouped[key] = (old_probability + probability, old_cost + cost_weight)
        for (next_state, observation), (probability, cost_weight) in grouped.items():
            child = _child(node, next_state, observation)
            child_path = (*path, f"{node.time}:{node.action}>{next_state}/{observation}")
            child_distribution, child_cost = recurse(mode, child, child_path)
            for history, child_probability in child_distribution.items():
                distributions[history] += probability * child_probability
            expected_cost += cost_weight + probability * child_cost
        return dict(distributions), expected_cost

    for mode in game.mode_ids:
        distribution, cost = recurse(mode, policy, ())
        by_mode[mode] = distribution
        expected_cost_by_mode[mode] = cost

    priors = dict(zip(game.mode_ids, initial_belief(game, backend), strict=True))
    all_histories = sorted({history for values in by_mode.values() for history in values})
    posteriors: dict[str, dict[str, Number]] = {}
    for history in all_histories:
        mass = sum(
            (priors[mode] * by_mode[mode].get(history, zero(backend)) for mode in game.mode_ids),
            start=zero(backend),
        )
        if mass == 0:
            continue
        posteriors[history] = {
            mode: priors[mode] * by_mode[mode].get(history, zero(backend)) / mass
            for mode in game.mode_ids
        }
    return HistoryDistributions(
        game_id=game.game_id,
        by_mode=by_mode,
        expected_cost_by_mode=expected_cost_by_mode,
        decisions_by_history=decisions,
        commitment_time_by_history=commitment_times,
        posterior_by_history=posteriors,
    )


def _mutual_information(
    game: FiniteConventionGame,
    histories: HistoryDistributions,
    class_by_mode: dict[str, str],
    backend: Backend,
) -> float:
    priors = dict(zip(game.mode_ids, initial_belief(game, backend), strict=True))
    class_prior: dict[str, float] = defaultdict(float)
    for mode, prior in priors.items():
        class_prior[class_by_mode[mode]] += float(prior)
    history_mass: dict[str, float] = defaultdict(float)
    joint: dict[tuple[str, str], float] = defaultdict(float)
    for mode in game.mode_ids:
        label = class_by_mode[mode]
        for history, conditional in histories.by_mode[mode].items():
            value = float(priors[mode] * conditional)
            joint[(label, history)] += value
            history_mass[history] += value
    result = 0.0
    for (label, history), value in joint.items():
        if value > 0:
            result += value * math.log2(value / (class_prior[label] * history_mass[history]))
    return result


def _residual_risk(
    game: FiniteConventionGame,
    histories: HistoryDistributions,
    backend: Backend,
) -> Number:
    priors = dict(zip(game.mode_ids, initial_belief(game, backend), strict=True))
    result = zero(backend)
    for history, posterior_map in histories.posterior_by_history.items():
        history_mass = sum(
            (
                priors[mode] * histories.by_mode[mode].get(history, zero(backend))
                for mode in game.mode_ids
            ),
            start=zero(backend),
        )
        posterior = tuple(posterior_map[mode] for mode in game.mode_ids)
        _, risk = best_decision(game, posterior, backend)
        result += history_mass * risk
    return result


def _eventual_risk(
    game: FiniteConventionGame,
    histories: HistoryDistributions,
    backend: Backend,
) -> Number:
    if not game.post_commitment_observations:
        return _residual_risk(game, histories, backend)
    post = {
        row.mode: {item.id: number(item.probability, backend) for item in row.observations}
        for row in game.post_commitment_observations
    }
    priors = dict(zip(game.mode_ids, initial_belief(game, backend), strict=True))
    result = zero(backend)
    histories_set = sorted({history for values in histories.by_mode.values() for history in values})
    post_observations = sorted({obs for values in post.values() for obs in values})
    for history in histories_set:
        for observation in post_observations:
            decision_masses: list[Number] = []
            for decision in game.decisions:
                decision_loss = zero(backend)
                for mode in game.mode_ids:
                    joint = (
                        priors[mode]
                        * histories.by_mode[mode].get(history, zero(backend))
                        * post[mode].get(observation, zero(backend))
                    )
                    decision_loss += joint * number(game.loss_exact(mode, decision), backend)
                decision_masses.append(decision_loss)
            result += min(decision_masses)
    return result


def evaluate(game: FiniteConventionGame, policy: PolicyNode, backend: Backend) -> PolicyEvaluation:
    histories = compute_distributions(game, policy, backend)
    prior = initial_belief(game, backend)
    _, prior_risk = best_decision(game, prior, backend)
    residual = _residual_risk(game, histories, backend)
    eventual = _eventual_risk(game, histories, backend)
    priors = dict(zip(game.mode_ids, prior, strict=True))
    expected_cost = sum(
        (priors[mode] * histories.expected_cost_by_mode[mode] for mode in game.mode_ids),
        start=zero(backend),
    )
    all_histories = sorted({history for values in histories.by_mode.values() for history in values})
    map_accuracy = zero(backend)
    actual_loss = zero(backend)
    expected_time = zero(backend)
    for history in all_histories:
        joints = {
            mode: priors[mode] * histories.by_mode[mode].get(history, zero(backend))
            for mode in game.mode_ids
        }
        map_accuracy += max(joints.values())
        history_mass = sum(joints.values(), start=zero(backend))
        expected_time += history_mass * number(
            str(histories.commitment_time_by_history[history]), backend
        )
        selected = histories.decisions_by_history[history]
        for mode, joint in joints.items():
            actual_loss += joint * number(game.loss_exact(mode, selected), backend)

    tv: dict[str, Number] = {}
    for left_index, left in enumerate(game.mode_ids):
        for right in game.mode_ids[left_index + 1 :]:
            support = set(histories.by_mode[left]) | set(histories.by_mode[right])
            distance_numerator = zero(backend)
            for item in support:
                left_probability = histories.by_mode[left].get(item, zero(backend))
                right_probability = histories.by_mode[right].get(item, zero(backend))
                difference = left_probability - right_probability
                distance_numerator += difference if difference >= 0 else -difference
            distance = distance_numerator / number("2", backend)
            tv[f"{left}|{right}"] = distance

    sets = zero_loss_sets(game)
    signatures = {mode: ",".join(sorted(sets[mode])) for mode in game.mode_ids}
    identity_labels = {mode: mode for mode in game.mode_ids}
    unique = all(len(values) == 1 for values in sets.values())
    decision_accuracy: Number | None = None
    if unique:
        decision_accuracy = zero(backend)
        targets = {mode: next(iter(sets[mode])) for mode in game.mode_ids}
        for history in all_histories:
            chosen = histories.decisions_by_history[history]
            for mode in game.mode_ids:
                if chosen == targets[mode]:
                    decision_accuracy += priors[mode] * histories.by_mode[mode].get(
                        history, zero(backend)
                    )

    required = prior_risk > 0
    dri = (prior_risk - residual) / prior_risk if required else None
    eventual_dri = (prior_risk - eventual) / prior_risk if required else None
    return PolicyEvaluation(
        game_id=game.game_id,
        backend=backend,
        prior_risk=prior_risk,
        residual_risk_precommitment=residual,
        residual_risk_eventual=eventual,
        expected_intervention_cost=expected_cost,
        net_oracle_regret=expected_cost + residual,
        dri_precommitment=dri,
        dri_eventual=eventual_dri,
        identification_required=required,
        decision_sufficient=(not required) or residual == 0,
        identity_mutual_information_bits=_mutual_information(
            game, histories, identity_labels, backend
        ),
        decision_signature_mutual_information_bits=_mutual_information(
            game, histories, signatures, backend
        ),
        map_type_accuracy=map_accuracy,
        decision_accuracy=decision_accuracy,
        expected_commitment_time=expected_time,
        pairwise_total_variation=tv,
        actual_policy_loss=actual_loss,
    )
