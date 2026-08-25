"""Belief updates, Bayes risks, and response relations."""

from __future__ import annotations

from dataclasses import dataclass

from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number, number, zero


@dataclass(frozen=True)
class BranchDynamics:
    next_state: str
    observation: str
    probability: Number
    expected_immediate_cost: Number
    posterior: tuple[Number, ...]


def initial_belief(game: FiniteConventionGame, backend: Backend) -> tuple[Number, ...]:
    return tuple(number(value, backend) for value in game.prior_exact())


def loss(game: FiniteConventionGame, mode_index: int, decision: str, backend: Backend) -> Number:
    return number(game.loss_exact(game.mode_ids[mode_index], decision), backend)


def decision_risk(
    game: FiniteConventionGame,
    belief: tuple[Number, ...],
    decision: str,
    backend: Backend,
) -> Number:
    result = zero(backend)
    for index, probability in enumerate(belief):
        result += probability * loss(game, index, decision, backend)
    return result


def best_decision(
    game: FiniteConventionGame,
    belief: tuple[Number, ...],
    backend: Backend,
) -> tuple[str, Number]:
    from zsc_identifiability.numeric import close, less

    selected = sorted(game.decisions)[0]
    selected_risk = decision_risk(game, belief, selected, backend)
    for decision in sorted(game.decisions)[1:]:
        candidate = decision_risk(game, belief, decision, backend)
        if less(candidate, selected_risk) or (
            close(candidate, selected_risk) and decision < selected
        ):
            selected, selected_risk = decision, candidate
    return selected, selected_risk


def branch_dynamics(
    game: FiniteConventionGame,
    time: int,
    state: str,
    action: str,
    belief: tuple[Number, ...],
    backend: Backend,
) -> tuple[BranchDynamics, ...]:
    branch_keys: set[tuple[str, str]] = set()
    for mode in game.mode_ids:
        row = game.kernel(time, state, action, mode)
        branch_keys.update((outcome.next_state, outcome.observation) for outcome in row.outcomes)

    result: list[BranchDynamics] = []
    for next_state, observation in sorted(branch_keys):
        likelihoods: list[Number] = []
        cost_numerator = zero(backend)
        probability = zero(backend)
        for mode_index, mode in enumerate(game.mode_ids):
            mode_likelihood = zero(backend)
            mode_cost_weight = zero(backend)
            for outcome in game.kernel(time, state, action, mode).outcomes:
                if (outcome.next_state, outcome.observation) != (next_state, observation):
                    continue
                outcome_probability = number(outcome.probability, backend)
                mode_likelihood += outcome_probability
                mode_cost_weight += outcome_probability * number(outcome.cost, backend)
            likelihoods.append(mode_likelihood)
            probability += belief[mode_index] * mode_likelihood
            cost_numerator += belief[mode_index] * mode_cost_weight
        if probability == 0:
            continue
        posterior = tuple(
            belief[index] * likelihoods[index] / probability for index in range(len(belief))
        )
        result.append(
            BranchDynamics(
                next_state=next_state,
                observation=observation,
                probability=probability,
                expected_immediate_cost=cost_numerator / probability,
                posterior=posterior,
            )
        )
    return tuple(result)


def zero_loss_sets(game: FiniteConventionGame) -> dict[str, frozenset[str]]:
    return {
        mode: frozenset(
            decision for decision in game.decisions if game.loss_exact(mode, decision) == 0
        )
        for mode in game.mode_ids
    }


def response_equivalent(game: FiniteConventionGame, left: str, right: str) -> bool:
    sets = zero_loss_sets(game)
    return sets[left] == sets[right]


def response_compatible(game: FiniteConventionGame, left: str, right: str) -> bool:
    sets = zero_loss_sets(game)
    return bool(sets[left] & sets[right])


def conflict_coefficient(
    game: FiniteConventionGame, left: str, right: str, backend: Backend = "fraction"
) -> Number:
    values = [
        number(game.loss_exact(left, decision) + game.loss_exact(right, decision), backend)
        for decision in game.decisions
    ]
    return min(values)
