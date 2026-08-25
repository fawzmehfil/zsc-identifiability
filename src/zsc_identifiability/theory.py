"""Executable checks for the Phase 2 theoretical package."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

from zsc_identifiability.belief import conflict_coefficient, zero_loss_sets
from zsc_identifiability.metrics import evaluate
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Number
from zsc_identifiability.policy import PolicyNode


@dataclass(frozen=True)
class TheoryCheck:
    name: str
    passed: bool
    left: str
    right: str
    explanation: str


def weighted_overlap(game: FiniteConventionGame, policy: PolicyNode) -> Fraction:
    """Binary weighted-overlap expression under unique opposing decisions."""
    from zsc_identifiability.metrics import compute_distributions

    if len(game.mode_ids) != 2:
        raise ValueError("weighted overlap requires exactly two modes")
    sets = zero_loss_sets(game)
    if any(len(sets[mode]) != 1 for mode in game.mode_ids):
        raise ValueError("weighted overlap checker requires one optimal decision per mode")
    left, right = game.mode_ids
    left_decision = next(iter(sets[left]))
    right_decision = next(iter(sets[right]))
    if left_decision == right_decision:
        raise ValueError("binary modes must require different decisions")
    mismatch_left = game.loss_exact(left, right_decision)
    mismatch_right = game.loss_exact(right, left_decision)
    priors = dict(zip(game.mode_ids, game.prior_exact(), strict=True))
    histories = compute_distributions(game, policy, "fraction")
    support = set(histories.by_mode[left]) | set(histories.by_mode[right])
    result = sum(
        (
            min(
                priors[left]
                * mismatch_left
                * cast(Fraction, histories.by_mode[left].get(history, Fraction(0))),
                priors[right]
                * mismatch_right
                * cast(Fraction, histories.by_mode[right].get(history, Fraction(0))),
            )
            for history in support
        ),
        start=Fraction(0),
    )
    return result


def binary_tv_risk(game: FiniteConventionGame, policy: PolicyNode) -> Fraction:
    if game.prior_exact() != (Fraction(1, 2), Fraction(1, 2)):
        raise ValueError("TV equality checker requires equal binary priors")
    sets = zero_loss_sets(game)
    left, right = game.mode_ids
    if len(sets[left]) != 1 or len(sets[right]) != 1:
        raise ValueError("TV equality checker requires unique optimal decisions")
    left_decision = next(iter(sets[left]))
    right_decision = next(iter(sets[right]))
    mismatch_left = game.loss_exact(left, right_decision)
    mismatch_right = game.loss_exact(right, left_decision)
    if mismatch_left != mismatch_right:
        raise ValueError("TV equality checker requires symmetric mismatch loss")
    evaluation = evaluate(game, policy, "fraction")
    tv = evaluation.pairwise_total_variation[f"{left}|{right}"]
    if not isinstance(tv, Fraction):
        raise TypeError("fraction evaluation returned a float")
    return mismatch_left * (Fraction(1) - tv) / 2


def multitype_pairwise_lower_bound(
    game: FiniteConventionGame, policy: PolicyNode, left: str, right: str
) -> Fraction:
    from zsc_identifiability.metrics import compute_distributions

    histories = compute_distributions(game, policy, "fraction")
    priors = dict(zip(game.mode_ids, game.prior_exact(), strict=True))
    coefficient = conflict_coefficient(game, left, right, "fraction")
    if not isinstance(coefficient, Fraction):
        raise TypeError("fraction conflict coefficient returned a float")
    support = set(histories.by_mode[left]) | set(histories.by_mode[right])
    overlap = sum(
        (
            min(
                priors[left] * cast(Fraction, histories.by_mode[left].get(history, Fraction(0))),
                priors[right] * cast(Fraction, histories.by_mode[right].get(history, Fraction(0))),
            )
            for history in support
        ),
        start=Fraction(0),
    )
    return coefficient * overlap


def one_intervention_is_strictly_optimal(
    cost: Fraction, remaining_decisions: int, mismatch_loss: Fraction, accuracy: Fraction
) -> bool:
    return cost < remaining_decisions * mismatch_loss * (accuracy - Fraction(1, 2))


def verify_binary_identities(
    game: FiniteConventionGame, policy: PolicyNode
) -> tuple[TheoryCheck, ...]:
    evaluation = evaluate(game, policy, "fraction")
    actual = evaluation.residual_risk_precommitment
    if not isinstance(actual, Fraction):
        raise TypeError("fraction evaluation returned a float")
    overlap = weighted_overlap(game, policy)
    tv_value = binary_tv_risk(game, policy)
    return (
        TheoryCheck(
            "binary_weighted_overlap_identity",
            actual == overlap,
            str(actual),
            str(overlap),
            "Bayes residual risk equals the weighted overlap of mode-conditioned histories.",
        ),
        TheoryCheck(
            "binary_total_variation_corollary",
            actual == tv_value,
            str(actual),
            str(tv_value),
            "Equal-prior symmetric binary risk equals M/2 times one minus total variation.",
        ),
    )


def check_non_decreasing(values: tuple[Number, ...]) -> bool:
    return all(
        float(left) <= float(right) + 1e-12 for left, right in zip(values, values[1:], strict=False)
    )
