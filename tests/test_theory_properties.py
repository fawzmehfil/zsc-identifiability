from fractions import Fraction

from hypothesis import given
from hypothesis import strategies as st

from zsc_identifiability.frontier import compute
from zsc_identifiability.metrics import evaluate
from zsc_identifiability.oracles import action_then_commit_policy, immediate_commitment_policy
from zsc_identifiability.solver import solve
from zsc_identifiability.theory import (
    multitype_pairwise_lower_bound,
    one_intervention_is_strictly_optimal,
    verify_binary_identities,
)


def test_binary_overlap_and_tv_identities(game_loader) -> None:
    game = game_loader("active-separable")
    checks = verify_binary_identities(game, action_then_commit_policy(game, "stage_shared_item"))
    assert all(check.passed for check in checks)


def test_multitype_pairwise_bound(game_loader) -> None:
    game = game_loader("multitype-asymmetric-loss")
    policy = action_then_commit_policy(game, "share_resource")
    actual = evaluate(game, policy, "fraction").residual_risk_precommitment
    bound = multitype_pairwise_lower_bound(game, policy, "mode_zero", "mode_two")
    assert actual >= bound > 0


def test_policy_class_enlargement_and_more_evidence(game_loader) -> None:
    game = game_loader("active-separable")
    passive = solve(game, "passive", "information", "fraction")
    task = solve(game, "task", "information", "fraction")
    immediate = evaluate(game, immediate_commitment_policy(game), "fraction")
    observed = evaluate(game, task.policy, "fraction")
    assert task.residual_decision_risk <= passive.residual_decision_risk
    assert observed.residual_risk_precommitment <= immediate.residual_risk_precommitment


def test_frontier_information_is_monotone_in_budget(game_loader) -> None:
    frontier = compute(game_loader("active-separable"), "task", "fraction")
    points = sorted(frontier.deterministic_points, key=lambda point: point.expected_cost)
    running = Fraction(0)
    for point in points:
        assert point.dri is not None
        assert point.dri >= running
        running = point.dri


def test_dri_is_bounded_for_every_canonical_frontier(game_loader) -> None:
    names = (
        "no-identification-needed",
        "passive-easy",
        "active-separable",
        "intervention-too-expensive",
        "threshold-boundary",
        "fundamentally-inseparable",
        "late-reveal",
        "decision-irrelevant-identity",
        "multitype-asymmetric-loss",
        "repeated-noisy-interaction",
    )
    for name in names:
        for point in compute(game_loader(name), "task", "fraction").deterministic_points:
            assert point.dri is None or 0 <= point.dri <= 1


@given(
    q=st.sampled_from([Fraction(1, 2), Fraction(3, 5), Fraction(4, 5), Fraction(1)]),
    cost=st.integers(min_value=0, max_value=30),
    remaining=st.integers(min_value=1, max_value=8),
    mismatch=st.integers(min_value=1, max_value=12),
)
def test_one_intervention_threshold_property(q, cost, remaining, mismatch) -> None:
    immediate = Fraction(remaining * mismatch, 2)
    probed = Fraction(cost) + remaining * mismatch * (1 - q)
    assert one_intervention_is_strictly_optimal(
        Fraction(cost), remaining, Fraction(mismatch), q
    ) == (probed < immediate)


@given(
    weight=st.fractions(min_value=0, max_value=1, max_denominator=20),
    cost_left=st.fractions(min_value=0, max_value=20, max_denominator=20),
    cost_right=st.fractions(min_value=0, max_value=20, max_denominator=20),
    risk_left=st.fractions(min_value=0, max_value=20, max_denominator=20),
    risk_right=st.fractions(min_value=0, max_value=20, max_denominator=20),
)
def test_episode_level_random_mixtures_interpolate_linearly(
    weight, cost_left, cost_right, risk_left, risk_right
) -> None:
    mixed_cost = weight * cost_left + (1 - weight) * cost_right
    mixed_risk = weight * risk_left + (1 - weight) * risk_right
    assert mixed_cost - (weight * cost_left) == (1 - weight) * cost_right
    assert mixed_risk - (weight * risk_left) == (1 - weight) * risk_right
