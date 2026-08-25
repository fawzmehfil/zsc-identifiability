import json
from fractions import Fraction

import pytest

from zsc_identifiability.bruteforce import brute_force_min_net
from zsc_identifiability.frontier import compute
from zsc_identifiability.metrics import evaluate
from zsc_identifiability.policy import policy_from_dict
from zsc_identifiability.solver import solve


@pytest.mark.parametrize(
    "name",
    [
        "no-identification-needed",
        "passive-easy",
        "active-separable",
        "intervention-too-expensive",
        "threshold-boundary",
        "fundamentally-inseparable",
        "late-reveal",
        "multitype-asymmetric-loss",
    ],
)
def test_bellman_matches_independent_brute_force(game_loader, name: str) -> None:
    game = game_loader(name)
    dynamic = solve(game, "task", "net_regret", "fraction")
    exhaustive = brute_force_min_net(game, "task")
    assert dynamic.total_cost_plus_risk == exhaustive.total
    assert dynamic.expected_intervention_cost == exhaustive.cost
    assert dynamic.expected_commitment_time == exhaustive.commitment_time


@pytest.mark.parametrize(
    "name",
    [
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
    ],
)
def test_fraction_and_float_backends_agree(game_loader, name: str) -> None:
    game = game_loader(name)
    exact = solve(game, "task", "net_regret", "fraction")
    approximate = solve(game, "task", "net_regret", "float")
    assert float(exact.expected_intervention_cost) == pytest.approx(
        approximate.expected_intervention_cost, abs=1e-10
    )
    assert float(exact.residual_decision_risk) == pytest.approx(
        approximate.residual_decision_risk, abs=1e-10
    )
    assert exact.policy.signature() == approximate.policy.signature()


def test_serialized_policy_is_independently_re_evaluated(game_loader) -> None:
    game = game_loader("active-separable")
    original = solve(game, "task", "net_regret", "fraction").policy
    restored = policy_from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    assert evaluate(game, restored, "fraction").net_oracle_regret == 13


def test_frontier_keeps_pure_points_and_convex_vertices(game_loader) -> None:
    game = game_loader("active-separable")
    frontier = compute(game, "task", "fraction")
    coordinates = [
        (point.expected_cost, point.residual_risk) for point in frontier.deterministic_points
    ]
    assert coordinates == [(Fraction(0), Fraction(20)), (Fraction(5), Fraction(8))]
    assert frontier.convexified_envelope == frontier.deterministic_points
    assert frontier.removed_points
    for removed in frontier.removed_points:
        assert any(
            point.expected_cost <= removed.expected_cost
            and point.residual_risk <= removed.residual_risk
            for point in frontier.deterministic_points
        )


def test_frontier_policy_node_cap_fails_loudly(game_loader) -> None:
    with pytest.raises(RuntimeError, match="policy-node cap"):
        compute(game_loader("repeated-noisy-interaction"), "task", "fraction", policy_node_cap=1)
