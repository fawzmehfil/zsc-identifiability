from fractions import Fraction

from zsc_identifiability.metrics import evaluate
from zsc_identifiability.oracles import action_then_commit_policy
from zsc_identifiability.solver import solve


def test_no_identification_needed(game_loader) -> None:
    game = game_loader("no-identification-needed")
    policy = action_then_commit_policy(game, "stage_shared_item")
    item = evaluate(game, policy, "fraction")
    assert item.prior_risk == 0
    assert item.dri_precommitment is None
    assert not item.identification_required
    assert item.decision_sufficient
    assert item.identity_mutual_information_bits == 1.0


def test_passive_easy_uses_free_evidence(game_loader) -> None:
    game = game_loader("passive-easy")
    passive = solve(game, "passive", "net_regret", "fraction")
    active = solve(game, "task", "net_regret", "fraction")
    assert passive.policy.action == "advance_task"
    assert active.policy.action == "advance_task"
    assert passive.residual_decision_risk == active.residual_decision_risk == 8
    assert passive.expected_intervention_cost == active.expected_intervention_cost == 0


def test_active_separable_exact_values(game_loader) -> None:
    game = game_loader("active-separable")
    solution = solve(game, "task", "net_regret", "fraction")
    item = evaluate(game, solution.policy, "fraction")
    assert solution.policy.action == "stage_shared_item"
    assert item.prior_risk == 20
    assert item.residual_risk_precommitment == 8
    assert item.dri_precommitment == Fraction(3, 5)
    assert item.expected_intervention_cost == 5
    assert item.net_oracle_regret == 13


def test_cost_and_boundary_change_only_task_choice(game_loader) -> None:
    expensive = solve(game_loader("intervention-too-expensive"), "task", "net_regret", "fraction")
    boundary = solve(game_loader("threshold-boundary"), "task", "net_regret", "fraction")
    assert expensive.policy.kind == "commit"
    assert boundary.policy.kind == "commit"
    assert expensive.total_cost_plus_risk == boundary.total_cost_plus_risk == 20


def test_inseparable_game_has_zero_tv_and_dri(game_loader) -> None:
    game = game_loader("fundamentally-inseparable")
    forced = action_then_commit_policy(game, "stage_shared_item")
    item = evaluate(game, forced, "fraction")
    assert item.residual_risk_precommitment == item.prior_risk == 20
    assert item.dri_precommitment == 0
    assert next(iter(item.pairwise_total_variation.values())) == 0


def test_late_reveal_is_excluded_before_commitment(game_loader) -> None:
    game = game_loader("late-reveal")
    item = evaluate(game, action_then_commit_policy(game, "advance_task"), "fraction")
    assert item.dri_precommitment == 0
    assert item.dri_eventual == 1
    assert item.residual_risk_precommitment == 20
    assert item.residual_risk_eventual == 0


def test_identity_information_can_be_decision_irrelevant(game_loader) -> None:
    game = game_loader("decision-irrelevant-identity")
    parity = evaluate(game, action_then_commit_policy(game, "yield_role"), "fraction")
    response = evaluate(game, action_then_commit_policy(game, "stage_shared_item"), "fraction")
    assert parity.identity_mutual_information_bits == 1.0
    assert parity.dri_precommitment == 0
    assert response.dri_precommitment == Fraction(3, 5)
    assert response.identity_mutual_information_bits < parity.identity_mutual_information_bits


def test_repeated_game_stops_after_decisive_evidence(game_loader) -> None:
    game = game_loader("repeated-noisy-interaction")
    solution = solve(game, "task", "net_regret", "fraction")
    assert solution.policy.action == "stage_shared_item"
    branches = {branch.observation: branch.child for branch in solution.policy.branches}
    assert branches["clear_left"].kind == "commit"
    assert branches["clear_right"].kind == "commit"
    assert branches["ambiguous"].kind == "act"
    assert solution.expected_commitment_time == Fraction(15, 8)
