from fractions import Fraction

from zsc_identifiability.belief import (
    branch_dynamics,
    conflict_coefficient,
    initial_belief,
    response_compatible,
    response_equivalent,
)


def test_exact_posterior_update(game_loader) -> None:
    game = game_loader("active-separable")
    branches = branch_dynamics(
        game, 0, "ready", "stage_shared_item", initial_belief(game, "fraction"), "fraction"
    )
    by_observation = {branch.observation: branch for branch in branches}
    assert by_observation["left_signal"].posterior == (Fraction(4, 5), Fraction(1, 5))
    assert by_observation["right_signal"].posterior == (Fraction(1, 5), Fraction(4, 5))
    assert sum(branch.probability for branch in branches) == 1


def test_zero_probability_branches_are_not_reachable(game_loader) -> None:
    game = game_loader("no-identification-needed")
    branches = branch_dynamics(
        game, 0, "ready", "stage_shared_item", (Fraction(1), Fraction(0)), "fraction"
    )
    assert [branch.observation for branch in branches] == ["signal_a"]


def test_equivalence_compatibility_and_conflict_are_distinct(game_loader) -> None:
    game = game_loader("multitype-asymmetric-loss")
    assert not response_equivalent(game, "mode_zero", "mode_one")
    assert response_compatible(game, "mode_zero", "mode_one")
    assert response_compatible(game, "mode_one", "mode_two")
    assert not response_compatible(game, "mode_zero", "mode_two")
    assert conflict_coefficient(game, "mode_zero", "mode_two") == 4
