"""Belief-state dynamic programming for task and information oracles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from zsc_identifiability.belief import best_decision, branch_dynamics, initial_belief
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number, close, less, number, zero
from zsc_identifiability.policy import PolicyBranch, PolicyNode
from zsc_identifiability.results import PolicySolution

ActionClass = Literal["passive", "task", "reconnaissance"]
Objective = Literal["net_regret", "information"]


@dataclass(frozen=True)
class Candidate:
    cost: Number
    risk: Number
    commitment_time: Number
    policy: PolicyNode

    @property
    def total(self) -> Number:
        return self.cost + self.risk


def _candidate_better(left: Candidate, right: Candidate, objective: Objective) -> bool:
    if objective == "information":
        pairs = ((left.risk, right.risk), (left.cost, right.cost))
    else:
        pairs = ((left.total, right.total), (left.cost, right.cost))
    for left_value, right_value in pairs:
        if less(left_value, right_value):
            return True
        if less(right_value, left_value):
            return False
    if less(left.commitment_time, right.commitment_time):
        return True
    if less(right.commitment_time, left.commitment_time):
        return False
    return left.policy.signature() < right.policy.signature()


def solve(
    game: FiniteConventionGame,
    action_class: ActionClass,
    objective: Objective,
    backend: Backend,
) -> PolicySolution:
    memo: dict[tuple[int, str, tuple[Number, ...]], Candidate] = {}

    def recurse(time: int, state: str, belief: tuple[Number, ...]) -> Candidate:
        key = (time, state, belief)
        if key in memo:
            return memo[key]
        decision, risk = best_decision(game, belief, backend)
        best = Candidate(
            cost=zero(backend),
            risk=risk,
            commitment_time=number(str(time), backend),
            policy=PolicyNode(
                kind="commit", time=time, state=state, belief=belief, decision=decision
            ),
        )
        if time < game.horizon:
            passive_only = action_class == "passive"
            for action in game.available_actions(state, time, passive_only=passive_only):
                branches = branch_dynamics(game, time, state, action, belief, backend)
                if not branches:
                    continue
                expected_cost = zero(backend)
                expected_risk = zero(backend)
                expected_time = zero(backend)
                policy_branches: list[PolicyBranch] = []
                for branch in branches:
                    child = recurse(time + 1, branch.next_state, branch.posterior)
                    expected_cost += branch.probability * (
                        branch.expected_immediate_cost + child.cost
                    )
                    expected_risk += branch.probability * child.risk
                    expected_time += branch.probability * child.commitment_time
                    policy_branches.append(
                        PolicyBranch(
                            next_state=branch.next_state,
                            observation=branch.observation,
                            probability=branch.probability,
                            expected_immediate_cost=branch.expected_immediate_cost,
                            child=child.policy,
                        )
                    )
                candidate = Candidate(
                    cost=expected_cost,
                    risk=expected_risk,
                    commitment_time=expected_time,
                    policy=PolicyNode(
                        kind="act",
                        time=time,
                        state=state,
                        belief=belief,
                        action=action,
                        branches=tuple(policy_branches),
                    ),
                )
                if _candidate_better(candidate, best, objective):
                    best = candidate
        memo[key] = best
        return best

    root = recurse(0, game.initial_state, initial_belief(game, backend))
    return PolicySolution(
        game_id=game.game_id,
        action_class=action_class,
        objective=objective,
        backend=backend,
        policy=root.policy,
        expected_intervention_cost=root.cost,
        residual_decision_risk=root.risk,
        total_cost_plus_risk=root.total,
        expected_commitment_time=root.commitment_time,
        tie_breaking_record=(
            "primary objective",
            "lowest intervention cost",
            "earlier commitment",
            "lexicographic policy signature",
        ),
    )


def assert_solution_consistent(solution: PolicySolution) -> None:
    if not close(
        solution.total_cost_plus_risk,
        solution.expected_intervention_cost + solution.residual_decision_risk,
    ):
        raise AssertionError("solution total is inconsistent")
