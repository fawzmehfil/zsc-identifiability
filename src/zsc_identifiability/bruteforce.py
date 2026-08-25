"""Independent exhaustive conditional-plan oracle for tiny verification games."""

from __future__ import annotations

import itertools

from zsc_identifiability.belief import branch_dynamics, decision_risk, initial_belief
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number, number, zero
from zsc_identifiability.policy import PolicyBranch, PolicyNode
from zsc_identifiability.solver import Candidate


def enumerate_plans(
    game: FiniteConventionGame,
    action_class: str,
    backend: Backend = "fraction",
    plan_cap: int = 500_000,
) -> tuple[Candidate, ...]:
    generated = 0

    def recurse(time: int, state: str, belief: tuple[Number, ...]) -> list[Candidate]:
        nonlocal generated
        candidates = [
            Candidate(
                zero(backend),
                decision_risk(game, belief, decision, backend),
                number(str(time), backend),
                PolicyNode(kind="commit", time=time, state=state, belief=belief, decision=decision),
            )
            for decision in sorted(game.decisions)
        ]
        if time == game.horizon:
            return candidates
        for action in game.available_actions(state, time, passive_only=action_class == "passive"):
            branches = branch_dynamics(game, time, state, action, belief, backend)
            options = [
                recurse(time + 1, branch.next_state, branch.posterior) for branch in branches
            ]
            for children in itertools.product(*options):
                generated += 1
                if generated > plan_cap:
                    raise RuntimeError(f"brute-force plan cap {plan_cap} exceeded")
                cost = zero(backend)
                risk = zero(backend)
                commit_time = zero(backend)
                policy_branches = []
                for branch, child in zip(branches, children, strict=True):
                    cost += branch.probability * (branch.expected_immediate_cost + child.cost)
                    risk += branch.probability * child.risk
                    commit_time += branch.probability * child.commitment_time
                    policy_branches.append(
                        PolicyBranch(
                            branch.next_state,
                            branch.observation,
                            branch.probability,
                            branch.expected_immediate_cost,
                            child.policy,
                        )
                    )
                candidates.append(
                    Candidate(
                        cost,
                        risk,
                        commit_time,
                        PolicyNode(
                            kind="act",
                            time=time,
                            state=state,
                            belief=belief,
                            action=action,
                            branches=tuple(policy_branches),
                        ),
                    )
                )
        return candidates

    return tuple(recurse(0, game.initial_state, initial_belief(game, backend)))


def brute_force_min_net(game: FiniteConventionGame, action_class: str) -> Candidate:
    candidates = enumerate_plans(game, action_class, "fraction")
    return min(
        candidates,
        key=lambda candidate: (
            candidate.total,
            candidate.cost,
            candidate.commitment_time,
            candidate.policy.signature(),
        ),
    )
