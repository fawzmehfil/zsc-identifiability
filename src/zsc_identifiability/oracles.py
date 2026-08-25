"""Explicit diagnostic control policies and oracle scalar baselines."""

from __future__ import annotations

from zsc_identifiability.belief import best_decision, branch_dynamics, initial_belief
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number, zero
from zsc_identifiability.policy import PolicyBranch, PolicyNode


def immediate_commitment_policy(
    game: FiniteConventionGame, backend: Backend = "fraction"
) -> PolicyNode:
    belief = initial_belief(game, backend)
    decision, _ = best_decision(game, belief, backend)
    return PolicyNode(
        kind="commit",
        time=0,
        state=game.initial_state,
        belief=belief,
        decision=decision,
    )


def action_then_commit_policy(
    game: FiniteConventionGame, action: str, backend: Backend = "fraction"
) -> PolicyNode:
    """Force one declared task action, then make the Bayes-optimal commitment."""
    belief = initial_belief(game, backend)
    branches = branch_dynamics(game, 0, game.initial_state, action, belief, backend)
    policy_branches = []
    for branch in branches:
        decision, _ = best_decision(game, branch.posterior, backend)
        policy_branches.append(
            PolicyBranch(
                next_state=branch.next_state,
                observation=branch.observation,
                probability=branch.probability,
                expected_immediate_cost=branch.expected_immediate_cost,
                child=PolicyNode(
                    kind="commit",
                    time=1,
                    state=branch.next_state,
                    belief=branch.posterior,
                    decision=decision,
                ),
            )
        )
    return PolicyNode(
        kind="act",
        time=0,
        state=game.initial_state,
        belief=belief,
        action=action,
        branches=tuple(policy_branches),
    )


def known_mode_risk(game: FiniteConventionGame, backend: Backend = "fraction") -> Number:
    """Bayes risk when the exact partner mode is supplied before commitment."""
    result = zero(backend)
    for mode in game.mode_ids:
        result += min(game.loss_exact(mode, decision) for decision in game.decisions)
    return result


def known_response_signature_risk(
    game: FiniteConventionGame, backend: Backend = "fraction"
) -> Number:
    """Risk when the zero-loss response set, but not identity, is supplied."""
    return known_mode_risk(game, backend)
