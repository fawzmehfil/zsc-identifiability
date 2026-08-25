"""Exact deterministic and convexified cost-risk frontiers."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from zsc_identifiability.belief import branch_dynamics, decision_risk, initial_belief
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number, close, less, number, zero
from zsc_identifiability.policy import PolicyBranch, PolicyNode
from zsc_identifiability.results import (
    FrontierPoint,
    FrontierResult,
    RemovedFrontierPoint,
)


@dataclass(frozen=True)
class _Candidate:
    cost: Number
    risk: Number
    commitment_time: Number
    policy: PolicyNode


def _dominates(left: _Candidate, right: _Candidate) -> bool:
    no_worse = (less(left.cost, right.cost) or close(left.cost, right.cost)) and (
        less(left.risk, right.risk) or close(left.risk, right.risk)
    )
    strictly = less(left.cost, right.cost) or less(left.risk, right.risk)
    return no_worse and strictly


def _same_point(left: _Candidate, right: _Candidate) -> bool:
    return close(left.cost, right.cost) and close(left.risk, right.risk)


def _prefer_duplicate(left: _Candidate, right: _Candidate) -> bool:
    if less(left.commitment_time, right.commitment_time):
        return True
    if less(right.commitment_time, left.commitment_time):
        return False
    return left.policy.signature() < right.policy.signature()


def _prune(candidates: list[_Candidate]) -> tuple[list[_Candidate], list[RemovedFrontierPoint]]:
    selected: list[_Candidate] = []
    removed: list[RemovedFrontierPoint] = []
    for candidate in sorted(candidates, key=lambda item: (float(item.cost), float(item.risk))):
        duplicate = next((item for item in selected if _same_point(item, candidate)), None)
        if duplicate is not None:
            if _prefer_duplicate(candidate, duplicate):
                selected.remove(duplicate)
                removed.append(
                    RemovedFrontierPoint(
                        duplicate.cost,
                        duplicate.risk,
                        "duplicate coordinates; deterministic tie-break lost",
                        duplicate.policy.signature(),
                    )
                )
                selected.append(candidate)
            else:
                removed.append(
                    RemovedFrontierPoint(
                        candidate.cost,
                        candidate.risk,
                        "duplicate coordinates; deterministic tie-break lost",
                        candidate.policy.signature(),
                    )
                )
            continue
        dominator = next((item for item in selected if _dominates(item, candidate)), None)
        if dominator is not None:
            removed.append(
                RemovedFrontierPoint(
                    candidate.cost,
                    candidate.risk,
                    f"dominated by {dominator.policy.signature()}",
                    candidate.policy.signature(),
                )
            )
            continue
        newly_dominated = [item for item in selected if _dominates(candidate, item)]
        for item in newly_dominated:
            selected.remove(item)
            removed.append(
                RemovedFrontierPoint(
                    item.cost,
                    item.risk,
                    f"dominated by {candidate.policy.signature()}",
                    item.policy.signature(),
                )
            )
        selected.append(candidate)
    selected.sort(key=lambda item: (float(item.cost), float(item.risk)))
    return selected, removed


def _convex_vertices(points: list[FrontierPoint]) -> tuple[FrontierPoint, ...]:
    hull: list[FrontierPoint] = []
    for point in sorted(
        points, key=lambda item: (float(item.expected_cost), float(item.residual_risk))
    ):
        while len(hull) >= 2:
            first, second = hull[-2], hull[-1]
            left_slope = (second.residual_risk - first.residual_risk) / (
                second.expected_cost - first.expected_cost
            )
            right_slope = (point.residual_risk - second.residual_risk) / (
                point.expected_cost - second.expected_cost
            )
            if less(left_slope, right_slope):
                break
            hull.pop()
        hull.append(point)
    return tuple(hull)


def compute(
    game: FiniteConventionGame,
    action_class: str,
    backend: Backend,
    policy_node_cap: int = 200_000,
) -> FrontierResult:
    memo: dict[tuple[int, str, tuple[Number, ...]], tuple[_Candidate, ...]] = {}
    generated = 0
    removed_root: list[RemovedFrontierPoint] = []
    root_belief = initial_belief(game, backend)

    def recurse(time: int, state: str, belief: tuple[Number, ...]) -> tuple[_Candidate, ...]:
        nonlocal generated
        key = (time, state, belief)
        if key in memo:
            return memo[key]
        candidates: list[_Candidate] = []
        for decision in sorted(game.decisions):
            candidates.append(
                _Candidate(
                    cost=zero(backend),
                    risk=decision_risk(game, belief, decision, backend),
                    commitment_time=number(str(time), backend),
                    policy=PolicyNode(
                        kind="commit", time=time, state=state, belief=belief, decision=decision
                    ),
                )
            )
        if time < game.horizon:
            passive_only = action_class == "passive"
            for action in game.available_actions(state, time, passive_only=passive_only):
                branches = branch_dynamics(game, time, state, action, belief, backend)
                child_frontiers = [
                    recurse(time + 1, branch.next_state, branch.posterior) for branch in branches
                ]
                for children in itertools.product(*child_frontiers):
                    generated += 1
                    if generated > policy_node_cap:
                        raise RuntimeError(
                            f"frontier policy-node cap {policy_node_cap} exceeded at "
                            f"time={time}, state={state}, action={action}"
                        )
                    cost = zero(backend)
                    risk = zero(backend)
                    commitment_time = zero(backend)
                    policy_branches: list[PolicyBranch] = []
                    for branch, child in zip(branches, children, strict=True):
                        cost += branch.probability * (branch.expected_immediate_cost + child.cost)
                        risk += branch.probability * child.risk
                        commitment_time += branch.probability * child.commitment_time
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
                        _Candidate(
                            cost,
                            risk,
                            commitment_time,
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
        pruned, removed = _prune(candidates)
        if time == 0 and state == game.initial_state and belief == root_belief:
            removed_root.extend(removed)
        memo[key] = tuple(pruned)
        return memo[key]

    root_candidates = list(recurse(0, game.initial_state, root_belief))
    deterministic, removed = _prune(root_candidates)
    removed_root.extend(removed)
    _, prior_risk = min(
        (
            (decision, decision_risk(game, initial_belief(game, backend), decision, backend))
            for decision in game.decisions
        ),
        key=lambda item: (float(item[1]), item[0]),
    )
    points = [
        FrontierPoint(
            item.cost,
            item.risk,
            (prior_risk - item.risk) / prior_risk if prior_risk > 0 else None,
            item.commitment_time,
            item.policy,
        )
        for item in deterministic
    ]
    return FrontierResult(
        game_id=game.game_id,
        action_class=action_class,
        backend=backend,
        deterministic_points=tuple(points),
        removed_points=tuple(removed_root),
        convexified_envelope=_convex_vertices(points),
    )
