"""Reference and restricted policy classes used by matched benchmark audits."""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from zsc_identifiability.belief import best_decision, branch_dynamics, initial_belief
from zsc_identifiability.metrics import evaluate
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import Backend, Number
from zsc_identifiability.policy import PolicyBranch, PolicyNode
from zsc_identifiability.results import PolicyEvaluation


def fixed_action_sequence_policy(
    game: FiniteConventionGame,
    actions: tuple[str, ...],
    backend: Backend = "fraction",
    fixed_decision: str | None = None,
) -> PolicyNode:
    """Follow a declared action sequence, then commit from the resulting belief."""
    if len(actions) > game.horizon:
        raise ValueError("reference action sequence exceeds the game horizon")

    def recurse(
        time: int,
        state: str,
        belief: tuple[Number, ...],
    ) -> PolicyNode:
        if time == len(actions):
            decision = fixed_decision or best_decision(game, belief, backend)[0]
            return PolicyNode(
                kind="commit",
                time=time,
                state=state,
                belief=belief,
                decision=decision,
            )
        action = actions[time]
        if action not in game.available_actions(state, time):
            raise ValueError(
                f"reference action {action!r} is unavailable at time={time}, state={state}"
            )
        branches = branch_dynamics(game, time, state, action, belief, backend)
        return PolicyNode(
            kind="act",
            time=time,
            state=state,
            belief=belief,
            action=action,
            branches=tuple(
                PolicyBranch(
                    next_state=branch.next_state,
                    observation=branch.observation,
                    probability=branch.probability,
                    expected_immediate_cost=branch.expected_immediate_cost,
                    child=recurse(time + 1, branch.next_state, branch.posterior),
                )
                for branch in branches
            ),
        )

    return recurse(0, game.initial_state, initial_belief(game, backend))


@dataclass(frozen=True)
class RestrictedPolicyResult:
    policy: PolicyNode
    evaluation: PolicyEvaluation


def solve_observation_restricted(
    game: FiniteConventionGame,
    commitment_states: frozenset[str],
    backend: Backend = "fraction",
    mask_observations: bool = False,
) -> RestrictedPolicyResult:
    """Enumerate deterministic policies indexed only by current observable state.

    With ``mask_observations=True``, every response token maps to the same local key,
    producing the evidence-blind policy class. Otherwise the policy may use the latest
    observation but cannot use any earlier interaction history.
    """
    keys = _reachable_local_keys(game, commitment_states, mask_observations)
    choices = {
        key: _choices_for_key(game, key, commitment_states) for key in sorted(keys)
    }
    if any(not value for value in choices.values()):
        empty = next(key for key, value in choices.items() if not value)
        raise ValueError(f"restricted policy has no feasible choice at {empty}")
    ordered_keys = tuple(sorted(choices))
    best: RestrictedPolicyResult | None = None
    best_key: tuple[float, float, float, str] | None = None
    for selected in itertools.product(*(choices[key] for key in ordered_keys)):
        mapping = dict(zip(ordered_keys, selected, strict=True))
        policy = _policy_from_local_mapping(
            game,
            mapping,
            backend,
            mask_observations,
        )
        evaluation = evaluate(game, policy, backend)
        comparison = (
            float(evaluation.expected_intervention_cost + evaluation.actual_policy_loss),
            float(evaluation.expected_intervention_cost),
            float(evaluation.expected_commitment_time),
            policy.signature(),
        )
        if best_key is None or comparison < best_key:
            best_key = comparison
            best = RestrictedPolicyResult(policy, evaluation)
    if best is None:  # pragma: no cover - nonempty choice product is validated above
        raise ValueError("restricted policy enumeration produced no policy")
    return best


LocalKey = tuple[int, str, str]
Choice = tuple[str, str]


def _reachable_local_keys(
    game: FiniteConventionGame,
    commitment_states: frozenset[str],
    mask_observations: bool,
) -> set[LocalKey]:
    start: LocalKey = (0, game.initial_state, "masked" if mask_observations else "start")
    frontier = {start}
    reached = {start}
    while frontier:
        next_frontier: set[LocalKey] = set()
        for time, state, _ in frontier:
            if time >= game.horizon:
                continue
            for action in game.available_actions(state, time):
                for mode in game.mode_ids:
                    for outcome in game.kernel(time, state, action, mode).outcomes:
                        observation = "masked" if mask_observations else outcome.observation
                        key = (time + 1, outcome.next_state, observation)
                        if key not in reached:
                            reached.add(key)
                            next_frontier.add(key)
            if state in commitment_states:
                continue
        frontier = next_frontier
    return reached


def _choices_for_key(
    game: FiniteConventionGame,
    key: LocalKey,
    commitment_states: frozenset[str],
) -> tuple[Choice, ...]:
    time, state, _ = key
    result: list[Choice] = []
    if state in commitment_states:
        result.extend(("commit", decision) for decision in sorted(game.decisions))
    if time < game.horizon:
        result.extend(("act", action) for action in game.available_actions(state, time))
    return tuple(result)


def _policy_from_local_mapping(
    game: FiniteConventionGame,
    mapping: dict[LocalKey, Choice],
    backend: Backend,
    mask_observations: bool,
) -> PolicyNode:
    def recurse(
        time: int,
        state: str,
        latest_observation: str,
        belief: tuple[Number, ...],
    ) -> PolicyNode:
        key = (
            time,
            state,
            "masked" if mask_observations else latest_observation,
        )
        kind, identifier = mapping[key]
        if kind == "commit":
            return PolicyNode(
                kind="commit",
                time=time,
                state=state,
                belief=belief,
                decision=identifier,
            )
        branches = branch_dynamics(game, time, state, identifier, belief, backend)
        return PolicyNode(
            kind="act",
            time=time,
            state=state,
            belief=belief,
            action=identifier,
            branches=tuple(
                PolicyBranch(
                    next_state=branch.next_state,
                    observation=branch.observation,
                    probability=branch.probability,
                    expected_immediate_cost=branch.expected_immediate_cost,
                    child=recurse(
                        time + 1,
                        branch.next_state,
                        branch.observation,
                        branch.posterior,
                    ),
                )
                for branch in branches
            ),
        )

    start_observation = "masked" if mask_observations else "start"
    return recurse(
        0,
        game.initial_state,
        start_observation,
        initial_belief(game, backend),
    )
