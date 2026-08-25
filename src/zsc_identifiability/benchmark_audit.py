"""Matching-contract and restricted-policy audits for Phase 3 populations."""

from __future__ import annotations

from typing import Any

from zsc_identifiability.benchmark_models import (
    AuditItem,
    GeneratedPopulation,
    MatchingAudit,
    MatchingContractSpec,
    PopulationMetrics,
    ShortcutAudit,
)
from zsc_identifiability.benchmark_policies import (
    fixed_action_sequence_policy,
    solve_observation_restricted,
)
from zsc_identifiability.metrics import compute_distributions, evaluate
from zsc_identifiability.numeric import Backend, Number, close, number
from zsc_identifiability.population_metrics import compute as compute_metrics
from zsc_identifiability.solver import solve


def audit_pair(
    left: GeneratedPopulation,
    right: GeneratedPopulation,
    contract: MatchingContractSpec,
    backend: Backend = "fraction",
    left_metrics: PopulationMetrics | None = None,
    right_metrics: PopulationMetrics | None = None,
) -> MatchingAudit:
    left_metrics = left_metrics or compute_metrics(left, backend)
    right_metrics = right_metrics or compute_metrics(right, backend)
    structural = _structural_checks(left, right)
    passive_match: bool | None = None
    if contract.require_passive_history_match:
        passive_match = _passive_histories_equal(left, right, backend)
    divergence_match: bool | None = None
    if contract.require_divergence_profile_match:
        divergence_match = _divergence_profiles_equal(left_metrics, right_metrics)
    items: list[AuditItem] = []
    for rule in contract.metric_rules:
        left_value = left_metrics.values.get(rule.metric)
        right_value = right_metrics.values.get(rule.metric)
        if left_value is None or right_value is None:
            items.append(
                AuditItem(
                    rule.metric,
                    left_value,
                    right_value,
                    None,
                    number(rule.tolerance, backend),
                    "not_applicable",
                    "metric is unavailable for at least one population",
                    rule.role,
                )
            )
            continue
        difference = _subtract(left_value, right_value, backend)
        tolerance = number(rule.tolerance, backend)
        if rule.relation == "equal":
            passed = _absolute(difference) <= tolerance
            reason = "difference lies within the declared equality tolerance"
        else:
            expected = number(rule.expected_difference or "0", backend)
            passed = _absolute(difference - expected) <= tolerance
            reason = "difference matches the declared treatment effect"
        items.append(
            AuditItem(
                rule.metric,
                left_value,
                right_value,
                difference,
                tolerance,
                "pass" if passed else "fail",
                reason if passed else f"observed difference {difference} violates the rule",
                rule.role,
            )
        )
    structural_pass = all(structural.values()) if contract.require_structural_match else True
    passed = (
        structural_pass
        and passive_match is not False
        and divergence_match is not False
        and all(item.status == "pass" for item in items)
    )
    return MatchingAudit(
        contract.contract_id,
        left.descriptor.population_id,
        right.descriptor.population_id,
        passed,
        structural,
        passive_match,
        divergence_match,
        tuple(items),
    )


def audit_shortcuts(
    population: GeneratedPopulation,
    backend: Backend = "fraction",
) -> ShortcutAudit:
    game = population.game
    descriptor = population.descriptor
    commitment_states = frozenset(descriptor.commitment_states)
    prior = evaluate(
        game,
        fixed_action_sequence_policy(
            game,
            descriptor.passive_reference_actions,
            backend,
            fixed_decision=sorted(game.decisions)[0],
        ),
        backend,
    ).prior_risk
    evidence_blind = solve_observation_restricted(
        game,
        commitment_states,
        backend,
        mask_observations=True,
    )
    memoryless = solve_observation_restricted(
        game,
        commitment_states,
        backend,
        mask_observations=False,
    )
    history_solution = solve(
        game,
        "task",
        "information",
        backend,
        commitment_states=commitment_states,
    )
    history_evaluation = evaluate(game, history_solution.policy, backend)
    reference = fixed_action_sequence_policy(
        game,
        descriptor.passive_reference_actions,
        backend,
    )
    original_reference = evaluate(game, reference, backend)
    stripped_game = game.model_copy(update={"post_commitment_observations": ()})
    stripped_reference = evaluate(stripped_game, reference, backend)
    postcommit_leak_free = close(
        original_reference.residual_risk_precommitment,
        stripped_reference.residual_risk_precommitment,
    ) and (
        original_reference.dri_precommitment == stripped_reference.dri_precommitment
        or (
            original_reference.dri_precommitment is not None
            and stripped_reference.dri_precommitment is not None
            and close(
                original_reference.dri_precommitment,
                stripped_reference.dri_precommitment,
            )
        )
    )
    public_state_leak_free = _public_state_leak_free(population, reference, backend)
    identifier_leak_free = _identifier_leak_free(population)
    universal_response_blocked = prior == 0 or prior > 0
    if prior > 0:
        universal_response_blocked = evidence_blind.evaluation.actual_policy_loss > 0
    valueless_probe_tie_break_ok = True
    if descriptor.cell_id == "active_boundary":
        net_solution = solve(
            game,
            "task",
            "net_regret",
            backend,
            commitment_states=commitment_states,
        )
        valueless_probe_tie_break_ok = net_solution.policy.kind == "commit"
    memory_required = (
        descriptor.family_kind == "factorized_identity_memory"
        and descriptor.cell_id == "remember_response"
        and int(descriptor.matched_nuisances["distractor_steps"]) > 0
    )
    checks = {
        "evidence_blind_cannot_remove_required_risk": (
            prior == 0 or evidence_blind.evaluation.actual_policy_loss > 0
        ),
        "memory_requirement": (
            not memory_required
            or memoryless.evaluation.actual_policy_loss > history_evaluation.actual_policy_loss
        ),
        "postcommit_leak_free": postcommit_leak_free,
        "public_state_leak_free": public_state_leak_free,
        "identifier_leak_free": identifier_leak_free,
        "universal_response_blocked": universal_response_blocked,
        "valueless_probe_tie_break_ok": valueless_probe_tie_break_ok,
    }
    return ShortcutAudit(
        descriptor.population_id,
        all(checks.values()),
        prior,
        evidence_blind.evaluation.actual_policy_loss,
        memoryless.evaluation.actual_policy_loss,
        history_evaluation.actual_policy_loss,
        postcommit_leak_free,
        public_state_leak_free,
        identifier_leak_free,
        universal_response_blocked,
        valueless_probe_tie_break_ok,
        checks,
    )


def _structural_checks(
    left: GeneratedPopulation,
    right: GeneratedPopulation,
) -> dict[str, bool]:
    left_game = left.game
    right_game = right.game
    return {
        "mode_priors": left_game.modes == right_game.modes,
        "response_signatures": (
            left.descriptor.response_signature_by_mode
            == right.descriptor.response_signature_by_mode
        ),
        "decision_losses": left_game.decision_losses == right_game.decision_losses,
        "base_team_return": (left.descriptor.base_team_return == right.descriptor.base_team_return),
        "horizon": left_game.horizon == right_game.horizon,
        "states": left_game.states == right_game.states,
        "observations": left_game.observations == right_game.observations,
        "actions_and_availability": left_game.actions == right_game.actions,
        "intervention_costs": _cost_signature(left) == _cost_signature(right),
        "best_response_event_features": (
            left.descriptor.best_response_event_features
            == right.descriptor.best_response_event_features
        ),
        "passive_reference_policy": (
            left.descriptor.passive_reference_actions == right.descriptor.passive_reference_actions
        ),
        "commitment_states": (
            left.descriptor.commitment_states == right.descriptor.commitment_states
        ),
        "observation_budget": (
            len(left.descriptor.passive_reference_actions)
            == len(right.descriptor.passive_reference_actions)
            and bool(left_game.post_commitment_observations)
            == bool(right_game.post_commitment_observations)
        ),
    }


def _cost_signature(population: GeneratedPopulation) -> dict[tuple[int, str, str], tuple[str, ...]]:
    signature: dict[tuple[int, str, str], set[str]] = {}
    for row in population.game.kernels:
        key = (row.time, row.state, row.action)
        signature.setdefault(key, set()).update(outcome.cost for outcome in row.outcomes)
    return {key: tuple(sorted(values)) for key, values in signature.items()}


def _passive_histories_equal(
    left: GeneratedPopulation,
    right: GeneratedPopulation,
    backend: Backend,
) -> bool:
    left_policy = fixed_action_sequence_policy(
        left.game, left.descriptor.passive_reference_actions, backend
    )
    right_policy = fixed_action_sequence_policy(
        right.game, right.descriptor.passive_reference_actions, backend
    )
    left_histories = compute_distributions(left.game, left_policy, backend)
    right_histories = compute_distributions(right.game, right_policy, backend)
    return left_histories.by_mode == right_histories.by_mode


def _divergence_profiles_equal(
    left: PopulationMetrics,
    right: PopulationMetrics,
) -> bool:
    left_values = sorted(
        tuple(float(item) for item in curve) for curve in left.prefix_tv_curves.values()
    )
    right_values = sorted(
        tuple(float(item) for item in curve) for curve in right.prefix_tv_curves.values()
    )
    return left_values == right_values


def _public_state_leak_free(
    population: GeneratedPopulation,
    reference_policy: Any,
    backend: Backend,
) -> bool:
    histories = compute_distributions(population.game, reference_policy, backend)
    terminal_states: set[str] = set()
    for history in histories.decisions_by_history:
        events = [token for token in history.split(";") if not token.startswith("stop@")]
        if events:
            terminal_states.add(events[-1].split(">", 1)[1].split("/", 1)[0])
    return len(terminal_states) <= 1


def _identifier_leak_free(population: GeneratedPopulation) -> bool:
    public = (
        *population.game.states,
        *population.game.observations,
        *population.game.action_ids,
    )
    for mode in population.game.mode_ids:
        if any(mode in identifier for identifier in public):
            return False
    return True


def _subtract(
    left: Number | float | str,
    right: Number | float | str,
    backend: Backend,
) -> Number:
    if isinstance(left, str) or isinstance(right, str):
        raise TypeError("non-numeric metric cannot be used in a matching rule")
    if isinstance(left, float) or isinstance(right, float):
        return float(left) - float(right)
    return number(left, backend) - number(right, backend)


def _absolute(value: Number) -> Number:
    return value if value >= 0 else -value
