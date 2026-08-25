"""Exact finite-tree evaluation for learned stochastic policies."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch

from zsc_identifiability.belief import best_decision, initial_belief, zero_loss_sets
from zsc_identifiability.frontier import compute as compute_frontier
from zsc_identifiability.learning_env import (
    VectorConventionEnvironment,
    build_observation_layout,
    encode_runtime_observation,
    runtime_action_mask,
)
from zsc_identifiability.learning_methods import LearnedPolicy
from zsc_identifiability.learning_models import (
    EvaluationMode,
    LearnedPolicyEvaluation,
    LearningGame,
    ReconnaissanceEvaluation,
)
from zsc_identifiability.numeric import parse_rational
from zsc_identifiability.solver import solve


@dataclass(frozen=True)
class _TerminalMass:
    mode_index: int
    evidence_key: str
    probability: float
    cost: float
    loss: float
    commitment_time: int
    probed: bool
    response_prediction: tuple[float, ...] | None
    response_target: int
    response_reconstruction_nll: float | None


def evaluate_neural_policy_exact(
    model: LearnedPolicy,
    item: LearningGame,
    *,
    method_id: str,
    mode: EvaluationMode = "greedy",
    action_class: str = "task",
    base_team_return: float = 100.0,
    loss_scale: float = 40.0,
    device: torch.device | None = None,
    identity_label_response_classes: tuple[int, ...] | None = None,
) -> LearnedPolicyEvaluation:
    """Enumerate every mode, policy action, and environment outcome exactly."""
    if device is None:
        device = next(model.parameters()).device
    game = item.game
    layout = build_observation_layout(game)
    was_training = model.training
    model.eval()
    terminals: list[_TerminalMass] = []
    prior = [float(value) for value in game.prior_exact()]
    active_actions = {action.id for action in game.actions if not action.passive}

    def recurse(
        mode_index: int,
        probability: float,
        time: int,
        state: str,
        latest_observation: str,
        previous_action: str,
        previous_reward: float,
        hidden: torch.Tensor,
        evidence: tuple[str, ...],
        accumulated_cost: float,
        probed: bool,
    ) -> None:
        mask = runtime_action_mask(item, time, state, layout, action_class)  # type: ignore[arg-type]
        observation = encode_runtime_observation(
            game,
            layout,
            time,
            state,
            latest_observation,
            previous_action,
            previous_reward,
            mask,
        )
        with torch.no_grad():
            output = model.forward_step(
                torch.as_tensor(observation[None, :], device=device),
                hidden,
                torch.as_tensor(mask[None, :], device=device),
            )
            if mode == "greedy":
                probabilities = torch.zeros_like(output.logits)
                probabilities[0, int(torch.argmax(output.logits[0]).item())] = 1
            else:
                probabilities = torch.softmax(output.logits, dim=-1)
        for action_index, action_probability in enumerate(probabilities[0].tolist()):
            if action_probability <= 0 or not mask[action_index]:
                continue
            branch_probability = probability * action_probability
            action_id = layout.action_ids[action_index]
            if action_id.startswith("commit:"):
                decision = action_id.removeprefix("commit:")
                loss = float(game.loss_exact(game.mode_ids[mode_index], decision))
                prediction = _response_prediction(
                    output.identity_logits,
                    output.cluster_logits,
                    identity_label_response_classes,
                    len(game.decisions),
                )
                reconstruction_nll = _response_reconstruction_nll(
                    output.response_logits,
                    latest_observation,
                    game.observations,
                )
                response_target = _mode_response_class(game, mode_index)
                terminals.append(
                    _TerminalMass(
                        mode_index,
                        "|".join(evidence) or "<empty>",
                        branch_probability,
                        accumulated_cost,
                        loss,
                        time,
                        probed,
                        prediction,
                        response_target,
                        reconstruction_nll,
                    )
                )
                continue
            row = game.kernel(time, state, action_id, game.mode_ids[mode_index])
            for outcome in row.outcomes:
                outcome_probability = float(parse_rational(outcome.probability))
                if outcome_probability <= 0:
                    continue
                cost = float(parse_rational(outcome.cost))
                token = f"t{time}:{action_id}>{outcome.next_state}/{outcome.observation}"
                recurse(
                    mode_index,
                    branch_probability * outcome_probability,
                    time + 1,
                    outcome.next_state,
                    outcome.observation,
                    action_id,
                    -cost / loss_scale,
                    output.hidden,
                    (*evidence, token),
                    accumulated_cost + cost,
                    probed or action_id in active_actions,
                )

    for mode_index, prior_probability in enumerate(prior):
        recurse(
            mode_index,
            prior_probability,
            0,
            game.initial_state,
            "<start>",
            "<start>",
            0.0,
            model.initial_hidden(1, device),
            (),
            0.0,
            False,
        )
    result = _summarize_terminals(
        item,
        terminals,
        method_id=method_id,
        mode=mode,
        evaluator="exact_neural_policy_tree",
        base_team_return=base_team_return,
        loss_scale=loss_scale,
    )
    if was_training:
        model.train()
    return result


def _summarize_terminals(
    item: LearningGame,
    terminals: list[_TerminalMass],
    *,
    method_id: str,
    mode: EvaluationMode,
    evaluator: str,
    base_team_return: float,
    loss_scale: float,
) -> LearnedPolicyEvaluation:
    game = item.game
    prior = [float(value) for value in game.prior_exact()]
    total_probability = sum(terminal.probability for terminal in terminals)
    if not math.isclose(total_probability, 1.0, abs_tol=1e-6):
        raise RuntimeError(f"learned policy evaluation has mass {total_probability}, expected one")
    expected_cost = sum(terminal.probability * terminal.cost for terminal in terminals)
    actual_loss = sum(terminal.probability * terminal.loss for terminal in terminals)
    commitment_time = sum(terminal.probability * terminal.commitment_time for terminal in terminals)
    probe_probability = sum(terminal.probability for terminal in terminals if terminal.probed)
    commitment_distribution: dict[str, float] = defaultdict(float)
    for terminal in terminals:
        commitment_distribution[str(terminal.commitment_time)] += terminal.probability
    history_masses = _history_masses(terminals, len(game.mode_ids))
    residual_risk = _residual_bayes_risk(game, history_masses)
    prior_belief = initial_belief(game, "float")
    _, prior_risk_number = best_decision(game, prior_belief, "float")
    prior_risk = float(prior_risk_number)
    policy_dri = None if prior_risk == 0 else (prior_risk - residual_risk) / prior_risk
    identity_mi = _mutual_information(history_masses, prior)
    signature_mi = _signature_mutual_information(game, history_masses, prior)
    total_regret = expected_cost + actual_loss
    decision_gap = actual_loss - residual_risk
    active_solution = solve(
        game,
        "task",
        "net_regret",
        "float",
        commitment_states=item.commitment_states,
    )
    fixed_return = base_team_return - prior_risk
    active_return = base_team_return - float(active_solution.total_cost_plus_risk)
    denominator = active_return - fixed_return
    team_return = base_team_return - total_regret
    normalized = (
        None if math.isclose(denominator, 0.0) else (team_return - fixed_return) / denominator
    )
    frontier = compute_frontier(
        game,
        "task",
        "float",
        commitment_states=item.commitment_states,
    )
    frontier_distance = _distance_to_frontier(
        expected_cost,
        residual_risk,
        tuple(
            (float(point.expected_cost), float(point.residual_risk))
            for point in frontier.convexified_envelope
        ),
        loss_scale,
    )
    response_accuracy, brier, calibration = _prediction_metrics(terminals)
    reconstruction_numerator = 0.0
    reconstruction_mass = 0.0
    for terminal in terminals:
        if terminal.response_reconstruction_nll is not None:
            reconstruction_numerator += terminal.probability * terminal.response_reconstruction_nll
            reconstruction_mass += terminal.probability
    reconstruction_loss = (
        None if reconstruction_mass == 0 else reconstruction_numerator / reconstruction_mass
    )
    return LearnedPolicyEvaluation(
        population_id=item.source_population_id,
        method_id=method_id,
        mode=mode,
        evaluator=evaluator,
        team_return=team_return,
        expected_intervention_cost=expected_cost,
        actual_confusion_loss=actual_loss,
        residual_bayes_risk=residual_risk,
        decision_utilization_gap=decision_gap,
        total_regret=total_regret,
        policy_dri=policy_dri,
        probe_probability=probe_probability,
        expected_commitment_time=commitment_time,
        identity_mutual_information_bits=identity_mi,
        decision_signature_mutual_information_bits=signature_mi,
        active_frontier_distance=frontier_distance,
        oracle_normalized_return=normalized,
        response_signature_accuracy=response_accuracy,
        belief_brier_score=brier,
        expected_calibration_error=calibration,
        partner_response_prediction_loss=reconstruction_loss,
        commitment_time_distribution=dict(sorted(commitment_distribution.items())),
        applicability_flags={
            "policy_dri": policy_dri is not None,
            "active_frontier_distance": True,
            "oracle_normalized_return": normalized is not None,
            "response_signature_accuracy": response_accuracy is not None,
            "belief_calibration": brier is not None,
            "partner_response_prediction_loss": reconstruction_loss is not None,
        },
    )


def evaluate_neural_policy_sampled(
    model: LearnedPolicy,
    item: LearningGame,
    *,
    method_id: str,
    mode: EvaluationMode = "greedy",
    action_class: str = "task",
    episodes: int = 100_000,
    batch_size: int = 1_024,
    seed: int = 1729,
    base_team_return: float = 100.0,
    loss_scale: float = 40.0,
    device: torch.device | None = None,
    identity_label_response_classes: tuple[int, ...] | None = None,
) -> LearnedPolicyEvaluation:
    """Independently calibrate exact traversal with PCG64-sampled episodes."""
    if episodes < 1 or batch_size < 1:
        raise ValueError("sampled evaluation requires positive episodes and batch size")
    if action_class not in {"passive", "task"}:
        raise ValueError(f"unsupported sampled action class: {action_class!r}")
    if device is None:
        device = next(model.parameters()).device
    game = item.game
    layout = build_observation_layout(game)
    active_actions = {action.id for action in game.actions if not action.passive}
    terminals: list[_TerminalMass] = []
    was_training = model.training
    torch_rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    model.eval()
    generated = 0
    while generated < episodes:
        count = min(batch_size, episodes - generated)
        environment = VectorConventionEnvironment(
            (item,),
            seed + generated,
            count,
            action_class=action_class,  # type: ignore[arg-type]
            loss_scale=loss_scale,
        )
        batch = environment.reset()
        hidden = model.initial_hidden(count, device)
        evidence: list[list[str]] = [[] for _ in range(count)]
        probed = np.zeros(count, dtype=np.bool_)
        active = np.ones(count, dtype=np.bool_)
        while active.any():
            safe_masks = batch.action_masks.copy()
            safe_masks[~active, 0] = True
            with torch.no_grad():
                output = model.forward_step(
                    torch.as_tensor(batch.observations, device=device),
                    hidden,
                    torch.as_tensor(safe_masks, device=device),
                )
                if mode == "greedy":
                    actions = torch.argmax(output.logits, dim=-1)
                else:
                    actions = torch.distributions.Categorical(  # type: ignore[no-untyped-call]
                        logits=output.logits
                    ).sample()
            action_values = actions.cpu().numpy()
            action_values[~active] = -1
            before_times = environment.times.copy()
            step = environment.step(action_values)
            for index in np.flatnonzero(active).tolist():
                info = step.infos[index]
                action_id = layout.action_ids[int(action_values[index])]
                if not action_id.startswith("commit:"):
                    evidence[index].append(
                        f"t{before_times[index]}:{action_id}>"
                        f"{info['next_state']}/{info['observation']}"
                    )
                    probed[index] = bool(probed[index] or action_id in active_actions)
                if step.terminated[index]:
                    prediction = _response_prediction(
                        output.identity_logits[index : index + 1]
                        if output.identity_logits is not None
                        else None,
                        output.cluster_logits[index : index + 1]
                        if output.cluster_logits is not None
                        else None,
                        identity_label_response_classes,
                        len(game.decisions),
                    )
                    reconstruction_nll = _response_reconstruction_nll(
                        output.response_logits[index : index + 1]
                        if output.response_logits is not None
                        else None,
                        str(environment.latest_observations[index]),
                        game.observations,
                    )
                    mode_index = int(environment.mode_indices[index])
                    terminals.append(
                        _TerminalMass(
                            mode_index=mode_index,
                            evidence_key="|".join(evidence[index]) or "<empty>",
                            probability=1 / episodes,
                            cost=float(environment.episode_costs[index]),
                            loss=float(environment.episode_losses[index]),
                            commitment_time=int(before_times[index]),
                            probed=bool(probed[index]),
                            response_prediction=prediction,
                            response_target=_mode_response_class(game, mode_index),
                            response_reconstruction_nll=reconstruction_nll,
                        )
                    )
            active &= ~step.terminated
            hidden = output.hidden
            batch = step
        generated += count
    result = _summarize_terminals(
        item,
        terminals,
        method_id=method_id,
        mode=mode,
        evaluator=f"sampled_neural_policy_{episodes}_episodes",
        base_team_return=base_team_return,
        loss_scale=loss_scale,
    )
    if was_training:
        model.train()
    torch.random.set_rng_state(torch_rng_state)
    return result


def evaluate_reconnaissance_policy(
    model: LearnedPolicy,
    item: LearningGame,
    *,
    method_id: str = "csp_style_reconnaissance",
    episodes: int = 10_000,
    seed: int = 1729,
    base_team_return: float = 100.0,
    loss_scale: float = 40.0,
    device: torch.device | None = None,
) -> ReconnaissanceEvaluation:
    """Evaluate an explicitly relaxed two-episode, same-partner protocol."""
    if episodes < 1:
        raise ValueError("reconnaissance evaluation requires at least one episode")
    if device is None:
        device = next(model.parameters()).device
    environment = VectorConventionEnvironment(
        (item,), seed, 1, action_class="task", loss_scale=loss_scale
    )
    model.eval()
    reconnaissance_returns: list[float] = []
    scored_returns: list[float] = []
    reconnaissance_costs: list[float] = []
    reconnaissance_losses: list[float] = []
    interaction_counts: list[int] = []
    for _ in range(episodes):
        environment.reset()
        mode_index = int(environment.mode_indices[0])
        hidden = model.initial_hidden(1, device)
        hidden, first_cost, first_loss, first_interactions = _rollout_sampled_episode(
            model, environment, hidden, device, greedy=True
        )
        environment.reset()
        environment.game_indices[0] = 0
        environment.mode_indices[0] = mode_index
        _, second_cost, second_loss, _ = _rollout_sampled_episode(
            model, environment, hidden, device, greedy=True
        )
        reconnaissance_returns.append(base_team_return - first_cost - first_loss)
        scored_returns.append(base_team_return - second_cost - second_loss)
        reconnaissance_costs.append(first_cost)
        reconnaissance_losses.append(first_loss)
        interaction_counts.append(first_interactions)
    first_mean = float(np.mean(reconnaissance_returns))
    second_mean = float(np.mean(scored_returns))
    return ReconnaissanceEvaluation(
        population_id=item.source_population_id,
        method_id=method_id,
        episodes=episodes,
        scored_episode_return=second_mean,
        reconnaissance_episode_return=first_mean,
        combined_return_sum=first_mean + second_mean,
        combined_return_mean=(first_mean + second_mean) / 2,
        reconnaissance_cost=float(np.mean(reconnaissance_costs)),
        reconnaissance_loss=float(np.mean(reconnaissance_losses)),
        extra_partner_interactions=float(np.mean(interaction_counts)),
    )


def _rollout_sampled_episode(
    model: LearnedPolicy,
    environment: VectorConventionEnvironment,
    hidden: torch.Tensor,
    device: torch.device,
    *,
    greedy: bool,
) -> tuple[torch.Tensor, float, float, int]:
    batch = environment.current_batch()
    cost = 0.0
    loss = 0.0
    interactions = 0
    while not batch.terminated[0]:
        with torch.no_grad():
            output = model.forward_step(
                torch.as_tensor(batch.observations, device=device),
                hidden,
                torch.as_tensor(batch.action_masks, device=device),
            )
            if greedy:
                action = int(torch.argmax(output.logits[0]).item())
            else:
                distribution = torch.distributions.Categorical(logits=output.logits)
                action = int(distribution.sample()[0].item())  # type: ignore[no-untyped-call]
        batch = environment.step(np.asarray([action]))
        hidden = output.hidden
        cost += float(batch.infos[0].get("intervention_cost", 0.0))
        loss += float(batch.infos[0].get("confusion_loss", 0.0))
        if not str(batch.infos[0].get("action", "")).startswith("commit:"):
            interactions += 1
    return hidden, cost, loss, interactions


def _history_masses(terminals: list[_TerminalMass], mode_count: int) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for terminal in terminals:
        result.setdefault(terminal.evidence_key, np.zeros(mode_count, dtype=np.float64))[
            terminal.mode_index
        ] += terminal.probability
    return result


def _residual_bayes_risk(game: object, histories: dict[str, np.ndarray]) -> float:
    from zsc_identifiability.models import FiniteConventionGame

    if not isinstance(game, FiniteConventionGame):
        raise TypeError("expected FiniteConventionGame")
    total = 0.0
    for masses in histories.values():
        total += min(
            sum(
                masses[index] * float(game.loss_exact(mode, decision))
                for index, mode in enumerate(game.mode_ids)
            )
            for decision in game.decisions
        )
    return total


def _mutual_information(histories: dict[str, np.ndarray], prior: list[float]) -> float:
    result = 0.0
    for masses in histories.values():
        history_probability = float(masses.sum())
        if history_probability <= 0:
            continue
        for index, joint in enumerate(masses.tolist()):
            if joint > 0:
                result += joint * math.log2(joint / (prior[index] * history_probability))
    return result


def _signature_mutual_information(
    game: object, histories: dict[str, np.ndarray], prior: list[float]
) -> float:
    from zsc_identifiability.models import FiniteConventionGame

    if not isinstance(game, FiniteConventionGame):
        raise TypeError("expected FiniteConventionGame")
    sets = zero_loss_sets(game)
    signatures = tuple(tuple(sorted(sets[mode])) for mode in game.mode_ids)
    unique = tuple(sorted(set(signatures)))
    signature_prior = [
        sum(prior[index] for index, value in enumerate(signatures) if value == signature)
        for signature in unique
    ]
    collapsed: dict[str, np.ndarray] = {}
    for history, masses in histories.items():
        collapsed[history] = np.asarray(
            [
                sum(masses[index] for index, value in enumerate(signatures) if value == signature)
                for signature in unique
            ]
        )
    return _mutual_information(collapsed, signature_prior)


def _response_prediction(
    identity_logits: torch.Tensor | None,
    cluster_logits: torch.Tensor | None,
    label_response_classes: tuple[int, ...] | None,
    response_class_count: int,
) -> tuple[float, ...] | None:
    if identity_logits is not None and label_response_classes is not None:
        if identity_logits.shape[-1] != len(label_response_classes):
            raise ValueError("identity head and response-class mapping disagree")
        probabilities = torch.softmax(identity_logits[0], dim=-1).tolist()
        result = [0.0] * response_class_count
        for probability, response_class in zip(probabilities, label_response_classes, strict=True):
            result[response_class] += float(probability)
        return tuple(result)
    if cluster_logits is not None and cluster_logits.shape[-1] == response_class_count:
        return tuple(float(value) for value in torch.softmax(cluster_logits[0], dim=-1).tolist())
    return None


def _distance_to_frontier(
    cost: float,
    risk: float,
    points: tuple[tuple[float, float], ...],
    scale: float,
) -> float:
    normalized = np.asarray((cost / scale, risk / scale), dtype=np.float64)
    coordinates = [np.asarray((x / scale, y / scale), dtype=np.float64) for x, y in points]
    distances = [float(np.linalg.norm(normalized - point)) for point in coordinates]
    for left, right in zip(coordinates, coordinates[1:], strict=False):
        direction = right - left
        denominator = float(direction @ direction)
        weight = 0.0 if denominator == 0 else float((normalized - left) @ direction / denominator)
        projection = left + min(1.0, max(0.0, weight)) * direction
        distances.append(float(np.linalg.norm(normalized - projection)))
    return min(distances)


def _response_reconstruction_nll(
    response_logits: torch.Tensor | None,
    latest_observation: str,
    observation_ids: tuple[str, ...],
) -> float | None:
    if response_logits is None or latest_observation not in observation_ids:
        return None
    target = observation_ids.index(latest_observation)
    return float(-torch.log_softmax(response_logits[0], dim=-1)[target].item())


def _mode_response_class(game: object, mode_index: int) -> int:
    from zsc_identifiability.models import FiniteConventionGame

    if not isinstance(game, FiniteConventionGame):
        raise TypeError("expected FiniteConventionGame")
    decisions = tuple(sorted(game.decisions))
    mode = game.mode_ids[mode_index]
    correct = next(decision for decision in decisions if game.loss_exact(mode, decision) == 0)
    return decisions.index(correct)


def _prediction_metrics(
    terminals: list[_TerminalMass],
) -> tuple[float | None, float | None, float | None]:
    predicted = [item for item in terminals if item.response_prediction is not None]
    if not predicted:
        return None, None, None
    accuracy = 0.0
    brier = 0.0
    bins: list[list[float]] = [[] for _ in range(10)]
    bin_correct: list[list[float]] = [[] for _ in range(10)]
    for item in predicted:
        if item.response_prediction is None:  # pragma: no cover - narrowed above
            continue
        probabilities = np.asarray(item.response_prediction)
        choice = int(np.argmax(probabilities))
        confidence = float(probabilities[choice])
        correct = float(choice == item.response_target)
        accuracy += item.probability * correct
        target = np.zeros(len(probabilities))
        target[item.response_target] = 1
        brier += item.probability * float(np.square(probabilities - target).sum())
        bin_index = min(9, int(confidence * 10))
        bins[bin_index].append(item.probability * confidence)
        bin_correct[bin_index].append(item.probability * correct)
    calibration = 0.0
    for confidence_values, correct_values in zip(bins, bin_correct, strict=True):
        if confidence_values:
            calibration += abs(sum(confidence_values) - sum(correct_values))
    return accuracy, brier, calibration
