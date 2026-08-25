"""Sequence-correct PPO training and checkpointing for Stage 4."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as functional

from zsc_identifiability.learning_env import StepBatch, VectorConventionEnvironment
from zsc_identifiability.learning_evaluation import evaluate_neural_policy_exact
from zsc_identifiability.learning_methods import (
    LearnedPolicy,
    TalentsStylePolicy,
    TomSelectorStylePolicy,
    action_distribution,
    build_method,
    model_metadata,
)
from zsc_identifiability.learning_models import (
    LearningAuditSuite,
    LearningCellPools,
    LearningMethodSpec,
    TrainingRunManifest,
)
from zsc_identifiability.learning_specialists import deterministic_two_means
from zsc_identifiability.numeric import parse_rational
from zsc_identifiability.policy import PolicyNode
from zsc_identifiability.solver import solve


@dataclass
class EpisodeRecord:
    observations: list[np.ndarray]
    action_masks: list[np.ndarray]
    actions: list[int]
    old_log_probabilities: list[float]
    values: list[float]
    rewards: list[float]
    partner_identities: list[int]
    response_targets: list[int]
    strategy_clusters: list[int]
    advantages: np.ndarray | None = None
    returns: np.ndarray | None = None


def train_method(
    suite: LearningAuditSuite,
    method: LearningMethodSpec,
    cell: LearningCellPools,
    output_dir: str | Path,
    *,
    seed: int,
    transitions: int,
    device_name: Literal["cpu", "mps", "auto"] | None = None,
    resume_checkpoint: str | Path | None = None,
) -> TrainingRunManifest:
    """Train one method/cell/seed with frozen validation-based selection."""
    if transitions < 1:
        raise ValueError("training transitions must be positive")
    _set_determinism(seed)
    device = _select_device(device_name or suite.budget.device)
    action_class: Literal["passive", "task"] = (
        "passive" if method.method_id == "gru_ppo_passive" else "task"
    )
    environment = VectorConventionEnvironment(
        cell.train,
        seed,
        suite.budget.num_envs,
        action_class=action_class,
        loss_scale=float(parse_rational(suite.loss_scale)),
    )
    model = build_method(
        method,
        environment.layout.observation_size,
        environment.layout.action_size,
        environment.partner_identity_count,
        environment.response_target_count,
    ).to(device)
    if isinstance(model, TalentsStylePolicy) and resume_checkpoint is None:
        model.pretraining_metadata = _bootstrap_talents_style(
            model,
            cell,
            seed=seed,
            device=device,
            loss_scale=float(parse_rational(suite.loss_scale)),
            learning_rate=method.config.learning_rate,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=method.config.learning_rate)
    run_id = f"{cell.cell_id}--{method.method_id}--seed-{seed}"
    run_dir = Path(output_dir).resolve() / run_id
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "training-metrics.json"
    training_rows: list[dict[str, float | int]] = []
    completed = 0
    best_key: tuple[float, float, int] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_step = 0
    if resume_checkpoint is not None:
        completed, training_rows, best_key, best_state, best_step = _restore_training_state(
            Path(resume_checkpoint),
            model,
            optimizer,
            environment,
            suite,
            method,
            cell,
            seed,
            run_dir,
        )
        if completed >= transitions:
            raise ValueError(
                f"resume checkpoint already contains {completed} transitions; "
                f"requested total is {transitions}"
            )
    next_checkpoint = (
        (completed // suite.budget.checkpoint_interval) + 1
    ) * suite.budget.checkpoint_interval
    next_checkpoint = min(next_checkpoint, transitions)
    while completed < transitions:
        target = min(method.config.transitions_per_update, transitions - completed)
        episodes, collected = _collect_complete_episodes(
            model,
            environment,
            method,
            target,
            completed,
            transitions,
            device,
        )
        _compute_advantages(episodes, method)
        losses = _ppo_update(model, optimizer, episodes, method, device)
        completed += collected
        row: dict[str, float | int] = {"transitions": completed, **losses}
        training_rows.append(row)
        if completed >= next_checkpoint or completed >= transitions:
            validation = evaluate_neural_policy_exact(
                model,
                cell.validation[0],
                method_id=method.method_id,
                mode="greedy",
                action_class=action_class,
                base_team_return=float(parse_rational(suite.base_team_return)),
                loss_scale=float(parse_rational(suite.loss_scale)),
                device=device,
                identity_label_response_classes=environment.partner_response_classes,
            )
            key = (
                validation.team_return,
                -validation.expected_intervention_cost,
                -completed,
            )
            row.update(
                {
                    "validation_team_return": validation.team_return,
                    "validation_cost": validation.expected_intervention_cost,
                    "validation_dri": (
                        float("nan") if validation.policy_dri is None else validation.policy_dri
                    ),
                }
            )
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_step = completed
            checkpoint_path = checkpoint_dir / f"step-{completed}.pt"
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                suite,
                method,
                cell,
                environment,
                seed,
                completed,
                best_key=best_key,
                best_state=best_state,
                best_step=best_step,
            )
            _write_json(metrics_path, training_rows)
            while next_checkpoint <= completed:
                next_checkpoint += suite.budget.checkpoint_interval
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_step = completed
    model.load_state_dict(best_state)
    best_path = run_dir / "best.pt"
    _save_checkpoint(
        best_path,
        model,
        optimizer,
        suite,
        method,
        cell,
        environment,
        seed,
        best_step,
        best_key=best_key,
        best_state=best_state,
        best_step=best_step,
    )
    _write_json(metrics_path, training_rows)
    configuration_hash = _configuration_hash(suite, method)
    manifest = TrainingRunManifest(
        schema_version=1,
        run_id=run_id,
        suite_id=suite.suite_id,
        cell_id=cell.cell_id,
        method_id=method.method_id,
        seed=seed,
        device=str(device),
        requested_transitions=transitions,
        completed_transitions=completed,
        configuration_hash=configuration_hash,
        training_pool_hashes=tuple(item.dynamics_hash for item in cell.train),
        validation_pool_hashes=tuple(item.dynamics_hash for item in cell.validation),
        checkpoint_path=str(best_path),
        metrics_path=str(metrics_path),
        deterministic=True,
        checkpoint_hash=hashlib.sha256(best_path.read_bytes()).hexdigest(),
        source_tree_hash=_source_hash(Path(__file__).resolve().parents[2]),
        python_version=platform.python_version(),
        dependency_versions=_dependency_versions(),
        rng_configuration={
            "seed": seed,
            "numpy_bit_generator": "PCG64",
            "torch_deterministic_algorithms": True,
        },
        pretraining_metadata=dict(getattr(model, "pretraining_metadata", {})),
    )
    _write_json(run_dir / "manifest.json", manifest.to_dict())
    return manifest


def _restore_training_state(
    checkpoint: Path,
    model: LearnedPolicy,
    optimizer: torch.optim.Optimizer,
    environment: VectorConventionEnvironment,
    suite: LearningAuditSuite,
    method: LearningMethodSpec,
    cell: LearningCellPools,
    seed: int,
    run_dir: Path,
) -> tuple[
    int,
    list[dict[str, float | int]],
    tuple[float, float, int] | None,
    dict[str, torch.Tensor] | None,
    int,
]:
    """Restore a complete PPO batch boundary for bitwise CPU continuation."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected = {
        "suite_id": suite.suite_id,
        "cell_id": cell.cell_id,
        "method_id": method.method_id,
        "seed": seed,
        "configuration_hash": _configuration_hash(suite, method),
    }
    observed = {
        "suite_id": payload.get("suite_id"),
        "cell_id": payload.get("cell_id"),
        "method_id": payload.get("method", {}).get("method_id"),
        "seed": payload.get("seed"),
        "configuration_hash": payload.get("configuration_hash"),
    }
    if observed != expected:
        raise ValueError(
            "resume checkpoint does not match the requested run: "
            f"expected {expected}, observed {observed}"
        )
    model.load_state_dict(payload["model_state"])
    if isinstance(model, TalentsStylePolicy):
        model.pretraining_metadata = dict(payload.get("pretraining_metadata", {}))
    optimizer.load_state_dict(payload["optimizer_state"])
    environment.load_state_dict(payload["environment_state"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.random.set_rng_state(payload["torch_random_state"])
    metrics_path = run_dir / "training-metrics.json"
    rows: list[dict[str, float | int]] = []
    if metrics_path.exists():
        decoded = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(decoded, list):
            raise ValueError("existing training metrics must be a JSON list")
        rows = [dict(item) for item in decoded]
    stored_key = payload.get("selection_key")
    best_key = None if stored_key is None else tuple(stored_key)
    selection_state = payload.get("selection_state")
    best_state = None if selection_state is None else dict(selection_state)
    return (
        int(payload["transitions"]),
        rows,
        best_key,
        best_state,
        int(payload.get("selection_step", 0)),
    )


def load_checkpoint(
    path: str | Path, device_name: str = "cpu"
) -> tuple[LearnedPolicy, dict[str, Any]]:
    device = torch.device(device_name)
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    method = LearningMethodSpec.model_validate(payload["method"])
    model = build_method(
        method,
        int(payload["observation_size"]),
        int(payload["action_size"]),
        int(payload["partner_identity_count"]),
        int(payload["response_count"]),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    if isinstance(model, TalentsStylePolicy):
        model.pretraining_metadata = dict(payload.get("pretraining_metadata", {}))
    model.eval()
    return model, payload


def _bootstrap_talents_style(
    model: TalentsStylePolicy,
    cell: LearningCellPools,
    *,
    seed: int,
    device: torch.device,
    loss_scale: float,
    learning_rate: float,
) -> dict[str, Any]:
    """Unsupervised sequence-VAE bootstrap from a balanced behavior mixture."""
    trajectories = _collect_balanced_offline_trajectories(
        cell,
        seed=seed + 70_000,
        trajectories_per_behavior=32,
        loss_scale=loss_scale,
    )
    maximum = max(len(item[0]) for item in trajectories)
    observation_size = trajectories[0][0][0].shape[0]
    action_size = trajectories[0][1][0].shape[0]
    count = len(trajectories)
    observations = np.zeros((count, maximum, observation_size), dtype=np.float32)
    masks = np.zeros((count, maximum, action_size), dtype=np.bool_)
    masks[:, :, 0] = True
    targets = np.full((count, maximum), -1, dtype=np.int64)
    valid = np.zeros((count, maximum), dtype=np.bool_)
    for index, (sequence, action_masks, response_targets, _) in enumerate(trajectories):
        length = len(sequence)
        observations[index, :length] = sequence
        masks[index, :length] = action_masks
        targets[index, :length] = response_targets
        valid[index, :length] = True
    observation_tensor = torch.as_tensor(observations, device=device)
    mask_tensor = torch.as_tensor(masks, device=device)
    target_tensor = torch.as_tensor(targets, device=device)
    valid_tensor = torch.as_tensor(valid, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(50):
        hidden = model.initial_hidden(count, device)
        response_logits: list[torch.Tensor] = []
        means: list[torch.Tensor] = []
        log_variances: list[torch.Tensor] = []
        for time in range(maximum):
            output = model.forward_step(observation_tensor[:, time], hidden, mask_tensor[:, time])
            hidden = torch.where(valid_tensor[:, time, None], output.hidden, hidden)
            if (
                output.response_logits is None
                or output.latent_mean is None
                or output.latent_log_variance is None
            ):
                raise RuntimeError("TALENTS-style bootstrap heads are unavailable")
            response_logits.append(output.response_logits)
            means.append(output.latent_mean)
            log_variances.append(output.latent_log_variance)
        response_stack = torch.stack(response_logits, dim=1)
        mean_stack = torch.stack(means, dim=1)
        log_variance_stack = torch.stack(log_variances, dim=1)
        response_valid = valid_tensor & (target_tensor >= 0)
        reconstruction = functional.cross_entropy(
            response_stack[response_valid], target_tensor[response_valid]
        )
        kl = (
            -0.5
            * (
                1
                + log_variance_stack[valid_tensor]
                - mean_stack[valid_tensor].square()
                - log_variance_stack[valid_tensor].exp()
            ).mean()
        )
        loss: torch.Tensor = reconstruction + 0.1 * kl
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
    with torch.no_grad():
        hidden = model.initial_hidden(count, device)
        final_means = torch.zeros(count, model.prototypes.shape[1], device=device)
        for time in range(maximum):
            output = model.forward_step(observation_tensor[:, time], hidden, mask_tensor[:, time])
            hidden = torch.where(valid_tensor[:, time, None], output.hidden, hidden)
            if output.latent_mean is None:
                raise RuntimeError("TALENTS-style latent mean is unavailable")
            final_means = torch.where(valid_tensor[:, time, None], output.latent_mean, final_means)
        assignments, centers = deterministic_two_means(final_means.cpu().numpy())
        model.prototypes.copy_(
            torch.as_tensor(centers, dtype=model.prototypes.dtype, device=device)
        )
    model.train()
    behavior_counts = {
        behavior: sum(item[3] == behavior for item in trajectories)
        for behavior in ("passive_oracle", "uniform_valid", "task_active_oracle")
    }
    return {
        "trajectory_count": count,
        "behavior_counts": behavior_counts,
        "hidden_mode_labels_used": False,
        "clustering": "deterministic_two_means",
        "cluster_sizes": [int(np.count_nonzero(assignments == item)) for item in range(2)],
        "pretraining_epochs": 50,
    }


def _collect_balanced_offline_trajectories(
    cell: LearningCellPools,
    *,
    seed: int,
    trajectories_per_behavior: int,
    loss_scale: float,
) -> list[tuple[list[np.ndarray], list[np.ndarray], list[int], str]]:
    rng = np.random.Generator(np.random.PCG64(seed))
    result: list[tuple[list[np.ndarray], list[np.ndarray], list[int], str]] = []
    policies: dict[tuple[str, str], PolicyNode] = {}
    behaviors = ("passive_oracle", "uniform_valid", "task_active_oracle")
    for behavior in behaviors:
        for episode in range(trajectories_per_behavior):
            environment = VectorConventionEnvironment(
                cell.train,
                seed + episode + len(result) * 997,
                1,
                action_class="task",
                loss_scale=loss_scale,
            )
            batch = environment.reset()
            item = cell.train[int(environment.game_indices[0])]
            node: PolicyNode | None = None
            if behavior != "uniform_valid":
                policy_class = "passive" if behavior == "passive_oracle" else "task"
                key = (item.game.game_id, policy_class)
                if key not in policies:
                    policies[key] = solve(
                        item.game,
                        policy_class,  # type: ignore[arg-type]
                        "net_regret",
                        "float",
                        commitment_states=item.commitment_states,
                    ).policy
                node = policies[key]
            observations: list[np.ndarray] = []
            action_masks: list[np.ndarray] = []
            response_targets: list[int] = []
            while not batch.terminated[0]:
                observations.append(batch.observations[0].copy())
                action_masks.append(batch.action_masks[0].copy())
                response_targets.append(int(batch.response_target[0]))
                if behavior == "uniform_valid":
                    valid_actions = np.flatnonzero(batch.action_masks[0])
                    action_index = int(rng.choice(valid_actions))
                else:
                    if node is None:
                        raise RuntimeError("oracle behavior lost its conditional policy")
                    action_id = f"commit:{node.decision}" if node.kind == "commit" else node.action
                    if action_id is None:
                        raise RuntimeError("oracle policy node has no action")
                    action_index = environment.layout.action_ids.index(action_id)
                step = environment.step(np.asarray([action_index]))
                if node is not None and node.kind == "act":
                    info = step.infos[0]
                    node = next(
                        branch.child
                        for branch in node.branches
                        if branch.next_state == info["next_state"]
                        and branch.observation == info["observation"]
                    )
                batch = step
            result.append((observations, action_masks, response_targets, behavior))
    return result


def _collect_complete_episodes(
    model: LearnedPolicy,
    environment: VectorConventionEnvironment,
    method: LearningMethodSpec,
    target_transitions: int,
    completed_transitions: int,
    total_transitions: int,
    device: torch.device,
) -> tuple[list[EpisodeRecord], int]:
    episodes: list[EpisodeRecord] = []
    transitions = 0
    model.eval()
    while transitions < target_transitions:
        batch = environment.reset()
        hidden = model.initial_hidden(environment.num_envs, device)
        records = [
            EpisodeRecord([], [], [], [], [], [], [], [], []) for _ in range(environment.num_envs)
        ]
        active = np.ones(environment.num_envs, dtype=np.bool_)
        while active.any():
            safe_masks = batch.action_masks.copy()
            safe_masks[~active, 0] = True
            observations = torch.as_tensor(batch.observations, device=device)
            masks = torch.as_tensor(safe_masks, device=device)
            with torch.no_grad():
                if isinstance(model, TomSelectorStylePolicy):
                    model.training_clusters = torch.as_tensor(batch.strategy_cluster, device=device)
                output = model.forward_step(observations, hidden, masks)
                distribution = action_distribution(output)
                sampled_actions = distribution.sample()  # type: ignore[no-untyped-call]
                log_probabilities = distribution.log_prob(  # type: ignore[no-untyped-call]
                    sampled_actions
                )
            actions = sampled_actions.cpu().numpy()
            actions[~active] = -1
            step = environment.step(actions)
            bonus = _pace_bonus(
                model,
                method,
                environment,
                step,
                output.hidden,
                active,
                completed_transitions + transitions,
                total_transitions,
                device,
            )
            for index in np.flatnonzero(active).tolist():
                record = records[index]
                record.observations.append(batch.observations[index].copy())
                record.action_masks.append(batch.action_masks[index].copy())
                record.actions.append(int(actions[index]))
                record.old_log_probabilities.append(float(log_probabilities[index].item()))
                record.values.append(float(output.value[index].item()))
                record.rewards.append(float(step.rewards[index] + bonus[index]))
                record.partner_identities.append(int(batch.partner_identity[index]))
                record.response_targets.append(int(batch.response_target[index]))
                record.strategy_clusters.append(int(batch.strategy_cluster[index]))
                transitions += 1
            newly_done = active & step.terminated
            for index in np.flatnonzero(newly_done).tolist():
                episodes.append(records[index])
            active &= ~newly_done
            hidden = output.hidden
            batch = step
    if isinstance(model, TomSelectorStylePolicy):
        model.training_clusters = None
    return episodes, transitions


def _pace_bonus(
    model: LearnedPolicy,
    method: LearningMethodSpec,
    environment: VectorConventionEnvironment,
    step: StepBatch,
    hidden: torch.Tensor,
    active: np.ndarray,
    completed: int,
    total: int,
    device: torch.device,
) -> np.ndarray:
    result = np.zeros(environment.num_envs, dtype=np.float64)
    if method.method_id not in {"pace_style", "csp_style_reconnaissance"}:
        return result
    interaction = active & ~step.terminated
    if not interaction.any():
        return result
    safe_masks = step.action_masks.copy()
    safe_masks[~interaction, 0] = True
    with torch.no_grad():
        next_output = model.forward_step(
            torch.as_tensor(step.observations, device=device),
            hidden,
            torch.as_tensor(safe_masks, device=device),
        )
        if next_output.identity_logits is None:
            return result
        probabilities = torch.softmax(next_output.identity_logits, dim=-1)
        identities = torch.as_tensor(step.partner_identity, device=device)
        scores = probabilities.gather(1, identities[:, None]).squeeze(1).cpu().numpy()
    fraction = min(1.0, completed / max(1, total))
    decay = max(0.0, 1.0 - fraction / method.config.pace_bonus_decay_fraction)
    result[interaction] = method.config.pace_bonus_initial * decay * scores[interaction]
    return result


def _compute_advantages(episodes: list[EpisodeRecord], method: LearningMethodSpec) -> None:
    for episode in episodes:
        length = len(episode.rewards)
        advantages = np.zeros(length, dtype=np.float32)
        accumulator = 0.0
        next_value = 0.0
        for index in reversed(range(length)):
            delta = (
                episode.rewards[index] + method.config.gamma * next_value - episode.values[index]
            )
            accumulator = delta + method.config.gamma * method.config.gae_lambda * accumulator
            advantages[index] = accumulator
            next_value = episode.values[index]
        episode.advantages = advantages
        episode.returns = advantages + np.asarray(episode.values, dtype=np.float32)
    combined = np.concatenate(
        [episode.advantages for episode in episodes if episode.advantages is not None]
    )
    mean = float(combined.mean())
    standard_deviation = float(combined.std()) + 1e-8
    for episode in episodes:
        if episode.advantages is not None:
            episode.advantages = (episode.advantages - mean) / standard_deviation


def _ppo_update(
    model: LearnedPolicy,
    optimizer: torch.optim.Optimizer,
    episodes: list[EpisodeRecord],
    method: LearningMethodSpec,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    order_rng = np.random.default_rng(sum(len(item.actions) for item in episodes))
    aggregate: dict[str, list[float]] = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "auxiliary_loss": [],
    }
    episodes_per_minibatch = max(
        1,
        method.config.minibatch_size
        // max(1, int(np.mean([len(item.actions) for item in episodes]))),
    )
    for _ in range(method.config.optimization_epochs):
        indices = order_rng.permutation(len(episodes))
        for start in range(0, len(indices), episodes_per_minibatch):
            selected = [
                episodes[index] for index in indices[start : start + episodes_per_minibatch]
            ]
            batch = _padded_batch(selected, device)
            hidden = model.initial_hidden(len(selected), device)
            outputs = []
            for time in range(batch["observations"].shape[1]):
                if isinstance(model, TomSelectorStylePolicy):
                    model.training_clusters = batch["strategy_clusters"][:, time]
                output = model.forward_step(
                    batch["observations"][:, time], hidden, batch["action_masks"][:, time]
                )
                hidden = output.hidden
                outputs.append(output)
            logits = torch.stack([item.logits for item in outputs], dim=1)
            values = torch.stack([item.value for item in outputs], dim=1)
            valid = batch["valid"]
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probabilities = distribution.log_prob(  # type: ignore[no-untyped-call]
                batch["actions"]
            )
            ratios = torch.exp(new_log_probabilities[valid] - batch["old_log_probabilities"][valid])
            advantages = batch["advantages"][valid]
            unclipped = ratios * advantages
            clipped = (
                torch.clamp(
                    ratios,
                    1 - method.config.clip_ratio,
                    1 + method.config.clip_ratio,
                )
                * advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = functional.mse_loss(values[valid], batch["returns"][valid])
            entropy = distribution.entropy()[valid].mean()  # type: ignore[no-untyped-call]
            auxiliary_loss = _auxiliary_loss(model, outputs, batch, method)
            loss = (
                policy_loss
                + method.config.value_coefficient * value_loss
                - method.config.entropy_coefficient * entropy
                + method.config.auxiliary_coefficient * auxiliary_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), method.config.max_gradient_norm)
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("policy_loss", policy_loss),
                ("value_loss", value_loss),
                ("entropy", entropy),
                ("auxiliary_loss", auxiliary_loss),
            ):
                aggregate[key].append(float(value.detach().item()))
    if isinstance(model, TomSelectorStylePolicy):
        model.training_clusters = None
    return {key: float(np.mean(values)) for key, values in aggregate.items()}


def _auxiliary_loss(
    model: LearnedPolicy,
    outputs: list[Any],
    batch: dict[str, torch.Tensor],
    method: LearningMethodSpec,
) -> torch.Tensor:
    valid = batch["valid"]
    losses: list[torch.Tensor] = []
    identity = outputs[0].identity_logits
    if identity is not None:
        logits = torch.stack([item.identity_logits for item in outputs], dim=1)
        losses.append(functional.cross_entropy(logits[valid], batch["partner_identities"][valid]))
    response = outputs[0].response_logits
    if response is not None:
        logits = torch.stack([item.response_logits for item in outputs], dim=1)
        response_valid = valid & (batch["response_targets"] >= 0)
        if response_valid.any():
            losses.append(
                functional.cross_entropy(
                    logits[response_valid], batch["response_targets"][response_valid]
                )
            )
    if (
        outputs[0].latent_mean is not None
        and outputs[0].latent_log_variance is not None
        and method.method_id != "odits_style"
    ):
        means = torch.stack([item.latent_mean for item in outputs], dim=1)[valid]
        log_variances = torch.stack([item.latent_log_variance for item in outputs], dim=1)[valid]
        kl = -0.5 * (1 + log_variances - means.square() - log_variances.exp()).mean()
        losses.append(method.config.kl_coefficient * kl)
    sequence_loss = model.sequence_auxiliary(
        batch["observations"], valid, batch["response_targets"], outputs
    )
    if sequence_loss is not None:
        losses.append(method.config.kl_coefficient * sequence_loss)
    if method.method_id == "tom_selector_style" and outputs[0].cluster_response_logits is not None:
        cluster_predictions = torch.stack([item.cluster_response_logits for item in outputs], dim=1)
        response_valid = valid & (batch["response_targets"] >= 0)
        if response_valid.any():
            batch_indices, time_indices = torch.where(response_valid)
            cluster_indices = batch["strategy_clusters"][response_valid]
            selected = cluster_predictions[batch_indices, time_indices, cluster_indices]
            losses.append(
                functional.cross_entropy(selected, batch["response_targets"][response_valid])
            )
    if not losses:
        return torch.zeros((), device=batch["observations"].device)
    return torch.stack(losses).sum()


def _padded_batch(episodes: list[EpisodeRecord], device: torch.device) -> dict[str, torch.Tensor]:
    batch_size = len(episodes)
    maximum = max(len(item.actions) for item in episodes)
    observation_size = episodes[0].observations[0].shape[0]
    action_size = episodes[0].action_masks[0].shape[0]
    observations = np.zeros((batch_size, maximum, observation_size), dtype=np.float32)
    action_masks = np.zeros((batch_size, maximum, action_size), dtype=np.bool_)
    action_masks[:, :, 0] = True
    actions = np.zeros((batch_size, maximum), dtype=np.int64)
    old_log = np.zeros((batch_size, maximum), dtype=np.float32)
    advantages = np.zeros((batch_size, maximum), dtype=np.float32)
    returns = np.zeros((batch_size, maximum), dtype=np.float32)
    identities = np.zeros((batch_size, maximum), dtype=np.int64)
    responses = np.full((batch_size, maximum), -1, dtype=np.int64)
    clusters = np.zeros((batch_size, maximum), dtype=np.int64)
    valid = np.zeros((batch_size, maximum), dtype=np.bool_)
    for index, episode in enumerate(episodes):
        length = len(episode.actions)
        observations[index, :length] = episode.observations
        action_masks[index, :length] = episode.action_masks
        actions[index, :length] = episode.actions
        old_log[index, :length] = episode.old_log_probabilities
        if episode.advantages is None or episode.returns is None:
            raise RuntimeError("advantages must be computed before PPO update")
        advantages[index, :length] = episode.advantages
        returns[index, :length] = episode.returns
        identities[index, :length] = episode.partner_identities
        responses[index, :length] = episode.response_targets
        clusters[index, :length] = episode.strategy_clusters
        valid[index, :length] = True
    return {
        "observations": torch.as_tensor(observations, device=device),
        "action_masks": torch.as_tensor(action_masks, device=device),
        "actions": torch.as_tensor(actions, device=device),
        "old_log_probabilities": torch.as_tensor(old_log, device=device),
        "advantages": torch.as_tensor(advantages, device=device),
        "returns": torch.as_tensor(returns, device=device),
        "partner_identities": torch.as_tensor(identities, device=device),
        "response_targets": torch.as_tensor(responses, device=device),
        "strategy_clusters": torch.as_tensor(clusters, device=device),
        "valid": torch.as_tensor(valid, device=device),
    }


def _save_checkpoint(
    path: Path,
    model: LearnedPolicy,
    optimizer: torch.optim.Optimizer,
    suite: LearningAuditSuite,
    method: LearningMethodSpec,
    cell: LearningCellPools,
    environment: VectorConventionEnvironment,
    seed: int,
    transitions: int,
    *,
    best_key: tuple[float, float, int] | None,
    best_state: dict[str, torch.Tensor] | None,
    best_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "suite_id": suite.suite_id,
            "cell_id": cell.cell_id,
            "method": method.model_dump(mode="json"),
            "seed": seed,
            "transitions": transitions,
            "model_metadata": model_metadata(model),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "observation_size": environment.layout.observation_size,
            "action_size": environment.layout.action_size,
            "partner_identity_count": environment.partner_identity_count,
            "partner_labels": environment.partner_labels,
            "partner_response_classes": environment.partner_response_classes,
            "partner_strategy_clusters": environment.partner_strategy_clusters,
            "response_count": environment.response_target_count,
            "environment_state": environment.state_dict(),
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.random.get_rng_state(),
            "configuration_hash": _configuration_hash(suite, method),
            "pretraining_metadata": dict(getattr(model, "pretraining_metadata", {})),
            "selection_key": best_key,
            "selection_state": best_state,
            "selection_step": best_step,
        },
        path,
    )


def _configuration_hash(suite: LearningAuditSuite, method: LearningMethodSpec) -> str:
    payload = json.dumps(
        {"suite": suite.model_dump(mode="json"), "method": method.model_dump(mode="json")},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _source_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "src").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("torch", "numpy", "pydantic", "pandas", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))


def _select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if name not in {"cpu", "mps"}:
        raise ValueError(f"unsupported learning device: {name!r}")
    return torch.device(name)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
