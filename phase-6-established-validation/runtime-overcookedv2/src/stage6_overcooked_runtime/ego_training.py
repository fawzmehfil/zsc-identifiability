"""Single learned ego PPO against a frozen OvercookedV2 partner population."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from overcooked_v2_experiments.ppo.models.model import get_actor_critic, initialize_carry

from stage6_overcooked_runtime.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
    validate_resume_target,
)
from stage6_overcooked_runtime.ported_methods import (
    PACE_AUXILIARY_WEIGHT,
    CSPTrajectoryModel,
    PaceActorCritic,
    csp_probe_reward,
)
from stage6_overcooked_runtime.resumable_upstream import pinned_ippo_source_hash


class EgoTransition(NamedTuple):
    done: jnp.ndarray
    observation: jnp.ndarray
    previous_action: jnp.ndarray
    previous_reward: jnp.ndarray
    action: jnp.ndarray
    log_probability: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    external_task_reward: jnp.ndarray
    partner_index: jnp.ndarray
    partner_action_target: jnp.ndarray


class EgoRunnerState(NamedTuple):
    train_state: TrainState
    environment_state: object
    observations_0: jnp.ndarray
    observations_1: jnp.ndarray
    done: jnp.ndarray
    ego_hidden: jnp.ndarray
    partner_hidden: jnp.ndarray
    partner_indices: jnp.ndarray
    previous_actions: jnp.ndarray
    previous_rewards: jnp.ndarray
    auxiliary_train_state: TrainState
    auxiliary_hidden: jnp.ndarray
    auxiliary_prediction_logits: jnp.ndarray
    auxiliary_prediction_valid: jnp.ndarray
    rng: jnp.ndarray
    completed_transitions: jnp.ndarray


def train_ego_method(request, project_root, *, method_id=None, component_id="task_policy"):
    """Train ordinary or PACE ego control with frozen partners and exact resume."""

    payload = request["payload"]
    method = payload["method_id"] if method_id is None else method_id
    if method not in {"pace_aux", "pace_style", "tbs_style", "csp_style_reconnaissance"}:
        raise ValueError(f"unsupported frozen-partner ego method: {method!r}")
    train_pool = _load_pool(payload["train_pool_path"])
    if not train_pool:
        raise ValueError("frozen-partner training pool is empty")
    validation_pool = _load_pool(payload["validation_pool_path"])
    if not validation_pool:
        raise ValueError("frozen-partner validation pool is empty")
    upstream = Path(project_root) / request["upstreams"]["overcookedv2"]["path"]
    config = _compose_config(payload, upstream)
    partner_params, partner_config = _load_partner_population(train_pool)
    validation_params, validation_config = _load_partner_population(validation_pool)
    pace = method in {"pace_aux", "pace_style"}
    network = (
        PaceActorCritic(action_dim=6, partner_count=len(train_pool))
        if pace
        else get_actor_critic(config)
    )
    output = Path(payload["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration_hash = _configuration_hash(config, method, len(train_pool), component_id)
    dataset_hashes = tuple(payload.get("dataset_hashes", ()))
    requested_target = int(payload["transitions"])
    step_size = int(config["model"]["NUM_STEPS"]) * int(config["model"]["NUM_ENVS"])
    target = (requested_target // step_size) * step_size
    if target <= 0:
        raise ValueError("requested target must contain at least one complete PPO update")
    stop_after = int(payload.get("stop_after_transitions", target))
    execution_target = min(target, (stop_after // step_size) * step_size)
    if execution_target <= 0:
        raise ValueError("execution stop must contain at least one complete PPO update")
    runner, policy_config = _build_runner(
        config,
        network,
        partner_params,
        partner_config,
        method=method,
        seed=int(payload["seed"]),
        target_transitions=target,
        initial_state=None,
    )
    parent_hash = None
    best_validation_metric = float("-inf")
    resumed = payload.get("resume_checkpoint") is not None
    if resumed:
        runner, metadata, _, parent_hash = restore_training_checkpoint(
            payload["resume_checkpoint"],
            expected={
                "suite_id": request["suite_id"],
                "method_id": method,
                "layout_id": payload["layout_id"],
                "seed": int(payload["seed"]),
                "component_id": component_id,
                "configuration_hash": configuration_hash,
                "dataset_hashes": dataset_hashes,
                "upstream_commit": request["upstreams"]["overcookedv2"]["commit"],
                "upstream_source_hash": pinned_ippo_source_hash(),
            },
            target_state=runner,
        )
        validate_resume_target(metadata, target)
        best_validation_metric = float(metadata.get("best_validation_metric", float("-inf")))
    validation_evaluator = _make_validation_evaluator(
        config,
        network,
        validation_params,
        validation_config,
        method=method,
        seed=int(payload["seed"]) + 91_000,
    )
    checkpoint_interval = int(payload.get("checkpoint_interval", 1_000_000))
    last_boundary = int(np.asarray(jax.device_get(runner.completed_transitions)))
    latest_checkpoint = None
    latest_checkpoint_hash = parent_hash
    metrics = {}
    update = jax.jit(
        lambda state: _run_one_update(
            state,
            config,
            network,
            partner_params,
            partner_config,
            method,
            target,
        )
    )
    while last_boundary < execution_target:
        runner, update_metrics = update(runner)
        metrics = {key: float(np.asarray(value)) for key, value in update_metrics.items()}
        completed = int(np.asarray(jax.device_get(runner.completed_transitions)))
        crossed = completed // checkpoint_interval > last_boundary // checkpoint_interval
        if completed >= execution_target or crossed:
            validation_metric = float(
                np.asarray(jax.device_get(validation_evaluator(runner.train_state.params)))
            )
            is_best = validation_metric > best_validation_metric + 1e-12
            best_validation_metric = max(best_validation_metric, validation_metric)
            metadata = {
                "suite_id": request["suite_id"],
                "method_id": method,
                "layout_id": payload["layout_id"],
                "seed": int(payload["seed"]),
                "component_id": component_id,
                "completed_transitions": completed,
                "target_transitions": target,
                "configuration_hash": configuration_hash,
                "dataset_hashes": dataset_hashes,
                "upstream_commit": request["upstreams"]["overcookedv2"]["commit"],
                "upstream_source_hash": pinned_ippo_source_hash(),
                "device": _device_description(),
                "parent_checkpoint_hash": latest_checkpoint_hash,
                "exact_continuation": all(device.platform == "cpu" for device in jax.devices()),
                "validation_metric": validation_metric,
                "best_validation_metric": best_validation_metric,
            }
            latest_checkpoint, latest_checkpoint_hash = save_training_checkpoint(
                output / "training-state", runner, metadata, is_best=is_best
            )
            _save_compact_policy(
                output / "run_0" / f"ckpt_{completed}",
                runner.train_state.params,
                policy_config,
            )
        last_boundary = completed
    policy_path = output / "run_0" / "ckpt_final"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    _save_compact_policy(policy_path, runner.train_state.params, policy_config)
    checkpointer = ocp.PyTreeCheckpointer()
    auxiliary_path = None
    if method == "csp_style_reconnaissance":
        auxiliary_path = output / "auxiliary" / "trajectory-model-final"
        auxiliary_payload = {
            "params": runner.auxiliary_train_state.params,
            "config": {
                "model": "csp_trajectory_model",
                "action_dim": 7,
                "hidden_size": 128,
                "latent_size": 32,
            },
        }
        if auxiliary_path.exists():
            import shutil

            shutil.rmtree(auxiliary_path)
        auxiliary_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer.save(
            auxiliary_path,
            auxiliary_payload,
            save_args=orbax_utils.save_args_from_target(auxiliary_payload),
        )
    parameter_hash = _tree_hash(runner.train_state.params)
    best_index = json.loads((output / "training-state" / "best.json").read_text())
    return {
        "method_id": method,
        "layout_id": payload["layout_id"],
        "seed": int(payload["seed"]),
        "requested_transitions": requested_target,
        "completed_transitions": int(np.asarray(runner.completed_transitions)),
        "checkpoint_paths": [str(policy_path)],
        "checkpoint_parameter_hashes": {str(policy_path): parameter_hash},
        "training_state_paths": [] if latest_checkpoint is None else [str(latest_checkpoint)],
        "training_state_hashes": (
            {} if latest_checkpoint is None else {str(latest_checkpoint): latest_checkpoint_hash}
        ),
        "best_validation_checkpoint_path": best_index["checkpoint_path"],
        "best_validation_checkpoint_hash": best_index["checkpoint_hash"],
        "best_validation_metric": best_index["validation_metric"],
        "configuration_hash": configuration_hash,
        "device": _device_description(),
        "resumed": resumed,
        "resume_checkpoint_path": payload.get("resume_checkpoint"),
        "parent_checkpoint_hash": parent_hash,
        "policy_kind": "pace" if pace else "ppo",
        "component_transitions": {component_id: int(np.asarray(runner.completed_transitions))},
        "aggregate_training_transitions": int(np.asarray(runner.completed_transitions)),
        "metrics": metrics,
        "partner_ids": [item["partner_id"] for item in train_pool],
        "policy_config": policy_config,
        "auxiliary_checkpoint_path": (None if auxiliary_path is None else str(auxiliary_path)),
    }


def _build_runner(
    config,
    network,
    partner_params,
    partner_config,
    *,
    method,
    seed,
    target_transitions,
    initial_state,
):
    if initial_state is not None:
        policy_config = _policy_config(config, method, partner_params)
        return initial_state, policy_config
    import jaxmarl

    environment = jaxmarl.make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
    num_envs = int(config["model"]["NUM_ENVS"])
    rng = jax.random.PRNGKey(seed)
    rng, reset_key, model_key = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_key, num_envs)
    observations, environment_state = jax.vmap(environment.reset)(reset_keys)
    observation_shape = environment.observation_space().shape
    ego_hidden = initialize_carry(config, num_envs)
    partner_hidden = initialize_carry(partner_config, num_envs)
    init_observation = jnp.zeros((1, num_envs, *observation_shape), dtype=jnp.float32)
    init_done = jnp.zeros((1, num_envs), dtype=jnp.bool_)
    if method in {"pace_aux", "pace_style"}:
        params = network.init(
            model_key,
            ego_hidden,
            init_observation,
            jnp.zeros((1, num_envs), dtype=jnp.int32),
            init_done,
            method=network.with_identity,
        )
    else:
        params = network.init(model_key, ego_hidden, (init_observation, init_done))
    steps_per_update = int(config["model"]["NUM_STEPS"]) * num_envs
    update_count = max(target_transitions // steps_per_update, 1)
    gradient_steps = update_count * int(config["model"]["UPDATE_EPOCHS"])
    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=float(config["model"]["LR"]),
        warmup_steps=max(int(0.05 * gradient_steps), 1),
        decay_steps=max(gradient_steps, 2),
        end_value=0.0,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(config["model"]["MAX_GRAD_NORM"])),
        optax.adam(learning_rate, eps=1e-5),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=optimizer)
    auxiliary_network = CSPTrajectoryModel()
    auxiliary_hidden = initialize_carry(config, num_envs)
    auxiliary_params = auxiliary_network.init(
        jax.random.fold_in(model_key, 91),
        auxiliary_hidden,
        init_observation,
        jnp.zeros((1, num_envs), dtype=jnp.int32),
        jnp.zeros((1, num_envs), dtype=jnp.float32),
        init_done,
    )
    auxiliary_path = config.get("STAGE6_CSP_MODEL_PATH")
    if auxiliary_path:
        auxiliary_payload = ocp.PyTreeCheckpointer().restore(
            Path(auxiliary_path).resolve(), item=None
        )
        auxiliary_params = auxiliary_payload["params"]
    auxiliary_state = TrainState.create(
        apply_fn=auxiliary_network.apply,
        params=auxiliary_params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(3e-4)),
    )
    rng, partner_key = jax.random.split(rng)
    partner_indices = jax.random.randint(
        partner_key, (num_envs,), 0, _population_size(partner_params)
    )
    runner = EgoRunnerState(
        train_state=train_state,
        environment_state=environment_state,
        observations_0=observations["agent_0"],
        observations_1=observations["agent_1"],
        done=jnp.zeros((num_envs,), dtype=jnp.bool_),
        ego_hidden=ego_hidden,
        partner_hidden=partner_hidden,
        partner_indices=partner_indices,
        previous_actions=jnp.zeros((num_envs,), dtype=jnp.int32),
        previous_rewards=jnp.zeros((num_envs,), dtype=jnp.float32),
        auxiliary_train_state=auxiliary_state,
        auxiliary_hidden=auxiliary_hidden,
        auxiliary_prediction_logits=jnp.zeros((num_envs, 7), dtype=jnp.float32),
        auxiliary_prediction_valid=jnp.zeros((num_envs,), dtype=jnp.bool_),
        rng=rng,
        completed_transitions=jnp.asarray(0, dtype=jnp.int32),
    )
    return runner, _policy_config(config, method, partner_params)


def _run_one_update(
    runner,
    config,
    network,
    partner_params,
    partner_config,
    method,
    target_transitions,
):
    import jaxmarl

    environment = jaxmarl.make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
    num_envs = int(config["model"]["NUM_ENVS"])
    num_steps = int(config["model"]["NUM_STEPS"])
    ego_is_zero = jnp.arange(num_envs) % 2 == 0
    partner_network = get_actor_critic(partner_config)
    auxiliary_network = CSPTrajectoryModel()
    initial_hidden = runner.ego_hidden

    def rollout_step(state, _):
        ego_obs = _choose(ego_is_zero, state.observations_0, state.observations_1)
        partner_obs = _choose(ego_is_zero, state.observations_1, state.observations_0)
        state_rng, ego_key, partner_key, environment_key, resample_key = jax.random.split(
            state.rng, 5
        )
        if method in {"pace_aux", "pace_style"}:
            ego_hidden, policy, value, _ = network.apply(
                state.train_state.params,
                state.ego_hidden,
                ego_obs[None, ...],
                state.previous_actions[None, ...],
                state.done[None, ...],
                method=network.with_identity,
            )
        else:
            ego_hidden, policy, value = network.apply(
                state.train_state.params,
                state.ego_hidden,
                (ego_obs[None, ...], state.done[None, ...]),
            )
        ego_action = policy.sample(seed=ego_key)[0]
        log_probability = policy.log_prob(ego_action[None, ...])[0]
        selected_params = jax.tree_util.tree_map(
            lambda values: values[state.partner_indices], partner_params
        )
        partner_keys = jax.random.split(partner_key, num_envs)

        def partner_action(params, hidden, observation, done, key):
            next_hidden, distribution, _ = partner_network.apply(
                params,
                hidden[None, ...],
                (observation[None, None, ...], done[None, None]),
            )
            return next_hidden[0], distribution.sample(seed=key)[0, 0]

        partner_hidden, partner_action_values = jax.vmap(partner_action)(
            selected_params,
            state.partner_hidden,
            partner_obs,
            state.done,
            partner_keys,
        )
        positions = state.environment_state.agents.pos.to_array()
        partner_visible = jnp.max(jnp.abs(positions[:, 0] - positions[:, 1]), axis=-1) <= 2
        partner_action_target = jnp.where(partner_visible, partner_action_values, 6)
        action_0 = jnp.where(ego_is_zero, ego_action, partner_action_values)
        action_1 = jnp.where(ego_is_zero, partner_action_values, ego_action)
        environment_keys = jax.random.split(environment_key, num_envs)
        observations, environment_state, rewards, dones, info = jax.vmap(environment.step)(
            environment_keys,
            state.environment_state,
            {"agent_0": action_0, "agent_1": action_1},
        )
        sparse = _choose(ego_is_zero, rewards["agent_0"], rewards["agent_1"])
        shaped = _choose(
            ego_is_zero,
            info["shaped_reward"]["agent_0"],
            info["shaped_reward"]["agent_1"],
        )
        fraction = jnp.minimum(
            state.completed_transitions / max(target_transitions / 2.0, 1.0), 1.0
        )
        external_task_reward = sparse + shaped * (1.0 - fraction)
        task_reward = external_task_reward
        done_all = dones["__all__"]
        if method == "pace_style":
            beta = pace_bonus_weight_array(state.completed_transitions, target_transitions)
            next_ego_observation = _choose(
                ego_is_zero, observations["agent_0"], observations["agent_1"]
            )
            _, _, _, next_identity_logits = network.apply(
                state.train_state.params,
                ego_hidden,
                next_ego_observation[None, ...],
                ego_action[None, ...],
                done_all[None, ...],
                method=network.with_identity,
            )
            probabilities = jax.nn.softmax(next_identity_logits[0], axis=-1)
            peer_reward = jnp.take_along_axis(
                probabilities, state.partner_indices[:, None], axis=-1
            )[:, 0]
            task_reward = task_reward + beta * jax.lax.stop_gradient(peer_reward) * (~done_all)
        if method == "csp_style_reconnaissance":
            prediction_loss = optax.softmax_cross_entropy_with_integer_labels(
                state.auxiliary_prediction_logits, partner_action_target
            )
            probe_reward = csp_probe_reward(task_reward, jax.lax.stop_gradient(prediction_loss))
            task_reward = jnp.where(state.auxiliary_prediction_valid, probe_reward, task_reward)
        auxiliary_hidden, _, auxiliary_logits = auxiliary_network.apply(
            state.auxiliary_train_state.params,
            state.auxiliary_hidden,
            ego_obs[None, ...],
            ego_action[None, ...],
            external_task_reward[None, ...],
            state.done[None, ...],
        )
        sampled_indices = jax.random.randint(
            resample_key, (num_envs,), 0, _population_size(partner_params)
        )
        next_indices = jnp.where(done_all, sampled_indices, state.partner_indices)
        transition = EgoTransition(
            done=done_all,
            observation=ego_obs,
            previous_action=state.previous_actions,
            previous_reward=state.previous_rewards,
            action=ego_action,
            log_probability=log_probability,
            value=value[0],
            reward=task_reward,
            external_task_reward=external_task_reward,
            partner_index=state.partner_indices,
            partner_action_target=partner_action_target,
        )
        next_state = state._replace(
            environment_state=environment_state,
            observations_0=observations["agent_0"],
            observations_1=observations["agent_1"],
            done=done_all,
            ego_hidden=ego_hidden,
            partner_hidden=partner_hidden,
            partner_indices=next_indices,
            previous_actions=jnp.where(done_all, 0, ego_action),
            previous_rewards=jnp.where(done_all, 0.0, task_reward),
            auxiliary_hidden=auxiliary_hidden,
            auxiliary_prediction_logits=auxiliary_logits[0],
            auxiliary_prediction_valid=~done_all,
            rng=state_rng,
            completed_transitions=state.completed_transitions + num_envs,
        )
        return next_state, transition

    runner, trajectory = jax.lax.scan(rollout_step, runner, None, num_steps)
    last_obs = _choose(ego_is_zero, runner.observations_0, runner.observations_1)
    if method in {"pace_aux", "pace_style"}:
        _, _, last_value, _ = network.apply(
            runner.train_state.params,
            runner.ego_hidden,
            last_obs[None, ...],
            runner.previous_actions[None, ...],
            runner.done[None, ...],
            method=network.with_identity,
        )
    else:
        _, _, last_value = network.apply(
            runner.train_state.params,
            runner.ego_hidden,
            (last_obs[None, ...], runner.done[None, ...]),
        )
    advantages, targets = _gae(
        trajectory,
        last_value[0],
        gamma=float(config["model"]["GAMMA"]),
        gae_lambda=float(config["model"]["GAE_LAMBDA"]),
    )

    def loss(params):
        if method in {"pace_aux", "pace_style"}:
            _, distribution, values, identity_logits = network.apply(
                params,
                initial_hidden,
                trajectory.observation,
                trajectory.previous_action,
                trajectory.done,
                method=network.with_identity,
            )
            identity_loss = optax.softmax_cross_entropy_with_integer_labels(
                identity_logits, trajectory.partner_index
            ).mean()
        else:
            _, distribution, values = network.apply(
                params,
                initial_hidden,
                (trajectory.observation, trajectory.done),
            )
            identity_loss = jnp.asarray(0.0)
        log_probability = distribution.log_prob(trajectory.action)
        ratio = jnp.exp(log_probability - trajectory.log_probability)
        normalized_advantage = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        unclipped = ratio * normalized_advantage
        clipped = (
            jnp.clip(
                ratio,
                1.0 - float(config["model"]["CLIP_EPS"]),
                1.0 + float(config["model"]["CLIP_EPS"]),
            )
            * normalized_advantage
        )
        actor_loss = -jnp.minimum(unclipped, clipped).mean()
        value_loss = 0.5 * jnp.square(values - targets).mean()
        entropy = distribution.entropy().mean()
        task_loss = (
            actor_loss
            + float(config["model"]["VF_COEF"]) * value_loss
            - float(config["model"]["ENT_COEF"]) * entropy
        )
        warmup = (
            pace_warmup_array(
                runner.completed_transitions - num_steps * num_envs,
                target_transitions,
            )
            if method in {"pace_aux", "pace_style"}
            else jnp.asarray(False)
        )
        total = jnp.where(warmup, 0.0, task_loss) + PACE_AUXILIARY_WEIGHT * identity_loss
        return total, {
            "loss": total,
            "actor_loss": actor_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "identity_loss": identity_loss,
            "mean_reward": trajectory.reward.mean(),
        }

    train_state = runner.train_state
    auxiliary_train_state = runner.auxiliary_train_state
    metrics = {}
    for _ in range(int(config["model"]["UPDATE_EPOCHS"])):
        (loss_value, metrics), gradients = jax.value_and_grad(loss, has_aux=True)(
            train_state.params
        )
        del loss_value
        train_state = train_state.apply_gradients(grads=gradients)
    if method == "csp_style_reconnaissance":

        def auxiliary_loss(parameters):
            auxiliary_initial = initialize_carry(config, num_envs)
            _, _, logits = auxiliary_network.apply(
                parameters,
                auxiliary_initial,
                trajectory.observation,
                trajectory.action,
                trajectory.external_task_reward,
                trajectory.done,
            )
            element = optax.softmax_cross_entropy_with_integer_labels(
                logits[:-1], trajectory.partner_action_target[1:]
            )
            valid = (~trajectory.done[:-1]).astype(jnp.float32)
            return (element * valid).sum() / jnp.maximum(valid.sum(), 1.0)

        prediction_loss, gradients = jax.value_and_grad(auxiliary_loss)(
            auxiliary_train_state.params
        )
        auxiliary_train_state = auxiliary_train_state.apply_gradients(grads=gradients)
        metrics = {**metrics, "csp_prediction_loss": prediction_loss}
    return runner._replace(
        train_state=train_state,
        auxiliary_train_state=auxiliary_train_state,
    ), metrics


def _make_validation_evaluator(
    config,
    network,
    partner_params,
    partner_config,
    *,
    method,
    seed,
):
    """Build a fixed-key, external-return evaluator for checkpoint selection."""

    import jaxmarl

    environment = jaxmarl.make(config["env"]["ENV_NAME"], **config["env"]["ENV_KWARGS"])
    partner_network = get_actor_critic(partner_config)
    population_size = _population_size(partner_params)
    num_envs = max(population_size, 2)
    partner_indices = jnp.arange(num_envs, dtype=jnp.int32) % population_size
    selected_params = jax.tree_util.tree_map(lambda values: values[partner_indices], partner_params)
    ego_is_zero = jnp.arange(num_envs) % 2 == 0
    root_key = jax.random.PRNGKey(seed)
    root_key, reset_key = jax.random.split(root_key)
    reset_keys = jax.random.split(reset_key, num_envs)
    initial_observations, initial_environment_state = jax.vmap(environment.reset)(reset_keys)
    initial_ego_hidden = initialize_carry(config, num_envs)
    initial_partner_hidden = initialize_carry(partner_config, num_envs)

    def evaluate(params):
        initial = (
            initial_environment_state,
            initial_observations["agent_0"],
            initial_observations["agent_1"],
            jnp.zeros((num_envs,), dtype=jnp.bool_),
            initial_ego_hidden,
            initial_partner_hidden,
            jnp.zeros((num_envs,), dtype=jnp.int32),
            root_key,
            jnp.zeros((num_envs,), dtype=jnp.float32),
        )

        def step(state, _):
            (
                environment_state,
                observations_0,
                observations_1,
                done,
                ego_hidden,
                partner_hidden,
                previous_actions,
                rng,
                returns,
            ) = state
            ego_observation = _choose(ego_is_zero, observations_0, observations_1)
            partner_observation = _choose(ego_is_zero, observations_1, observations_0)
            rng, environment_key = jax.random.split(rng)
            if method in {"pace_aux", "pace_style"}:
                ego_hidden, distribution, _, _ = network.apply(
                    params,
                    ego_hidden,
                    ego_observation[None, ...],
                    previous_actions[None, ...],
                    done[None, ...],
                    method=network.with_identity,
                )
            else:
                ego_hidden, distribution, _ = network.apply(
                    params,
                    ego_hidden,
                    (ego_observation[None, ...], done[None, ...]),
                )
            ego_action = jnp.argmax(distribution.probs[0], axis=-1)

            def partner_action(policy_params, hidden, observation, reset):
                next_hidden, partner_distribution, _ = partner_network.apply(
                    policy_params,
                    hidden[None, ...],
                    (observation[None, None, ...], reset[None, None]),
                )
                return next_hidden[0], jnp.argmax(partner_distribution.probs[0, 0])

            partner_hidden, partner_actions = jax.vmap(partner_action)(
                selected_params,
                partner_hidden,
                partner_observation,
                done,
            )
            action_0 = jnp.where(ego_is_zero, ego_action, partner_actions)
            action_1 = jnp.where(ego_is_zero, partner_actions, ego_action)
            environment_keys = jax.random.split(environment_key, num_envs)
            observations, environment_state, rewards, dones, _ = jax.vmap(environment.step)(
                environment_keys,
                environment_state,
                {"agent_0": action_0, "agent_1": action_1},
            )
            sparse = _choose(ego_is_zero, rewards["agent_0"], rewards["agent_1"])
            done_all = dones["__all__"]
            return (
                environment_state,
                observations["agent_0"],
                observations["agent_1"],
                done_all,
                ego_hidden,
                partner_hidden,
                jnp.where(done_all, 0, ego_action),
                rng,
                returns + sparse,
            ), None

        final, _ = jax.lax.scan(step, initial, None, 400)
        return final[-1].mean()

    return jax.jit(evaluate)


def _gae(trajectory, last_value, *, gamma, gae_lambda):
    def step(carry, transition):
        advantage, next_value = carry
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        advantage = delta + gamma * gae_lambda * (1 - transition.done) * advantage
        return (advantage, transition.value), advantage

    _, advantages = jax.lax.scan(
        step,
        (jnp.zeros_like(last_value), last_value),
        trajectory,
        reverse=True,
    )
    return advantages, advantages + trajectory.value


def _choose(mask, first, second):
    expanded = mask.reshape((mask.shape[0],) + (1,) * (first.ndim - 1))
    return jnp.where(expanded, first, second)


def pace_bonus_weight_array(completed, target):
    end = max(5.0 * target / 6.0, 1.0)
    return 0.2 * (1.0 - jnp.minimum(completed / end, 1.0))


def pace_warmup_array(completed, target):
    return completed < target / 30.0


def _load_pool(path):
    payload = json.loads(Path(path).read_text())
    checkpoints = payload.get("checkpoints", payload if isinstance(payload, list) else [])
    return tuple(item for item in checkpoints if item.get("competent", True))


def _load_partner_population(pool):
    payloads = [
        ocp.PyTreeCheckpointer().restore(Path(item["checkpoint_path"]).resolve(), item=None)
        for item in pool
    ]
    configs = [item["config"] for item in payloads]
    reference = json.dumps(configs[0], sort_keys=True, default=str)
    if any(json.dumps(item, sort_keys=True, default=str) != reference for item in configs[1:]):
        raise ValueError("frozen partner checkpoints do not share one policy architecture")
    params = jax.tree_util.tree_map(
        lambda *values: jnp.stack(values),
        *[item["params"] for item in payloads],
    )
    return params, configs[0]


def _compose_config(payload, upstream):
    config_dir = upstream / "experiments/overcooked_v2_experiments/ppo/config"
    overrides = [
        "+experiment=rnn-sp",
        f"+env={payload['layout_id']}",
        f"SEED={int(payload['seed'])}",
        "NUM_SEEDS=1",
        f"model.TOTAL_TIMESTEPS={int(payload['transitions'])}",
        f"model.REW_SHAPING_HORIZON={int(payload['transitions']) // 2}",
        f"model.LR={float(payload['learning_rate'])}",
        f"model.ENT_COEF={float(payload['entropy_coefficient'])}",
    ]
    if bool(payload.get("smoke", False)):
        overrides.extend(("model.NUM_ENVS=32", "model.NUM_STEPS=32", "model.UPDATE_EPOCHS=2"))
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = OmegaConf.to_container(compose(config_name="base", overrides=overrides))
    if payload.get("csp_model_path"):
        config["STAGE6_CSP_MODEL_PATH"] = payload["csp_model_path"]
    return config


def _configuration_hash(config, method, partner_count, component_id):
    payload = {
        "config": copy.deepcopy(config),
        "method": method,
        "partner_count": partner_count,
        "component_id": component_id,
        "runtime_source_hash": _training_source_hash(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _training_source_hash():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("checkpointing.py", "ego_training.py", "ported_methods.py"):
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _policy_config(config, method, partner_params):
    if method in {"pace_aux", "pace_style"}:
        return {
            "policy_kind": "pace",
            "action_dim": 6,
            "partner_count": _population_size(partner_params),
            "hidden_size": 128,
        }
    return config


def _population_size(params):
    return int(jax.tree_util.tree_leaves(params)[0].shape[0])


def _tree_hash(params):
    digest = hashlib.sha256()
    for value in jax.tree_util.tree_leaves(params):
        array = np.asarray(jax.device_get(value))
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _save_compact_policy(path, params, config):
    import shutil

    destination = Path(path).resolve()
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {"params": params, "config": config}
    save_args = orbax_utils.save_args_from_target(payload)
    ocp.PyTreeCheckpointer().save(destination, payload, save_args=save_args)


def _device_description():
    return ",".join(
        sorted(
            {
                f"{device.platform}:{getattr(device, 'device_kind', type(device).__name__)}"
                for device in jax.devices()
            }
        )
    )
