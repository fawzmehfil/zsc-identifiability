"""Supervised sequence training for TBS concept models and CSP dynamics."""

from __future__ import annotations

import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN

from stage6_overcooked_runtime.ported_methods import CSPTrajectoryModel, VisibleHistoryPredictor


def train_visible_predictor(
    observations,
    targets,
    visibility_mask,
    *,
    seed: int,
    output_path: str | Path,
    validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    learning_rate: float = 3e-4,
    maximum_epochs: int = 100,
    patience: int = 10,
):
    """Train a Bernoulli concept predictor with validation early stopping."""

    obs = jnp.asarray(observations, dtype=jnp.float32)
    labels = jnp.asarray(targets, dtype=jnp.float32)
    mask = jnp.asarray(visibility_mask, dtype=jnp.float32)
    if obs.ndim < 4 or labels.shape[:2] != obs.shape[:2]:
        raise ValueError("TBS observations and concept targets must be time-major sequences")
    output_dim = int(labels.shape[-1])
    network = VisibleHistoryPredictor(output_dim=output_dim)
    hidden = ScannedRNN.initialize_carry(obs.shape[1], 128)
    resets = jnp.zeros(obs.shape[:2], dtype=jnp.bool_).at[0].set(True)
    params = network.init(jax.random.PRNGKey(seed), hidden, obs, resets)
    state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate)),
    )

    def loss(parameters, batch_obs, batch_labels, batch_mask):
        batch_hidden = ScannedRNN.initialize_carry(batch_obs.shape[1], 128)
        batch_resets = jnp.zeros(batch_obs.shape[:2], dtype=jnp.bool_).at[0].set(True)
        _, probabilities = network.apply(parameters, batch_hidden, batch_obs, batch_resets)
        probabilities = jnp.clip(probabilities, 1e-6, 1 - 1e-6)
        element = -batch_labels * jnp.log(probabilities) - (1 - batch_labels) * jnp.log(
            1 - probabilities
        )
        weighted = element * batch_mask[..., None]
        return weighted.sum() / jnp.maximum(batch_mask.sum() * batch_labels.shape[-1], 1.0)

    @jax.jit
    def update(train_state):
        value, gradients = jax.value_and_grad(loss)(train_state.params, obs, labels, mask)
        return train_state.apply_gradients(grads=gradients), value

    best_params = state.params
    best_validation = float("inf")
    stale = 0
    history = []
    for _ in range(maximum_epochs):
        state, training_loss = update(state)
        validation_loss = _visible_validation_loss(
            network,
            state.params,
            validation,
            loss,
            float(training_loss),
        )
        history.append((float(training_loss), validation_loss))
        if validation_loss < best_validation - 1e-8:
            best_validation = validation_loss
            best_params = state.params
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    _save_auxiliary(
        output_path,
        best_params,
        {
            "model": "visible_history_predictor",
            "output_dim": output_dim,
            "hidden_size": 128,
        },
    )
    return {
        "epochs": len(history),
        "training_loss": history[-1][0],
        "validation_loss": best_validation,
    }


def train_csp_trajectory_model(
    observations,
    ego_actions,
    rewards,
    partner_action_targets,
    *,
    seed: int,
    output_path: str | Path,
    learning_rate: float = 3e-4,
    maximum_epochs: int = 100,
    patience: int = 10,
):
    obs = jnp.asarray(observations, dtype=jnp.float32)
    actions = jnp.asarray(ego_actions, dtype=jnp.int32)
    reward_values = jnp.asarray(rewards, dtype=jnp.float32)
    targets = jnp.asarray(partner_action_targets, dtype=jnp.int32)
    if targets.shape != obs.shape[:2]:
        raise ValueError("CSP partner-action targets must match the sequence axes")
    network = CSPTrajectoryModel()
    hidden = ScannedRNN.initialize_carry(obs.shape[1], 128)
    resets = jnp.zeros(obs.shape[:2], dtype=jnp.bool_).at[0].set(True)
    params = network.init(jax.random.PRNGKey(seed), hidden, obs, actions, reward_values, resets)
    state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate)),
    )

    def loss(parameters):
        initial = ScannedRNN.initialize_carry(obs.shape[1], 128)
        _, _, logits = network.apply(parameters, initial, obs, actions, reward_values, resets)
        element = optax.softmax_cross_entropy_with_integer_labels(logits[:-1], targets[1:])
        valid = (~resets[1:]).astype(jnp.float32)
        return (element * valid).sum() / jnp.maximum(valid.sum(), 1.0)

    @jax.jit
    def update(train_state):
        value, gradients = jax.value_and_grad(loss)(train_state.params)
        return train_state.apply_gradients(grads=gradients), value

    best_params = state.params
    best_loss = float("inf")
    stale = 0
    history = []
    for _ in range(maximum_epochs):
        state, value = update(state)
        scalar = float(value)
        history.append(scalar)
        if scalar < best_loss - 1e-8:
            best_loss = scalar
            best_params = state.params
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    _save_auxiliary(
        output_path,
        best_params,
        {
            "model": "csp_trajectory_model",
            "action_dim": 7,
            "hidden_size": 128,
            "latent_size": 32,
        },
    )
    return {"epochs": len(history), "training_loss": history[-1]}


def encode_csp_trajectories(
    checkpoint_path,
    observations,
    ego_actions,
    rewards,
) -> np.ndarray:
    payload = ocp.PyTreeCheckpointer().restore(Path(checkpoint_path).resolve(), item=None)
    network = CSPTrajectoryModel()
    obs = jnp.asarray(observations, dtype=jnp.float32)
    actions = jnp.asarray(ego_actions, dtype=jnp.int32)
    reward_values = jnp.asarray(rewards, dtype=jnp.float32)
    resets = jnp.zeros(obs.shape[:2], dtype=jnp.bool_).at[0].set(True)
    hidden = ScannedRNN.initialize_carry(obs.shape[1], 128)
    _, latent, _ = network.apply(payload["params"], hidden, obs, actions, reward_values, resets)
    return np.asarray(latent[-1])


def _visible_validation_loss(
    network,
    parameters,
    validation,
    loss_function,
    training_loss,
):
    if validation is None:
        return training_loss
    observations, targets, mask = validation
    return float(
        loss_function(
            parameters,
            jnp.asarray(observations, dtype=jnp.float32),
            jnp.asarray(targets, dtype=jnp.float32),
            jnp.asarray(mask, dtype=jnp.float32),
        )
    )


def _save_auxiliary(path, params, config):
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    payload = {"params": params, "config": config}
    save_args = orbax_utils.save_args_from_target(payload)
    ocp.PyTreeCheckpointer().save(destination, payload, save_args=save_args)
