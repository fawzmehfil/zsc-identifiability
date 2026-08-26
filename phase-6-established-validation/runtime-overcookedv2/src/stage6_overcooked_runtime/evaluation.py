"""Paired checkpoint competence evaluation in the pinned environment."""

from __future__ import annotations

import jax
import numpy as np

from stage6_overcooked_runtime.collect import _load_policy
from stage6_overcooked_runtime.events import transition_event_features


def evaluate_checkpoint_pair(request):
    payload = request["payload"]
    ego = _load_policy(payload["ego_checkpoint"], stochastic=False)
    partner = _load_policy(payload["partner_checkpoint"], stochastic=False)
    episode_returns: list[float] = []
    correct_delivery_episodes = 0
    commitment_episodes = 0
    total_deliveries = 0
    for environment_key in payload["environment_keys"]:
        result = _episode(
            ego,
            partner,
            payload["layout_id"],
            int(environment_key),
        )
        episode_returns.append(result["return"])
        correct_delivery_episodes += int(result["correct_deliveries"] > 0)
        commitment_episodes += int(result["commitment_reached"])
        total_deliveries += result["correct_deliveries"]
    episode_count = len(episode_returns)
    if episode_count == 0:
        raise ValueError("checkpoint evaluation requires at least one environment key")
    return {
        "layout_id": payload["layout_id"],
        "episode_count": episode_count,
        "mean_sparse_return": float(np.mean(episode_returns)),
        "correct_delivery_episode_rate": correct_delivery_episodes / episode_count,
        "commitment_reached_rate": commitment_episodes / episode_count,
        "total_correct_deliveries": total_deliveries,
        "stochastic": False,
    }


def _episode(ego_policy, partner_policy, layout, environment_key):
    import jaxmarl

    environment = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=400,
        agent_view_size=2,
        random_agent_positions=True,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
    )
    key = jax.random.PRNGKey(environment_key)
    key, reset_key = jax.random.split(key)
    observations, state = environment.reset(reset_key)
    hidden = [ego_policy.init_hstate(1), partner_policy.init_hstate(1)]
    done = [False, False]
    total_return = 0.0
    correct_deliveries = 0
    commitment_reached = False
    for _ in range(400):
        key, ego_key, partner_key, step_key = jax.random.split(key, 4)
        ego_action, hidden[0] = ego_policy.compute_action(
            observations["agent_0"], done[0], hidden[0], ego_key
        )
        partner_action, hidden[1] = partner_policy.compute_action(
            observations["agent_1"], done[1], hidden[1], partner_key
        )
        actions = {"agent_0": ego_action, "agent_1": partner_action}
        next_observations, next_state, rewards, dones, _ = environment.step(
            step_key, state, actions
        )
        features = transition_event_features(
            environment, state, next_state, actions, rewards
        )
        commitment_reached |= float(np.asarray(features["pot_fill"]).sum()) > 0
        correct_deliveries += int(bool(next_state.new_correct_delivery))
        total_return += float(rewards["agent_0"])
        observations, state = next_observations, next_state
        done = [bool(dones["agent_0"]), bool(dones["agent_1"])]
        if bool(dones["__all__"]):
            break
    return {
        "return": total_return,
        "correct_deliveries": correct_deliveries,
        "commitment_reached": commitment_reached,
    }
