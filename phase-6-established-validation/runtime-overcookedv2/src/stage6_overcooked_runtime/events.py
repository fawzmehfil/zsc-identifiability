"""Transition-derived task events and behavior-preference shaping."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxmarl.environments.overcooked_v2.common import Actions, DynamicObject, StaticObject
from jaxmarl.wrappers.baselines import JaxMARLWrapper

SUPPORTED_BEHAVIOR_EVENTS = frozenset(
    {
        "ingredient_pickup_0",
        "ingredient_pickup_1",
        "ingredient_pot_placement_0",
        "ingredient_pot_placement_1",
        "counter_stage",
        "counter_retrieval",
        "pot_fill",
        "cook_start",
        "plate_pickup",
        "delivery",
        "recipe_button",
        "task_region_time",
        "corridor_entry",
        "yield",
        "idle",
        "movement",
        "successful_task_reward",
    }
)


class BehaviorPreferenceWrapper(JaxMARLWrapper):
    """Adds at most three declared event preferences to official shaping.

    The sparse shared task reward returned by the environment is unchanged.
    """

    def __init__(self, env, preferences):
        super().__init__(env)
        unknown = set(preferences) - SUPPORTED_BEHAVIOR_EVENTS
        if unknown:
            raise ValueError(f"unknown OvercookedV2 behavior events: {sorted(unknown)}")
        nonzero = {key: float(value) for key, value in preferences.items() if value != 0}
        if len(nonzero) > 3:
            raise ValueError("behavior preference vector may contain at most three non-zero terms")
        self.preferences = nonzero

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        return self._env.reset(key)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, state, actions, reset_state=None):
        observations, next_state, rewards, dones, info = self._env.step(
            key, state, actions, reset_state=reset_state
        )
        features = transition_event_features(self._env, state, next_state, actions, rewards)
        additions = {
            agent: sum(
                coefficient * features[event][index]
                for event, coefficient in self.preferences.items()
            )
            for index, agent in enumerate(self._env.agents)
        }
        original_shaping = info.get(
            "shaped_reward", {agent: jnp.asarray(0.0) for agent in self._env.agents}
        )
        info["stage6_behavior_features"] = features
        info["stage6_behavior_shaping"] = additions
        info["shaped_reward"] = {
            agent: original_shaping[agent] + additions[agent] for agent in self._env.agents
        }
        return observations, next_state, rewards, dones, info


def transition_event_features(env, state, next_state, actions, rewards):
    """Return per-agent visible task-event indicators from the state transition."""

    old_inventory = state.agents.inventory
    new_inventory = next_state.agents.inventory
    old_positions = state.agents.pos.to_array()
    new_positions = next_state.agents.pos.to_array()
    action_array = jnp.stack([actions[agent] for agent in env.agents])
    moved = jnp.any(old_positions != new_positions, axis=-1)
    stayed = action_array == Actions.stay
    interacted = action_array == Actions.interact
    old_empty = old_inventory == DynamicObject.EMPTY
    new_empty = new_inventory == DynamicObject.EMPTY
    old_ingredient = DynamicObject.is_ingredient(old_inventory)
    new_ingredient = DynamicObject.is_ingredient(new_inventory)
    pickup = old_empty & new_ingredient & interacted
    plate_pickup = old_empty & (new_inventory == DynamicObject.PLATE) & interacted
    delivery = DynamicObject.is_ingredient(old_inventory) == 0
    delivery &= (old_inventory & DynamicObject.COOKED) != 0
    delivery &= new_empty & interacted
    placed = old_ingredient & new_empty & interacted
    pot_mask = state.grid[:, :, 0] == StaticObject.POT
    old_pot = jnp.where(pot_mask, state.grid[:, :, 1], 0)
    new_pot = jnp.where(pot_mask, next_state.grid[:, :, 1], 0)
    pot_added = jnp.sum(jax.vmap(DynamicObject.ingredient_count)(old_pot.ravel())) < jnp.sum(
        jax.vmap(DynamicObject.ingredient_count)(new_pot.ravel())
    )
    pot_placement = placed & pot_added
    pot_became_cooking = jnp.any(
        pot_mask & (state.grid[:, :, 2] <= 0) & (next_state.grid[:, :, 2] > 0)
    )
    counter_mask = state.grid[:, :, 0] == StaticObject.WALL
    counter_total_old = jnp.sum(jnp.where(counter_mask, state.grid[:, :, 1] != 0, False))
    counter_total_new = jnp.sum(jnp.where(counter_mask, next_state.grid[:, :, 1] != 0, False))
    counter_stage = placed & (counter_total_new > counter_total_old) & ~pot_added
    counter_retrieval = pickup & (counter_total_new < counter_total_old)
    button_mask = state.grid[:, :, 0] == StaticObject.BUTTON_RECIPE_INDICATOR
    button_active = jnp.any(
        button_mask & (state.grid[:, :, 2] <= 0) & (next_state.grid[:, :, 2] > 0)
    )
    button_actor = interacted & old_empty & button_active
    task_mask = (
        (state.grid[:, :, 0] == StaticObject.POT)
        | (state.grid[:, :, 0] == StaticObject.GOAL)
        | StaticObject.is_ingredient_pile(state.grid[:, :, 0])
    )
    task_region = _adjacent_to_mask(new_positions, task_mask)
    move_area = state.grid[:, :, 0] == StaticObject.EMPTY
    numeric_move_area = move_area.astype(jnp.int32)
    degrees = (
        jnp.roll(numeric_move_area, 1, axis=0)
        + jnp.roll(numeric_move_area, -1, axis=0)
        + jnp.roll(numeric_move_area, 1, axis=1)
        + jnp.roll(numeric_move_area, -1, axis=1)
    )
    corridor = moved & (degrees[new_positions[:, 1], new_positions[:, 0]] <= 2)
    pair_distance = jnp.abs(old_positions[0] - old_positions[1]).sum()
    yielding = stayed & (pair_distance <= 2)
    successful = jnp.asarray([rewards[agent] > 0 for agent in env.agents])
    result = {
        "ingredient_pickup_0": pickup & (new_inventory == DynamicObject.ingredient(0)),
        "ingredient_pickup_1": pickup & (new_inventory == DynamicObject.ingredient(1)),
        "ingredient_pot_placement_0": pot_placement
        & (old_inventory == DynamicObject.ingredient(0)),
        "ingredient_pot_placement_1": pot_placement
        & (old_inventory == DynamicObject.ingredient(1)),
        "counter_stage": counter_stage,
        "counter_retrieval": counter_retrieval,
        "pot_fill": pot_placement,
        "cook_start": interacted & pot_became_cooking,
        "plate_pickup": plate_pickup,
        "delivery": delivery,
        "recipe_button": button_actor,
        "task_region_time": task_region,
        "corridor_entry": corridor,
        "yield": yielding,
        "idle": stayed,
        "movement": moved,
        "successful_task_reward": successful,
    }
    return {key: value.astype(jnp.float32) for key, value in result.items()}


def _adjacent_to_mask(positions, mask):
    y = positions[:, 1]
    x = positions[:, 0]
    padded = jnp.pad(mask, 1)
    py = y + 1
    px = x + 1
    return (
        padded[py - 1, px]
        | padded[py + 1, px]
        | padded[py, px - 1]
        | padded[py, px + 1]
    )
