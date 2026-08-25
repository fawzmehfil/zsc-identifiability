"""Native vector environment for finite convention-game learning."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np

from zsc_identifiability.learning_models import LearningGame
from zsc_identifiability.learning_specialists import cluster_training_strategies
from zsc_identifiability.models import FiniteConventionGame
from zsc_identifiability.numeric import parse_rational


@dataclass(frozen=True)
class ObservationLayout:
    state_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    feature_names: tuple[str, ...]

    @property
    def observation_size(self) -> int:
        return len(self.feature_names)

    @property
    def action_size(self) -> int:
        return len(self.action_ids)


@dataclass(frozen=True)
class StepBatch:
    observations: np.ndarray
    action_masks: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    partner_identity: np.ndarray
    response_target: np.ndarray
    strategy_cluster: np.ndarray
    infos: tuple[dict[str, Any], ...]


class VectorConventionEnvironment:
    """Small deterministic-shape vector environment with exact schema semantics."""

    def __init__(
        self,
        games: tuple[LearningGame, ...],
        seed: int,
        num_envs: int,
        *,
        action_class: Literal["passive", "task"] = "task",
        loss_scale: float = 40.0,
    ) -> None:
        if not games:
            raise ValueError("vector environment requires at least one game")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if loss_scale <= 0:
            raise ValueError("loss_scale must be positive")
        self.games = games
        self.num_envs = num_envs
        self.action_class = action_class
        self.loss_scale = loss_scale
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._validate_compatible_games()
        first = games[0].game
        self.layout = build_observation_layout(first)
        action_ids = self.layout.action_ids
        self._state_index = {item: index for index, item in enumerate(self.layout.state_ids)}
        self._observation_index = {
            item: index for index, item in enumerate(self.layout.observation_ids)
        }
        self._action_index = {item: index for index, item in enumerate(action_ids)}
        self._partner_labels = tuple(
            f"{item.partner_identity_prefix}:{mode}"
            for item in games
            for mode in item.game.mode_ids
        )
        self._partner_index = {label: index for index, label in enumerate(self._partner_labels)}
        self.game_indices = np.zeros(num_envs, dtype=np.int64)
        self.mode_indices = np.zeros(num_envs, dtype=np.int64)
        self.times = np.zeros(num_envs, dtype=np.int64)
        self.states = np.empty(num_envs, dtype=object)
        self.latest_observations = np.empty(num_envs, dtype=object)
        self.previous_actions = np.empty(num_envs, dtype=object)
        self.previous_rewards = np.zeros(num_envs, dtype=np.float64)
        self.terminated = np.ones(num_envs, dtype=np.bool_)
        self.episode_costs = np.zeros(num_envs, dtype=np.float64)
        self.episode_losses = np.zeros(num_envs, dtype=np.float64)
        self.episode_steps = np.zeros(num_envs, dtype=np.int64)
        self.reset()

    @property
    def partner_identity_count(self) -> int:
        return len(self._partner_labels)

    @property
    def partner_labels(self) -> tuple[str, ...]:
        return self._partner_labels

    @property
    def partner_response_classes(self) -> tuple[int, ...]:
        result: list[int] = []
        decisions = tuple(sorted(self.games[0].game.decisions))
        for item in self.games:
            for mode in item.game.mode_ids:
                zero_loss = next(
                    decision for decision in decisions if item.game.loss_exact(mode, decision) == 0
                )
                result.append(decisions.index(zero_loss))
        return tuple(result)

    @property
    def partner_strategy_clusters(self) -> tuple[int, ...]:
        """Training-only clusters recovered from exact cross-play return rows."""
        result: list[int] = []
        for item in self.games:
            clusters = cluster_training_strategies(item.game)
            for mode in item.game.mode_ids:
                result.append(clusters.mode_to_cluster[mode])
        return tuple(result)

    @property
    def response_target_count(self) -> int:
        return len(self.games[0].game.observations)

    def reset(self, indices: np.ndarray | None = None) -> StepBatch:
        selected = np.arange(self.num_envs) if indices is None else np.asarray(indices)
        for env_index in selected.tolist():
            game_index = int(self.rng.integers(len(self.games)))
            item = self.games[game_index]
            prior = np.asarray([float(value) for value in item.game.prior_exact()])
            mode_index = int(self.rng.choice(len(prior), p=prior))
            self.game_indices[env_index] = game_index
            self.mode_indices[env_index] = mode_index
            self.times[env_index] = 0
            self.states[env_index] = item.game.initial_state
            self.latest_observations[env_index] = "<start>"
            self.previous_actions[env_index] = "<start>"
            self.previous_rewards[env_index] = 0.0
            self.terminated[env_index] = False
            self.episode_costs[env_index] = 0.0
            self.episode_losses[env_index] = 0.0
            self.episode_steps[env_index] = 0
        return self.current_batch(rewards=np.zeros(self.num_envs), infos=())

    def current_batch(
        self,
        *,
        rewards: np.ndarray | None = None,
        infos: tuple[dict[str, Any], ...] = (),
    ) -> StepBatch:
        masks = np.stack([self._action_mask(index) for index in range(self.num_envs)])
        observations = np.stack(
            [self._observation_vector(index, masks[index]) for index in range(self.num_envs)]
        )
        if rewards is None:
            rewards = np.zeros(self.num_envs, dtype=np.float64)
        if not infos:
            infos = tuple({} for _ in range(self.num_envs))
        return StepBatch(
            observations=observations,
            action_masks=masks,
            rewards=np.asarray(rewards, dtype=np.float64),
            terminated=self.terminated.copy(),
            partner_identity=self._partner_identity_indices(),
            response_target=self._response_target_indices(),
            strategy_cluster=self._strategy_cluster_indices(),
            infos=infos,
        )

    def step(self, actions: np.ndarray) -> StepBatch:
        selected = np.asarray(actions, dtype=np.int64)
        if selected.shape != (self.num_envs,):
            raise ValueError(f"actions must have shape ({self.num_envs},)")
        rewards = np.zeros(self.num_envs, dtype=np.float64)
        infos: list[dict[str, Any]] = []
        for index, action_index in enumerate(selected.tolist()):
            if self.terminated[index]:
                if action_index != -1:
                    raise RuntimeError("terminated environments accept only the -1 inactive action")
                infos.append({"inactive": True})
                continue
            mask = self._action_mask(index)
            if action_index < 0 or action_index >= len(mask) or not mask[action_index]:
                raise ValueError(f"invalid action {action_index} for environment {index}")
            action_id = self.layout.action_ids[action_index]
            item = self.games[int(self.game_indices[index])]
            game = item.game
            mode = game.mode_ids[int(self.mode_indices[index])]
            info: dict[str, Any] = {
                "game_id": game.game_id,
                "mode": mode,
                "profile_id": item.profile_id,
                "action": action_id,
                "intervention_cost": 0.0,
                "confusion_loss": 0.0,
            }
            if action_id.startswith("commit:"):
                decision = action_id.removeprefix("commit:")
                loss = float(game.loss_exact(mode, decision))
                rewards[index] = -loss / self.loss_scale
                self.episode_losses[index] += loss
                self.terminated[index] = True
                info.update(
                    {
                        "decision": decision,
                        "confusion_loss": loss,
                        "episode_cost": float(self.episode_costs[index]),
                        "episode_loss": float(self.episode_losses[index]),
                        "episode_steps": int(self.episode_steps[index]),
                    }
                )
            else:
                row = game.kernel(int(self.times[index]), str(self.states[index]), action_id, mode)
                probabilities = np.asarray(
                    [float(parse_rational(outcome.probability)) for outcome in row.outcomes]
                )
                outcome = row.outcomes[int(self.rng.choice(len(row.outcomes), p=probabilities))]
                cost = float(parse_rational(outcome.cost))
                rewards[index] = -cost / self.loss_scale
                self.episode_costs[index] += cost
                self.times[index] += 1
                self.episode_steps[index] += 1
                self.states[index] = outcome.next_state
                self.latest_observations[index] = outcome.observation
                info.update(
                    {
                        "observation": outcome.observation,
                        "next_state": outcome.next_state,
                        "intervention_cost": cost,
                    }
                )
            self.previous_actions[index] = action_id
            self.previous_rewards[index] = rewards[index]
            infos.append(info)
        return self.current_batch(rewards=rewards, infos=tuple(infos))

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "game_indices": self.game_indices.tolist(),
            "mode_indices": self.mode_indices.tolist(),
            "times": self.times.tolist(),
            "states": self.states.tolist(),
            "latest_observations": self.latest_observations.tolist(),
            "previous_actions": self.previous_actions.tolist(),
            "previous_rewards": self.previous_rewards.tolist(),
            "terminated": self.terminated.tolist(),
            "episode_costs": self.episode_costs.tolist(),
            "episode_losses": self.episode_losses.tolist(),
            "episode_steps": self.episode_steps.tolist(),
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.rng.bit_generator.state = payload["rng_state"]
        for name, dtype in (
            ("game_indices", np.int64),
            ("mode_indices", np.int64),
            ("times", np.int64),
            ("previous_rewards", np.float64),
            ("terminated", np.bool_),
            ("episode_costs", np.float64),
            ("episode_losses", np.float64),
            ("episode_steps", np.int64),
        ):
            setattr(self, name, np.asarray(payload[name], dtype=dtype))
        self.states = np.asarray(payload["states"], dtype=object)
        self.latest_observations = np.asarray(payload["latest_observations"], dtype=object)
        self.previous_actions = np.asarray(payload["previous_actions"], dtype=object)

    def _validate_compatible_games(self) -> None:
        first = self.games[0].game
        signature = (
            tuple(sorted(first.states)),
            tuple(sorted(first.observations)),
            tuple(sorted(first.action_ids)),
            tuple(sorted(first.decisions)),
            first.horizon,
        )
        for item in self.games[1:]:
            game = item.game
            current = (
                tuple(sorted(game.states)),
                tuple(sorted(game.observations)),
                tuple(sorted(game.action_ids)),
                tuple(sorted(game.decisions)),
                game.horizon,
            )
            if current != signature:
                raise ValueError("all games in one vector environment must share vocabularies")

    def _action_mask(self, index: int) -> np.ndarray:
        if self.terminated[index]:
            return np.zeros(len(self.layout.action_ids), dtype=np.bool_)
        item = self.games[int(self.game_indices[index])]
        return runtime_action_mask(
            item,
            int(self.times[index]),
            str(self.states[index]),
            self.layout,
            self.action_class,
        )

    def _observation_vector(self, index: int, mask: np.ndarray) -> np.ndarray:
        return encode_runtime_observation(
            self.games[int(self.game_indices[index])].game,
            self.layout,
            int(self.times[index]),
            str(self.states[index]),
            str(self.latest_observations[index]),
            str(self.previous_actions[index]),
            float(self.previous_rewards[index]),
            mask,
        )

    def _partner_identity_indices(self) -> np.ndarray:
        result = np.empty(self.num_envs, dtype=np.int64)
        for index in range(self.num_envs):
            item = self.games[int(self.game_indices[index])]
            mode = item.game.mode_ids[int(self.mode_indices[index])]
            result[index] = self._partner_index[f"{item.partner_identity_prefix}:{mode}"]
        return result

    def _response_target_indices(self) -> np.ndarray:
        result = np.full(self.num_envs, -1, dtype=np.int64)
        for index, observation in enumerate(self.latest_observations.tolist()):
            if observation == "<start>":
                continue
            result[index] = self.games[0].game.observations.index(str(observation))
        return result

    def _strategy_cluster_indices(self) -> np.ndarray:
        labels = self.partner_strategy_clusters
        identities = self._partner_identity_indices()
        return np.asarray([labels[int(identity)] for identity in identities], dtype=np.int64)


def expected_kernel_distribution(
    item: LearningGame, time: int, state: str, action: str, mode: str
) -> dict[tuple[str, str, float], Fraction]:
    """Expose exact transition probabilities for environment calibration tests."""
    return {
        (outcome.next_state, outcome.observation, float(parse_rational(outcome.cost))): (
            parse_rational(outcome.probability)
        )
        for outcome in item.game.kernel(time, state, action, mode).outcomes
    }


def build_observation_layout(game: FiniteConventionGame) -> ObservationLayout:
    task_actions = tuple(sorted(game.action_ids))
    decisions = tuple(f"commit:{item}" for item in sorted(game.decisions))
    action_ids = task_actions + decisions
    observations = ("<start>",) + tuple(sorted(game.observations))
    previous_actions = ("<start>",) + action_ids
    feature_names = (
        *(f"state:{item}" for item in sorted(game.states)),
        *(f"observation:{item}" for item in observations),
        *(f"previous_action:{item}" for item in previous_actions),
        "previous_external_reward",
        "time_fraction",
        "remaining_fraction",
        *(f"legal:{item}" for item in action_ids),
    )
    return ObservationLayout(
        state_ids=tuple(sorted(game.states)),
        observation_ids=observations,
        action_ids=action_ids,
        feature_names=tuple(feature_names),
    )


def runtime_action_mask(
    item: LearningGame,
    time: int,
    state: str,
    layout: ObservationLayout,
    action_class: Literal["passive", "task"],
) -> np.ndarray:
    game = item.game
    action_index = {action: index for index, action in enumerate(layout.action_ids)}
    mask = np.zeros(len(layout.action_ids), dtype=np.bool_)
    if time < game.horizon:
        for action in game.available_actions(state, time, passive_only=action_class == "passive"):
            mask[action_index[action]] = True
    if state in item.commitment_states or time >= game.horizon:
        for decision in game.decisions:
            mask[action_index[f"commit:{decision}"]] = True
    if not mask.any():
        raise RuntimeError(f"no legal action at time={time}, state={state}")
    return mask


def encode_runtime_observation(
    game: FiniteConventionGame,
    layout: ObservationLayout,
    time: int,
    state: str,
    latest_observation: str,
    previous_action: str,
    previous_external_reward: float,
    action_mask: np.ndarray,
) -> np.ndarray:
    state_features = np.zeros(len(layout.state_ids), dtype=np.float32)
    state_features[layout.state_ids.index(state)] = 1
    observation_features = np.zeros(len(layout.observation_ids), dtype=np.float32)
    observation_features[layout.observation_ids.index(latest_observation)] = 1
    previous_features = np.zeros(len(layout.action_ids) + 1, dtype=np.float32)
    previous_index = (
        0 if previous_action == "<start>" else layout.action_ids.index(previous_action) + 1
    )
    previous_features[previous_index] = 1
    scalars = np.asarray(
        [
            previous_external_reward,
            time / game.horizon,
            max(0, game.horizon - time) / game.horizon,
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        (
            state_features,
            observation_features,
            previous_features,
            scalars,
            action_mask.astype(np.float32),
        )
    )
