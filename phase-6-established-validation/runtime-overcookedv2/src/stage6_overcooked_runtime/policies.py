"""Deployment-only policy factory for ordinary and composite Stage 6 methods."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from overcooked_v2_experiments.ppo.policy import PPOPolicy

from stage6_overcooked_runtime.ported_methods import (
    CSPTrajectoryModel,
    PaceActorCritic,
    VisibleHistoryPredictor,
    bernoulli_kl,
    select_csp_cluster,
)


def load_established_policy(path: str | Path, *, stochastic: bool):
    source = Path(path).resolve()
    if source.is_file():
        artifact = json.loads(source.read_text())
        if int(artifact.get("schema_version", -1)) != 1:
            raise ValueError("unsupported established policy artifact schema")
        kind = artifact["policy_kind"]
        if kind == "ppo":
            return _load_ppo(_component_path(artifact, "task_policy"), stochastic)
        if kind == "pace":
            return PaceDeploymentPolicy(artifact, stochastic=stochastic)
        if kind == "tbs_selector":
            return TBSDeploymentPolicy(artifact, stochastic=stochastic)
        if kind == "csp_reconnaissance":
            return CSPDeploymentPolicy(artifact, stochastic=stochastic)
        raise ValueError(f"unknown established policy kind: {kind!r}")
    return _load_ppo(source, stochastic)


def _load_ppo(path: str | Path, stochastic: bool):
    checkpoint = ocp.PyTreeCheckpointer().restore(Path(path).resolve(), item=None)
    return JittedPPOPolicy(
        PPOPolicy(checkpoint["params"], checkpoint["config"], stochastic=stochastic)
    )


class JittedPPOPolicy:
    protocol_phases = ("scored",)

    def __init__(self, policy):
        self.policy = policy
        self._compute = jax.jit(policy.compute_action)

    def init_hstate(self, batch_size, protocol_phase="scored"):
        if protocol_phase not in self.protocol_phases:
            raise ValueError(f"policy does not support phase {protocol_phase!r}")
        return self.policy.init_hstate(batch_size)

    def compute_action(self, observation, done, hidden, key, protocol_phase="scored"):
        if protocol_phase != "scored":
            raise ValueError("ordinary PPO policies have no reconnaissance phase")
        action, state = self._compute(observation, done, hidden, key)
        return action, state, {}

    def record_action(self, state, actual_action):
        del actual_action
        return state


@dataclass
class PaceState:
    hidden: object
    previous_action: object


class PaceDeploymentPolicy:
    protocol_phases = ("scored",)

    def __init__(self, artifact, *, stochastic):
        component = _component(artifact, "task_policy")
        payload = ocp.PyTreeCheckpointer().restore(Path(component["path"]).resolve(), item=None)
        config = payload["config"]
        self.network = PaceActorCritic(
            action_dim=int(config.get("action_dim", 6)),
            partner_count=int(config["partner_count"]),
        )
        self.params = payload["params"]
        self.stochastic = stochastic

    def init_hstate(self, batch_size, protocol_phase="scored"):
        if protocol_phase != "scored":
            raise ValueError("PACE has no reconnaissance phase")
        from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN

        return PaceState(
            hidden=ScannedRNN.initialize_carry(batch_size, 128),
            previous_action=jnp.zeros((batch_size,), dtype=jnp.int32),
        )

    def compute_action(self, observation, done, state, key, protocol_phase="scored"):
        obs = jnp.asarray(observation)[None, None, ...]
        resets = jnp.asarray([[done]], dtype=jnp.bool_)
        actions = jnp.asarray(state.previous_action).reshape((1, 1))
        hidden, distribution, _, identity_logits = self.network.apply(
            self.params,
            state.hidden,
            obs,
            actions,
            resets,
            method=self.network.with_identity,
        )
        action = (
            distribution.sample(seed=key)[0, 0]
            if self.stochastic
            else jnp.argmax(distribution.probs[0, 0])
        )
        probabilities = jax.nn.softmax(identity_logits[0, 0])
        return (
            action,
            PaceState(hidden=hidden, previous_action=jnp.asarray([action])),
            {"identity_probabilities": np.asarray(probabilities).tolist()},
        )

    def record_action(self, state, actual_action):
        return PaceState(hidden=state.hidden, previous_action=jnp.asarray([actual_action]))


@dataclass
class TBSState:
    specialist_states: tuple[object, ...]
    global_state: object
    cluster_states: tuple[object, ...]
    cumulative_kl: object
    selected_cluster: int | None


class TBSDeploymentPolicy:
    protocol_phases = ("scored",)

    def __init__(self, artifact, *, stochastic):
        self.specialists = tuple(
            _load_ppo(component["path"], stochastic)
            for component in _components(artifact, "specialist")
        )
        if not self.specialists:
            raise ValueError("TBS artifacts require specialists")
        self.global_predictor, self.global_params = _load_predictor(
            _component(artifact, "global_tom"), len(artifact["concept_schema"])
        )
        cluster_predictors = _components(artifact, "cluster_tom")
        self.cluster_predictors = tuple(
            _load_predictor(component, len(artifact["concept_schema"]))
            for component in cluster_predictors
        )
        if len(self.cluster_predictors) != len(self.specialists):
            raise ValueError("TBS requires one predictor per specialist cluster")

    def init_hstate(self, batch_size, protocol_phase="scored"):
        from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN

        return TBSState(
            specialist_states=tuple(policy.init_hstate(batch_size) for policy in self.specialists),
            global_state=ScannedRNN.initialize_carry(batch_size, 128),
            cluster_states=tuple(
                ScannedRNN.initialize_carry(batch_size, 128) for _ in self.cluster_predictors
            ),
            cumulative_kl=jnp.zeros((len(self.specialists),)),
            selected_cluster=None,
        )

    def compute_action(self, observation, done, state, key, protocol_phase="scored"):
        keys = jax.random.split(key, len(self.specialists) + 1)
        specialist_outputs = tuple(
            policy.compute_action(observation, done, hidden, policy_key)
            for policy, hidden, policy_key in zip(
                self.specialists, state.specialist_states, keys[:-1], strict=True
            )
        )
        obs = jnp.asarray(observation)[None, None, ...]
        reset = jnp.asarray([[done]], dtype=jnp.bool_)
        global_state, global_prediction = self.global_predictor.apply(
            self.global_params, state.global_state, obs, reset
        )
        cluster_outputs = tuple(
            predictor.apply(params, hidden, obs, reset)
            for (predictor, params), hidden in zip(
                self.cluster_predictors, state.cluster_states, strict=True
            )
        )
        cluster_predictions = jnp.stack([output[1][0, 0] for output in cluster_outputs])
        increment = bernoulli_kl(global_prediction[0, 0], cluster_predictions).sum(-1)
        cumulative = state.cumulative_kl + increment
        selected = (
            int(jax.random.randint(keys[-1], (), 0, len(self.specialists)))
            if state.selected_cluster is None
            else state.selected_cluster
        )
        next_selected = int(jnp.argmin(cumulative))
        return (
            specialist_outputs[selected][0],
            TBSState(
                specialist_states=tuple(output[1] for output in specialist_outputs),
                global_state=global_state,
                cluster_states=tuple(output[0] for output in cluster_outputs),
                cumulative_kl=cumulative,
                selected_cluster=next_selected,
            ),
            {
                "selected_cluster": selected,
                "next_selected_cluster": next_selected,
                "cluster_kl": np.asarray(cumulative).tolist(),
            },
        )

    def record_action(self, state, actual_action):
        del actual_action
        return state


@dataclass
class CSPState:
    policy_state: object
    encoder_state: object
    previous_action: object
    previous_reward: object
    final_embedding: object
    selected_cluster: int | None
    pending_observation: object | None
    pending_done: bool


class CSPDeploymentPolicy:
    protocol_phases = ("reconnaissance", "scored")

    def __init__(self, artifact, *, stochastic):
        self.probe = _load_ppo(_component_path(artifact, "probe_policy"), stochastic)
        self.specialists = tuple(
            _load_ppo(component["path"], stochastic)
            for component in _components(artifact, "specialist")
        )
        encoder_payload = ocp.PyTreeCheckpointer().restore(
            Path(_component_path(artifact, "trajectory_encoder")).resolve(), item=None
        )
        self.encoder = CSPTrajectoryModel()
        self.encoder_params = encoder_payload["params"]
        self.centroids = np.asarray(artifact["centroids"], dtype=np.float64)

    def init_hstate(self, batch_size, protocol_phase="reconnaissance"):
        from overcooked_v2_experiments.ppo.models.rnn import ScannedRNN

        policy = self.probe if protocol_phase == "reconnaissance" else self.specialists[0]
        return CSPState(
            policy_state=policy.init_hstate(batch_size),
            encoder_state=ScannedRNN.initialize_carry(batch_size, 128),
            previous_action=jnp.zeros((batch_size,), dtype=jnp.int32),
            previous_reward=jnp.zeros((batch_size,), dtype=jnp.float32),
            final_embedding=jnp.zeros((32,), dtype=jnp.float32),
            selected_cluster=None,
            pending_observation=None,
            pending_done=False,
        )

    def select_specialist(self, reconnaissance_state: CSPState) -> int:
        return select_csp_cluster(reconnaissance_state.final_embedding, self.centroids)

    def scored_state(self, reconnaissance_state: CSPState, batch_size=1):
        cluster = self.select_specialist(reconnaissance_state)
        state = self.init_hstate(batch_size, protocol_phase="scored")
        return CSPState(**{**state.__dict__, "selected_cluster": cluster})

    def record_reward(self, state: CSPState, reward: float) -> CSPState:
        if state.pending_observation is None:
            return CSPState(**{**state.__dict__, "previous_reward": jnp.asarray([reward])})
        observation = jnp.asarray(state.pending_observation)[None, None, ...]
        action = jnp.asarray(state.previous_action).reshape((1, 1))
        reward_value = jnp.asarray([[reward]], dtype=jnp.float32)
        reset = jnp.asarray([[state.pending_done]], dtype=jnp.bool_)
        encoder_state, latent, _ = self.encoder.apply(
            self.encoder_params,
            state.encoder_state,
            observation,
            action,
            reward_value,
            reset,
        )
        return CSPState(
            **{
                **state.__dict__,
                "encoder_state": encoder_state,
                "previous_reward": jnp.asarray([reward]),
                "final_embedding": latent[0, 0],
                "pending_observation": None,
            }
        )

    def record_action(self, state: CSPState, actual_action) -> CSPState:
        return CSPState(
            **{
                **state.__dict__,
                "previous_action": jnp.asarray([actual_action]),
            }
        )

    def compute_action(self, observation, done, state, key, protocol_phase="reconnaissance"):
        if protocol_phase == "reconnaissance":
            policy = self.probe
        else:
            if state.selected_cluster is None:
                raise ValueError("CSP scored phase requires a reconnaissance embedding")
            policy = self.specialists[state.selected_cluster]
        action, policy_state, _ = policy.compute_action(observation, done, state.policy_state, key)
        return (
            action,
            CSPState(
                policy_state=policy_state,
                encoder_state=state.encoder_state,
                previous_action=jnp.asarray([action]),
                previous_reward=state.previous_reward,
                final_embedding=state.final_embedding,
                selected_cluster=state.selected_cluster,
                pending_observation=(
                    np.asarray(observation) if protocol_phase == "reconnaissance" else None
                ),
                pending_done=bool(done),
            ),
            {"selected_cluster": state.selected_cluster},
        )


def _load_predictor(component, output_dim):
    payload = ocp.PyTreeCheckpointer().restore(Path(component["path"]).resolve(), item=None)
    return VisibleHistoryPredictor(output_dim=output_dim), payload["params"]


def _component(artifact, role):
    matches = _components(artifact, role)
    if len(matches) != 1:
        raise ValueError(f"policy artifact requires exactly one {role!r} component")
    return matches[0]


def _components(artifact, role):
    return tuple(
        sorted(
            (item for item in artifact["components"] if item["role"] == role),
            key=lambda item: (item.get("cluster_id", -1), item["component_id"]),
        )
    )


def _component_path(artifact, role):
    return _component(artifact, role)["path"]
