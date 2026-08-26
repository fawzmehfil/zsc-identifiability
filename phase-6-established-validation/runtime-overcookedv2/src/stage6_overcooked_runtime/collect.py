"""Ego-visible commitment trace collection from frozen official checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from overcooked_v2_experiments.ppo.policy import PPOPolicy

from stage6_overcooked_runtime.controller import DiagnosticGoalController
from stage6_overcooked_runtime.events import transition_event_features


def collect_traces(request):
    payload = request["payload"]
    ego_policy = _load_policy(payload["ego_checkpoint"], stochastic=payload.get("stochastic", True))
    partner_specs = payload["partners"]
    layout = payload["layout_id"]
    output = Path(payload["trace_path"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    commitment_episodes = 0
    first_units = 0
    source_hashes = [_file_tree_hash(Path(payload["ego_checkpoint"]))]
    for partner_spec in partner_specs:
        partner_policy = _load_policy(
            partner_spec["checkpoint_path"],
            stochastic=payload.get("stochastic", True),
        )
        source_hashes.append(_file_tree_hash(Path(partner_spec["checkpoint_path"])))
        for environment_key in payload["environment_keys"]:
            episode_records, reached = _episode(
                ego_policy,
                partner_policy,
                layout,
                int(environment_key),
                partner_spec,
                payload,
            )
            records.extend(episode_records)
            first_units += 1
            commitment_episodes += int(reached)
    records.sort(key=lambda row: (row["episode_id"], row["work_unit"], row["step"]))
    serialized = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in records
    )
    output.write_text(serialized)
    trace_hash = hashlib.sha256(serialized.encode()).hexdigest()
    return {
        "trace_path": str(output),
        "trace_hash": trace_hash,
        "evidence_policy": payload["evidence_policy"],
        "split": payload["split"],
        "partner_ids": [item["partner_id"] for item in partner_specs],
        "episode_count": len(partner_specs) * len(payload["environment_keys"]),
        "first_work_unit_count": first_units,
        "commitment_reached_rate": commitment_episodes / max(first_units, 1),
        "post_commitment_excluded": True,
        "source_checkpoint_hashes": sorted(source_hashes),
    }


def _episode(ego_policy, partner_policy, layout, environment_key, partner_spec, payload):
    import jaxmarl

    env = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=400,
        agent_view_size=2,
        random_agent_positions=True,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
    )
    root_key = jax.random.PRNGKey(environment_key)
    root_key, reset_key = jax.random.split(root_key)
    observations, state = env.reset(reset_key)
    hidden = [ego_policy.init_hstate(1), partner_policy.init_hstate(1)]
    done = [False, False]
    work_unit = 0
    committed_in_first = False
    records = []
    episode_id = f"{partner_spec['partner_id']}:{environment_key}"
    controller = DiagnosticGoalController(
        payload["evidence_policy"], payload.get("candidate_ingredient", 0)
    )
    option_steps = min(int(payload.get("option_steps", 16)), 16)
    for step in range(400):
        root_key, ego_key, partner_key, transition_key = jax.random.split(root_key, 4)
        ego_action, hidden[0] = ego_policy.compute_action(
            observations["agent_0"], done[0], hidden[0], ego_key
        )
        controller_action = controller.action(state) if step < option_steps else None
        if controller_action is not None:
            ego_action = controller_action
        partner_action, hidden[1] = partner_policy.compute_action(
            observations["agent_1"], done[1], hidden[1], partner_key
        )
        actions = {"agent_0": ego_action, "agent_1": partner_action}
        next_observations, next_state, rewards, dones, _ = env.step(
            transition_key, state, actions
        )
        features = transition_event_features(env, state, next_state, actions, rewards)
        visible = _partner_visible(state, view_size=2)
        task_events = []
        visible_events = []
        for name, values in features.items():
            values = np.asarray(values)
            if float(values[0]) > 0:
                visible_events.append(f"ego:{name}")
            if float(values[1]) > 0 and visible:
                visible_events.append(f"partner:{name}")
            if name == "pot_fill":
                for actor_index, actor in enumerate(("ego", "partner")):
                    if float(values[actor_index]) > 0:
                        task_events.append(
                            {
                                "name": "successful_pot_ingredient_placement",
                                "actor": actor,
                                "success": True,
                                "metadata": {},
                            }
                        )
                        if work_unit == 0:
                            committed_in_first = True
            if name == "delivery" and float(values.sum()) > 0:
                task_events.append(
                    {
                        "name": "successful_delivery",
                        "actor": "ego" if float(values[0]) > 0 else "partner",
                        "success": True,
                        "metadata": {"correct": bool(next_state.new_correct_delivery)},
                    }
                )
        records.append(
            {
                "episode_id": episode_id,
                "partner_id": partner_spec["partner_id"],
                "reward_vector_id": partner_spec["reward_vector_id"],
                "layout_id": layout,
                "environment_key": str(environment_key),
                "work_unit": work_unit,
                "step": step,
                "ego_observation": np.asarray(observations["agent_0"]).reshape(-1).tolist(),
                "ego_action": int(ego_action),
                "visible_partner_action": int(partner_action) if visible else None,
                "reward": float(rewards["agent_0"]),
                "events": task_events,
                "high_level_events": sorted(visible_events),
            }
        )
        if any(event["name"] == "successful_delivery" for event in task_events):
            work_unit += 1
        observations, state = next_observations, next_state
        done = [bool(dones["agent_0"]), bool(dones["agent_1"])]
        if bool(dones["__all__"]):
            break
    return records, committed_in_first


def _load_policy(path, stochastic):
    checkpoint = ocp.PyTreeCheckpointer().restore(Path(path).resolve(), item=None)
    return _JittedPolicy(
        PPOPolicy(checkpoint["params"], checkpoint["config"], stochastic=stochastic)
    )


class _JittedPolicy:
    def __init__(self, policy):
        self.policy = policy
        self._compute = jax.jit(policy.compute_action)

    def init_hstate(self, batch_size):
        return self.policy.init_hstate(batch_size)

    def compute_action(self, observation, done, hidden, key):
        return self._compute(observation, done, hidden, key)


def _partner_visible(state, view_size):
    positions = np.asarray(state.agents.pos.to_array())
    return bool(np.max(np.abs(positions[0] - positions[1])) <= view_size)


def _file_tree_hash(path):
    digest = hashlib.sha256()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()
