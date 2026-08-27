"""Resumable multi-component pipelines for PACE, TBS, and CSP-style ports."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import jax
import jaxmarl
import numpy as np

from stage6_overcooked_runtime.auxiliary_training import (
    encode_csp_trajectories,
    train_csp_trajectory_model,
    train_visible_predictor,
)
from stage6_overcooked_runtime.checkpointing import directory_hash, write_deployment_artifact
from stage6_overcooked_runtime.collect import _episode, _load_policy
from stage6_overcooked_runtime.ego_training import train_ego_method
from stage6_overcooked_runtime.events import SUPPORTED_BEHAVIOR_EVENTS
from stage6_overcooked_runtime.ported_methods import deterministic_kmeans, pinned_tbs_clusters


def train_ported_method(request, project_root):
    method = request["payload"]["method_id"]
    if method in {"pace_aux", "pace_style"}:
        return _train_pace(request, project_root)
    if method == "tbs_style":
        return _train_tbs(request, project_root)
    if method == "csp_style_reconnaissance":
        return _train_csp(request, project_root)
    raise ValueError(f"unsupported Stage 6 method pipeline: {method!r}")


def _train_pace(request, project_root):
    result = train_ego_method(request, project_root)
    payload = request["payload"]
    artifact = _base_artifact(request, result, "pace")
    artifact["components"] = [
        _component("task-policy", "task_policy", result["checkpoint_paths"][-1])
    ]
    return _finish_artifact(payload, result, artifact)


def _train_tbs(request, project_root):
    payload = request["payload"]
    output = Path(payload["output_dir"]).resolve()
    resume_parent_hash = _resume_parent_hash(payload.get("resume_checkpoint"), output)
    pipeline = _load_pipeline_state(request, output)
    train_pool = _load_pool(payload["train_pool_path"])
    validation_pool = _load_pool(payload["validation_pool_path"])
    matrix, matrix_partner_ids = _load_cross_play(payload["cross_play_values_path"])
    if tuple(item["partner_id"] for item in train_pool) != matrix_partner_ids:
        raise ValueError("TBS cross-play order does not match the training pool")
    tomzsc_path = Path(project_root) / payload["tomzsc_path"]
    observed = subprocess.run(
        ["git", "-C", str(tomzsc_path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if observed != payload["tomzsc_commit"]:
        raise ValueError("TBS clustering repository does not match the pinned commit")
    labels = pinned_tbs_clusters(matrix, tomzsc_path, maximum_clusters=6)
    clusters = sorted(set(labels))
    primary_budget = int(payload["transitions"])
    component_budget = (
        primary_budget
        if payload.get("compute_allocation", "per-specialist") == "per-specialist"
        else max(primary_budget // len(clusters), 1)
    )
    specialist_results = []
    component_transitions = {}
    cluster_sequences = {}
    validation_sequences = {}
    for cluster_id in clusters:
        cluster_pool = tuple(
            item for item, label in zip(train_pool, labels, strict=True) if label == cluster_id
        )
        cluster_pool_path = output / "pools" / f"tbs-cluster-{cluster_id}.json"
        _write_pool(cluster_pool_path, cluster_pool)
        component_request = _component_request(
            request,
            train_pool_path=cluster_pool_path,
            transitions=component_budget,
            output_dir=output / "components" / f"specialist-{cluster_id}",
        )
        result = _run_ego_component(
            component_request,
            project_root,
            pipeline,
            output,
            method_id="tbs_style",
            component_id=f"specialist-{cluster_id}",
        )
        specialist_results.append(result)
        component_transitions[f"specialist-{cluster_id}"] = int(result["completed_transitions"])
        train_count = 2 if payload.get("smoke", False) else 256
        validation_count = 1 if payload.get("smoke", False) else 128
        cluster_sequences[cluster_id] = _collect_sequences(
            result["checkpoint_paths"][-1],
            cluster_pool,
            payload["layout_id"],
            episodes_per_partner=train_count,
            seed=int(payload["seed"]) + 1000 + cluster_id,
        )
        validation_sequences[cluster_id] = _collect_sequences(
            result["checkpoint_paths"][-1],
            validation_pool,
            payload["layout_id"],
            episodes_per_partner=validation_count,
            seed=int(payload["seed"]) + 2000 + cluster_id,
        )
    global_training = _concatenate_sequences(tuple(cluster_sequences.values()))
    global_validation = _concatenate_sequences(tuple(validation_sequences.values()))
    concept_schema = tuple(sorted(SUPPORTED_BEHAVIOR_EVENTS))
    global_path = output / "components" / "global-tom"
    if not _component_complete(pipeline, "global-tom"):
        train_visible_predictor(
            global_training[0],
            global_training[1],
            global_training[2],
            seed=int(payload["seed"]) + 3000,
            output_path=global_path,
            validation=(global_validation[0], global_validation[1], global_validation[2]),
            maximum_epochs=2 if payload.get("smoke", False) else 100,
        )
        _record_file_component(pipeline, output, "global-tom", global_path)
    cluster_tom_paths = []
    for cluster_id in clusters:
        path = output / "components" / f"cluster-tom-{cluster_id}"
        training = cluster_sequences[cluster_id]
        validation = validation_sequences[cluster_id]
        component_id = f"cluster-tom-{cluster_id}"
        if not _component_complete(pipeline, component_id):
            train_visible_predictor(
                training[0],
                training[1],
                training[2],
                seed=int(payload["seed"]) + 4000 + cluster_id,
                output_path=path,
                validation=(validation[0], validation[1], validation[2]),
                maximum_epochs=2 if payload.get("smoke", False) else 100,
            )
            _record_file_component(pipeline, output, component_id, path)
        cluster_tom_paths.append(path)
    result = specialist_results[0].copy()
    result["checkpoint_paths"] = [item["checkpoint_paths"][-1] for item in specialist_results]
    result["checkpoint_parameter_hashes"] = {
        key: value
        for item in specialist_results
        for key, value in item["checkpoint_parameter_hashes"].items()
    }
    result["component_transitions"] = component_transitions
    result["aggregate_training_transitions"] = sum(component_transitions.values())
    _apply_pipeline_resume_lineage(result, payload, resume_parent_hash)
    artifact = _base_artifact(request, result, "tbs_selector")
    artifact["cluster_assignments"] = {
        partner_id: int(label) for partner_id, label in zip(matrix_partner_ids, labels, strict=True)
    }
    artifact["concept_schema"] = list(concept_schema)
    artifact["components"] = [
        *[
            _component(
                f"specialist-{cluster_id}",
                "specialist",
                specialist_results[index]["checkpoint_paths"][-1],
                cluster_id,
            )
            for index, cluster_id in enumerate(clusters)
        ],
        _component("global-tom", "global_tom", global_path),
        *[
            _component(f"cluster-tom-{cluster_id}", "cluster_tom", path, cluster_id)
            for cluster_id, path in zip(clusters, cluster_tom_paths, strict=True)
        ],
    ]
    finished = _finish_artifact(payload, result, artifact)
    _complete_pipeline_state(pipeline, output, finished)
    return finished


def _train_csp(request, project_root):
    payload = request["payload"]
    output = Path(payload["output_dir"]).resolve()
    resume_parent_hash = _resume_parent_hash(payload.get("resume_checkpoint"), output)
    pipeline = _load_pipeline_state(request, output)
    train_pool = _load_pool(payload["train_pool_path"])
    validation_pool = _load_pool(payload["validation_pool_path"])
    bootstrap_count = (
        1 if payload.get("smoke", False) else max(1, int(np.ceil(64 / (3 * len(train_pool)))))
    )
    bootstrap = _collect_bootstrap_sequences(
        train_pool,
        payload["layout_id"],
        repetitions_per_policy=bootstrap_count,
        seed=int(payload["seed"]) + 4500,
    )
    bootstrap_path = output / "components" / "trajectory-model-bootstrap"
    if not _component_complete(pipeline, "trajectory-model-bootstrap"):
        train_csp_trajectory_model(
            bootstrap[0],
            bootstrap[3],
            bootstrap[4],
            bootstrap[5],
            seed=int(payload["seed"]) + 4600,
            output_path=bootstrap_path,
            maximum_epochs=2 if payload.get("smoke", False) else 100,
        )
        _record_file_component(pipeline, output, "trajectory-model-bootstrap", bootstrap_path)
    probe_request = _component_request(
        request,
        train_pool_path=Path(payload["train_pool_path"]),
        transitions=int(payload["transitions"]),
        output_dir=output / "components" / "probe-policy",
    )
    probe_request["payload"]["csp_model_path"] = str(bootstrap_path)
    probe_result = _run_ego_component(
        probe_request,
        project_root,
        pipeline,
        output,
        method_id="csp_style_reconnaissance",
        component_id="probe-policy",
    )
    train_count = 2 if payload.get("smoke", False) else 64
    validation_count = 1 if payload.get("smoke", False) else 32
    training = _collect_sequences(
        probe_result["checkpoint_paths"][-1],
        train_pool,
        payload["layout_id"],
        episodes_per_partner=train_count,
        seed=int(payload["seed"]) + 5000,
    )
    validation = _collect_sequences(
        probe_result["checkpoint_paths"][-1],
        validation_pool,
        payload["layout_id"],
        episodes_per_partner=validation_count,
        seed=int(payload["seed"]) + 6000,
    )
    trajectory_path = Path(probe_result["auxiliary_checkpoint_path"])
    embeddings = encode_csp_trajectories(trajectory_path, training[0], training[3], training[4])
    validation_embeddings = encode_csp_trajectories(
        trajectory_path, validation[0], validation[3], validation[4]
    )
    labels, centroids, _, _ = deterministic_kmeans(
        embeddings,
        validation_embeddings=validation_embeddings,
        maximum_clusters=min(6, len(train_pool) - 1),
        seed=int(payload["seed"]) + 8000,
    )
    partner_labels = _partner_majority_labels(labels, len(train_pool), train_count)
    trajectory_labels = np.asarray(labels).reshape((len(train_pool), train_count))
    # Every trajectory centroid is a possible deployment route, even when it
    # is not the majority behavior of any single training partner.
    clusters = tuple(range(len(centroids)))
    primary_budget = int(payload["transitions"])
    specialist_budget = (
        primary_budget
        if payload.get("compute_allocation", "per-specialist") == "per-specialist"
        else max(primary_budget // len(clusters), 1)
    )
    specialists = []
    component_transitions = {"probe-policy": int(probe_result["completed_transitions"])}
    for cluster_id in clusters:
        cluster_pool = tuple(
            item
            for item, row in zip(train_pool, trajectory_labels, strict=True)
            if bool(np.any(row == cluster_id))
        )
        if not cluster_pool:
            raise RuntimeError(f"CSP cluster {cluster_id} has no supporting training partner")
        pool_path = output / "pools" / f"csp-cluster-{cluster_id}.json"
        _write_pool(pool_path, cluster_pool)
        specialist_request = _component_request(
            request,
            train_pool_path=pool_path,
            transitions=specialist_budget,
            output_dir=output / "components" / f"specialist-{cluster_id}",
        )
        result = _run_ego_component(
            specialist_request,
            project_root,
            pipeline,
            output,
            method_id="csp_style_reconnaissance",
            component_id=f"specialist-{cluster_id}",
        )
        specialists.append(result)
        component_transitions[f"specialist-{cluster_id}"] = int(result["completed_transitions"])
    result = probe_result.copy()
    result["checkpoint_paths"] = [
        probe_result["checkpoint_paths"][-1],
        *[item["checkpoint_paths"][-1] for item in specialists],
    ]
    result["checkpoint_parameter_hashes"] = {
        **probe_result["checkpoint_parameter_hashes"],
        **{
            key: value
            for item in specialists
            for key, value in item["checkpoint_parameter_hashes"].items()
        },
    }
    result["component_transitions"] = component_transitions
    result["aggregate_training_transitions"] = sum(component_transitions.values())
    _apply_pipeline_resume_lineage(result, payload, resume_parent_hash)
    artifact = _base_artifact(request, result, "csp_reconnaissance")
    artifact["reconnaissance_episodes"] = 1
    artifact["centroids"] = centroids
    artifact["cluster_assignments"] = {
        item["partner_id"]: int(label)
        for item, label in zip(train_pool, partner_labels, strict=True)
    }
    artifact["components"] = [
        _component("probe-policy", "probe_policy", probe_result["checkpoint_paths"][-1]),
        _component("trajectory-encoder", "trajectory_encoder", trajectory_path),
        _component("response-decoder", "response_decoder", trajectory_path),
        *[
            _component(
                f"specialist-{cluster_id}",
                "specialist",
                specialists[index]["checkpoint_paths"][-1],
                cluster_id,
            )
            for index, cluster_id in enumerate(clusters)
        ],
    ]
    finished = _finish_artifact(payload, result, artifact)
    _complete_pipeline_state(pipeline, output, finished)
    return finished


def _collect_sequences(
    ego_checkpoint,
    pool,
    layout,
    *,
    episodes_per_partner,
    seed,
):
    ego = _load_policy(ego_checkpoint, stochastic=True)
    environment = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=400,
        agent_view_size=2,
        random_agent_positions=True,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
    )
    observation_shape = environment.observation_space().shape
    concept_schema = tuple(sorted(SUPPORTED_BEHAVIOR_EVENTS))
    episodes = []
    for partner_index, partner_spec in enumerate(pool):
        partner = _load_policy(partner_spec["checkpoint_path"], stochastic=True)
        for episode_index in range(episodes_per_partner):
            key = seed + partner_index * episodes_per_partner + episode_index
            records, _, _ = _episode(
                ego,
                partner,
                layout,
                key,
                partner_spec,
                {"evidence_policy": "ordinary_progress", "option_steps": 0},
            )
            episodes.append(_records_to_sequence(records, observation_shape, concept_schema))
    return tuple(np.stack([item[index] for item in episodes], axis=1) for index in range(6))


class _RandomValidPolicy:
    def init_hstate(self, batch_size, protocol_phase="scored"):
        del batch_size, protocol_phase
        return None

    def compute_action(self, observation, done, hidden, key, protocol_phase="scored"):
        del observation, done, protocol_phase
        return jax.random.randint(key, (), 0, 6), hidden, {}


def _collect_bootstrap_sequences(
    pool,
    layout,
    *,
    repetitions_per_policy,
    seed,
):
    """Balanced passive, random-valid, and ordinary task-active CSP warm-up."""

    environment = jaxmarl.make(
        "overcooked_v2",
        layout=layout,
        max_steps=400,
        agent_view_size=2,
        random_agent_positions=True,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
    )
    observation_shape = environment.observation_space().shape
    concept_schema = tuple(sorted(SUPPORTED_BEHAVIOR_EVENTS))
    passive = _load_policy(pool[0]["checkpoint_path"], stochastic=True)
    random_valid = _RandomValidPolicy()
    policies = (
        (passive, "ordinary_progress", 0),
        (random_valid, "ordinary_progress", 0),
        (passive, "stage_candidate_ingredient", 16),
    )
    episodes = []
    episode_number = 0
    for partner_index, partner_spec in enumerate(pool):
        partner = _load_policy(partner_spec["checkpoint_path"], stochastic=True)
        for policy, evidence_policy, option_steps in policies:
            for _ in range(repetitions_per_policy):
                key = seed + partner_index * 10_000 + episode_number
                episode_number += 1
                records, _, _ = _episode(
                    policy,
                    partner,
                    layout,
                    key,
                    partner_spec,
                    {
                        "evidence_policy": evidence_policy,
                        "option_steps": option_steps,
                    },
                )
                episodes.append(_records_to_sequence(records, observation_shape, concept_schema))
    return tuple(np.stack([item[index] for item in episodes], axis=1) for index in range(6))


def _records_to_sequence(records, observation_shape, concept_schema):
    horizon = 400
    observations = np.zeros((horizon, *observation_shape), dtype=np.float32)
    concepts = np.zeros((horizon, len(concept_schema)), dtype=np.float32)
    visible = np.zeros((horizon,), dtype=np.float32)
    ego_actions = np.zeros((horizon,), dtype=np.int32)
    rewards = np.zeros((horizon,), dtype=np.float32)
    partner_actions = np.full((horizon,), 6, dtype=np.int32)
    event_index = {name: index for index, name in enumerate(concept_schema)}
    for row in records[:horizon]:
        step = int(row["step"])
        observations[step] = np.asarray(row["ego_observation"], dtype=np.float32).reshape(
            observation_shape
        )
        ego_actions[step] = int(row["ego_action"])
        rewards[step] = float(row["reward"])
        if row["visible_partner_action"] is not None:
            visible[step] = 1.0
            partner_actions[step] = int(row["visible_partner_action"])
        for event in row["high_level_events"]:
            if event.startswith("partner:"):
                name = event.split(":", 1)[1]
                if name in event_index:
                    concepts[step, event_index[name]] = 1.0
    return observations, concepts, visible, ego_actions, rewards, partner_actions


def _concatenate_sequences(sequences):
    return tuple(np.concatenate([item[index] for item in sequences], axis=1) for index in range(6))


def _partner_majority_labels(labels, partner_count, episodes_per_partner):
    values = np.asarray(labels).reshape((partner_count, episodes_per_partner))
    return tuple(int(np.bincount(row).argmax()) for row in values)


def _load_pool(path):
    payload = json.loads(Path(path).read_text())
    checkpoints = payload.get("checkpoints", payload if isinstance(payload, list) else [])
    competent = tuple(item for item in checkpoints if item.get("competent", True))
    if not competent:
        raise ValueError(f"partner pool contains no competent checkpoints: {path}")
    return competent


def _write_pool(path, checkpoints):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"checkpoints": checkpoints}, indent=2, sort_keys=True) + "\n")


def _load_cross_play(path):
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        values = payload.get("values", payload.get("matrix"))
        partner_ids = tuple(payload.get("partner_ids", ()))
    else:
        values = payload
        partner_ids = ()
    matrix = np.asarray(values, dtype=np.float64)
    if not partner_ids:
        raise ValueError("TBS cross-play input requires an explicit partner_ids order")
    return matrix, partner_ids


def _component_request(request, *, train_pool_path, transitions, output_dir):
    payload = dict(request["payload"])
    payload.update(
        {
            "train_pool_path": str(Path(train_pool_path).resolve()),
            "transitions": int(transitions),
            "output_dir": str(Path(output_dir).resolve()),
            "resume_checkpoint": _resume_for_component(
                request["payload"].get("resume_checkpoint"), Path(output_dir).name
            ),
        }
    )
    return {**request, "payload": payload}


def _pipeline_fingerprint(request):
    payload = dict(request["payload"])
    payload.pop("resume_checkpoint", None)
    payload.pop("output_dir", None)
    stable = {
        "suite_id": request["suite_id"],
        "upstreams": request["upstreams"],
        "payload": payload,
        "runtime_source_hash": _runtime_source_hash(),
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _runtime_source_hash():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _resume_parent_hash(resume, output):
    state_path = _resolve_pipeline_state(resume, output)
    if state_path is not None:
        return hashlib.sha256(state_path.read_bytes()).hexdigest()
    if resume is None:
        return None
    path = Path(resume).resolve()
    return directory_hash(path) if path.is_dir() else hashlib.sha256(path.read_bytes()).hexdigest()


def _apply_pipeline_resume_lineage(result, payload, parent_hash):
    resume = payload.get("resume_checkpoint")
    if resume is None:
        return
    result["resumed"] = True
    result["resume_checkpoint_path"] = str(Path(resume).resolve())
    result["parent_checkpoint_hash"] = parent_hash


def _load_pipeline_state(request, output):
    fingerprint = _pipeline_fingerprint(request)
    state_path = _resolve_pipeline_state(request["payload"].get("resume_checkpoint"), output)
    if state_path is None:
        return {
            "schema_version": 1,
            "request_fingerprint": fingerprint,
            "components": {},
        }
    state = json.loads(state_path.read_text())
    if state.get("schema_version") != 1:
        raise ValueError("unsupported method pipeline checkpoint schema")
    if state.get("request_fingerprint") != fingerprint:
        raise ValueError("method pipeline checkpoint does not match the current request")
    if state.get("complete", False):
        raise ValueError("method pipeline already completed its registered target")
    return state


def _resolve_pipeline_state(resume, output):
    if resume is None:
        return None
    path = Path(resume).resolve()
    candidates = []
    if path.is_file():
        candidates.append(path if path.name == "pipeline-state.json" else path.parent)
    else:
        candidates.append(path)
    candidates.extend(path.parents)
    candidates.append(output)
    for candidate in candidates:
        state_path = (
            candidate
            if candidate.name == "pipeline-state.json"
            else candidate / "pipeline-state.json"
        )
        if state_path.is_file():
            return state_path
    return None


def _resume_for_component(resume, component_id):
    if resume is None:
        return None
    path = Path(resume).resolve()
    if component_id in path.parts or component_id in str(path):
        return str(path)
    return None


def _component_complete(pipeline, component_id):
    component = pipeline.get("components", {}).get(component_id)
    if not component:
        return False
    artifacts = component.get("artifacts", {})
    if not artifacts:
        return False
    return all(
        Path(path).is_dir() and directory_hash(path) == content_hash
        for path, content_hash in artifacts.items()
    )


def _run_ego_component(
    request,
    project_root,
    pipeline,
    output,
    *,
    method_id,
    component_id,
):
    if _component_complete(pipeline, component_id):
        return pipeline["components"][component_id]["result"]
    result = train_ego_method(
        request,
        project_root,
        method_id=method_id,
        component_id=component_id,
    )
    artifacts = {
        path: directory_hash(path)
        for path in (
            *result.get("checkpoint_paths", ()),
            *result.get("training_state_paths", ()),
            *(
                ()
                if result.get("best_validation_checkpoint_path") is None
                else (result["best_validation_checkpoint_path"],)
            ),
        )
    }
    pipeline["components"][component_id] = {
        "result": result,
        "artifacts": artifacts,
    }
    _write_pipeline_state(output, pipeline)
    return result


def _record_file_component(pipeline, output, component_id, path):
    pipeline["components"][component_id] = {
        "artifacts": {str(Path(path).resolve()): directory_hash(path)}
    }
    _write_pipeline_state(output, pipeline)


def _write_pipeline_state(output, state):
    destination = Path(output).resolve() / "pipeline-state.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def _complete_pipeline_state(pipeline, output, result):
    pipeline["complete"] = True
    pipeline["deployment_artifact_path"] = result["deployment_artifact_path"]
    pipeline["deployment_artifact_hash"] = result["deployment_artifact_hash"]
    _write_pipeline_state(output, pipeline)


def _base_artifact(request, result, policy_kind):
    payload = request["payload"]
    train_pool = _load_pool(payload["train_pool_path"])
    return {
        "schema_version": 1,
        "policy_kind": policy_kind,
        "method_id": payload["method_id"],
        "layout_id": payload["layout_id"],
        "seed": int(payload["seed"]),
        "backbone_config": result.get("policy_config", {}),
        "components": [],
        "partner_ids": [item["partner_id"] for item in train_pool],
        "cluster_assignments": {},
        "concept_schema": [],
        "centroids": [],
        "reconnaissance_episodes": 0,
        "source_configuration_hash": result["configuration_hash"],
        "aggregate_training_transitions": int(result["aggregate_training_transitions"]),
    }


def _component(component_id, role, path, cluster_id=None):
    result = {
        "component_id": component_id,
        "role": role,
        "path": str(Path(path).resolve()),
        "content_hash": directory_hash(path),
        "cluster_id": cluster_id,
    }
    return result


def _finish_artifact(payload, result, artifact):
    path, content_hash = write_deployment_artifact(
        Path(payload["output_dir"]).resolve() / "policy-artifact.json", artifact
    )
    result["deployment_artifact_path"] = str(path)
    result["deployment_artifact_hash"] = content_hash
    result["policy_kind"] = artifact["policy_kind"]
    return result
