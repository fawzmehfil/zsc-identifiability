"""JSON request dispatcher for the isolated OvercookedV2 runtime."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    request_path = Path(args.request).resolve()
    result_path = Path(args.result).resolve()
    request = json.loads(request_path.read_text())
    expected = request.pop("request_hash")
    observed = hashlib.sha256(
        json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed != expected:
        raise ValueError("runtime request hash mismatch")
    project_root = _project_root(request_path)
    _verify_upstreams(request, project_root)
    operation = request["operation"]
    if operation == "validate":
        payload = _validate_environment(request["payload"])
    elif operation in {"train_partner", "train_method"}:
        method = request["payload"]["method_id"]
        if method in {
            "tbs_style",
            "pace_aux",
            "pace_style",
            "csp_style_reconnaissance",
        }:
            from stage6_overcooked_runtime.method_pipeline import train_ported_method

            payload = train_ported_method(request, project_root)
        else:
            from stage6_overcooked_runtime.training import train_official_method

            payload = train_official_method(request, project_root)
    elif operation == "collect":
        from stage6_overcooked_runtime.collect import collect_traces

        payload = collect_traces(request)
    elif operation == "evaluate_pair":
        from stage6_overcooked_runtime.evaluation import evaluate_checkpoint_pair

        payload = evaluate_checkpoint_pair(request)
    else:
        raise ValueError(f"unsupported OvercookedV2 runtime operation: {operation!r}")
    result = {
        "schema_version": 1,
        "request_hash": expected,
        "operation": operation,
        "status": "complete",
        "python_version": platform.python_version(),
        "dependency_versions": _versions(),
        "payload": payload,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


def _validate_environment(payload):
    import jax
    import jax.numpy as jnp
    import jaxmarl
    import numpy as np
    from jaxmarl.environments.overcooked_v2.common import (
        Actions,
        DynamicObject,
        StaticObject,
    )

    from stage6_overcooked_runtime.controller import DiagnosticGoalController
    from stage6_overcooked_runtime.events import (
        BehaviorPreferenceWrapper,
        transition_event_features,
    )

    environment = jaxmarl.make(
        "overcooked_v2",
        layout=payload.get("layout_id", "demo_cook_simple"),
        max_steps=400,
        agent_view_size=2,
        random_agent_positions=True,
        negative_rewards=True,
        sample_recipe_on_delivery=True,
    )
    reset_key = jax.random.PRNGKey(0)
    step_key = jax.random.PRNGKey(1)
    observations, state = environment.reset(reset_key)
    stay_actions = {
        agent: jnp.asarray(Actions.stay, dtype=jnp.int32) for agent in environment.agents
    }
    _, next_state, base_rewards, dones, _ = environment.step(step_key, state, stay_actions)

    wrapped = BehaviorPreferenceWrapper(environment, {"idle": 0.5})
    _, wrapped_state = wrapped.reset(reset_key)
    _, _, wrapped_rewards, _, wrapped_info = wrapped.step(step_key, wrapped_state, stay_actions)
    sparse_reward_preserved = all(
        np.isclose(float(base_rewards[agent]), float(wrapped_rewards[agent]))
        for agent in environment.agents
    )
    behavior_shaping = wrapped_info["stage6_behavior_shaping"]
    idle_features = wrapped_info["stage6_behavior_features"]["idle"]
    wrapper_checks_passed = (
        sparse_reward_preserved
        and all(np.isclose(float(behavior_shaping[agent]), 0.5) for agent in environment.agents)
        and np.allclose(np.asarray(idle_features), np.ones(environment.num_agents))
    )
    commitment_check = _validate_commitment_detection(
        environment,
        Actions,
        DynamicObject,
        StaticObject,
        DiagnosticGoalController,
        transition_event_features,
        jax,
        jnp,
        np,
    )
    method_port_checks = _validate_method_ports()
    return {
        "layout_id": payload.get("layout_id", "demo_cook_simple"),
        "agents": list(environment.agents),
        "observation_shapes": {key: list(value.shape) for key, value in observations.items()},
        "state_time": int(state.time),
        "next_state_time": int(next_state.time),
        "terminated_after_one_step": bool(dones["__all__"]),
        "sparse_reward_preserved": bool(sparse_reward_preserved),
        "behavior_wrapper_checks_passed": bool(wrapper_checks_passed),
        "commitment_check": commitment_check,
        "method_port_checks": method_port_checks,
        "settings_match": bool(
            int(state.time) == 0
            and int(next_state.time) == 1
            and not bool(dones["__all__"])
            and wrapper_checks_passed
            and commitment_check["passed"]
            and method_port_checks["passed"]
        ),
    }


def _validate_method_ports():
    import numpy as np

    from stage6_overcooked_runtime.ported_methods import (
        csp_probe_reward,
        pace_bonus_weight,
        select_csp_cluster,
        select_tbs_cluster,
        tbs_similarity_matrix,
    )
    from stage6_overcooked_runtime.resumable_upstream import load_resumable_make_train

    similarity = tbs_similarity_matrix(np.asarray([[1.0, 0.2], [0.2, 1.0]]))
    checks = {
        "pace_schedule": np.isclose(pace_bonus_weight(0, 300), 0.2)
        and np.isclose(pace_bonus_weight(250, 300), 0.0),
        "tbs_similarity": np.allclose(similarity, [[1.0, 0.2], [0.2, 1.0]]),
        "tbs_selector": select_tbs_cluster([[0.8, 0.2]], [[[0.8, 0.2]], [[0.2, 0.8]]]) == 0,
        "csp_reward": np.isclose(float(csp_probe_reward(1.0, 2.0)), 1.2),
        "csp_selector": select_csp_cluster([0, 0], [[0, 0], [2, 2]]) == 0,
        "resumable_source_adapter": callable(load_resumable_make_train()),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {"passed": all(checks.values()), **checks}


def _validate_commitment_detection(
    environment,
    actions_enum,
    dynamic_object,
    static_object,
    controller_type,
    feature_function,
    jax,
    jnp,
    np,
):
    """Reach a real pot placement and verify the event is the count increase."""

    key = jax.random.PRNGKey(7)
    key, reset_key = jax.random.split(key)
    _, state = environment.reset(reset_key)
    controller = controller_type("temporary_role_takeover", candidate_ingredient=0)
    for step in range(100):
        inventory = int(np.asarray(state.agents.inventory)[0])
        target = state.grid[:, :, 0] == (
            static_object.ingredient_pile(0)
            if inventory == dynamic_object.EMPTY
            else static_object.POT
        )
        ego_action = controller._approach_and_interact(state, target)
        action_map = {
            "agent_0": jnp.asarray(ego_action, dtype=jnp.int32),
            "agent_1": jnp.asarray(actions_enum.stay, dtype=jnp.int32),
        }
        key, step_key = jax.random.split(key)
        _, next_state, rewards, _, _ = environment.step(step_key, state, action_map)
        features = feature_function(environment, state, next_state, action_map, rewards)
        pot_event = float(np.asarray(features["pot_fill"])[0]) > 0
        old_count = _pot_ingredient_count(state, static_object, dynamic_object, np)
        new_count = _pot_ingredient_count(next_state, static_object, dynamic_object, np)
        if pot_event:
            return {
                "passed": new_count == old_count + 1,
                "step": step,
                "old_pot_ingredient_count": old_count,
                "new_pot_ingredient_count": new_count,
            }
        state = next_state
    return {
        "passed": False,
        "step": None,
        "old_pot_ingredient_count": None,
        "new_pot_ingredient_count": None,
    }


def _pot_ingredient_count(state, static_object, dynamic_object, np):
    pot_mask = state.grid[:, :, 0] == static_object.POT
    pot_contents = np.asarray(state.grid[:, :, 1])[np.asarray(pot_mask)].ravel()
    return int(
        sum(int(np.asarray(dynamic_object.ingredient_count(value))) for value in pot_contents)
    )


def _verify_upstreams(request, project_root):
    for repository in request["upstreams"].values():
        path = (project_root / repository["path"]).resolve()
        observed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if observed != repository["commit"]:
            raise ValueError(f"upstream pin mismatch at {path}")


def _project_root(request_path):
    for parent in request_path.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "src/zsc_identifiability").is_dir():
            return parent
    raise ValueError("could not locate zsc-identifiability project root")


def _versions():
    result = {}
    for name in (
        "flax",
        "jax",
        "jaxlib",
        "jaxmarl",
        "numpy",
        "optax",
        "orbax-checkpoint",
        "overcooked-v2-experiments",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


if __name__ == "__main__":
    main()
