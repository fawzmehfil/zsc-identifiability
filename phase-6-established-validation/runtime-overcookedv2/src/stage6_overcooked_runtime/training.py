"""Official OvercookedV2 IPPO/Other-Play/FCP training adapter."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import jax
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stage6_overcooked_runtime.checkpointing import (
    restore_training_checkpoint,
    save_training_checkpoint,
    validate_completed_target,
    validate_resume_target,
)
from stage6_overcooked_runtime.events import BehaviorPreferenceWrapper
from stage6_overcooked_runtime.resumable_upstream import pinned_ippo_source_hash


def train_official_method(request, project_root):
    payload = request["payload"]
    method = payload["method_id"]
    experiment = {
        "partner_ippo": "rnn-sp",
        "rnn_ippo": "rnn-sp",
        "other_play": "rnn-op",
        "fcp": "rnn-fcp",
    }.get(method)
    if experiment is None:
        raise ValueError(
            f"{method} requires a Stage 6 method-specific adapter, not the official IPPO adapter"
        )
    upstream = Path(project_root) / request["upstreams"]["overcookedv2"]["path"]
    config_dir = upstream / "experiments/overcooked_v2_experiments/ppo/config"
    schedule_target = int(payload.get("schedule_target_transitions", payload["transitions"]))
    overrides = [
        f"+experiment={experiment}",
        f"+env={payload['layout_id']}",
        f"SEED={int(payload['seed'])}",
        "NUM_SEEDS=1",
        f"model.TOTAL_TIMESTEPS={schedule_target}",
        f"model.REW_SHAPING_HORIZON={schedule_target // 2}",
        f"model.LR={float(payload['learning_rate'])}",
        f"model.ENT_COEF={float(payload['entropy_coefficient'])}",
        "NUM_CHECKPOINTS=3",
    ]
    if bool(payload.get("smoke", False)):
        overrides.extend(
            (
                "model.NUM_ENVS=32",
                "model.NUM_STEPS=32",
                "model.NUM_MINIBATCHES=8",
                "model.UPDATE_EPOCHS=2",
                "NUM_CHECKPOINTS=2",
            )
        )
    if method == "fcp":
        overrides.append(f"+FCP={payload['population_path']}")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = OmegaConf.to_container(compose(config_name="base", overrides=overrides))
    output = Path(payload["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["RUN_BASE_DIR"] = output

    import jaxmarl
    import wandb

    original_make = jaxmarl.make
    preferences = payload.get("behavior_preferences", {})

    def wrapped_make(env_id, **kwargs):
        env = original_make(env_id, **kwargs)
        return BehaviorPreferenceWrapper(env, preferences) if preferences else env

    jaxmarl.make = wrapped_make
    wandb_mode = os.environ.get("WANDB_MODE", "disabled")
    wandb_run = wandb.init(
        project="zsc-identifiability-stage6",
        mode=wandb_mode,
        config=json.loads(json.dumps(config, default=str)),
        reinit=True,
    )
    configuration_hash = _configuration_hash(config)
    resume_path = payload.get("resume_checkpoint")
    resumed = resume_path is not None
    initial_runner_state = None
    completed = 0
    parent_checkpoint_hash = None
    resume_parent_hash = None
    if resumed:
        template = _single_run(config, initialize_only=True)
        template = jax.block_until_ready(template)
        initial_runner_state, resume_metadata, _resolved_resume, parent_checkpoint_hash = (
            restore_training_checkpoint(
                resume_path,
                expected={
                    "suite_id": request["suite_id"],
                    "method_id": method,
                    "layout_id": payload["layout_id"],
                    "seed": int(payload["seed"]),
                    "configuration_hash": configuration_hash,
                    "upstream_commit": request["upstreams"]["overcookedv2"]["commit"],
                    "upstream_source_hash": pinned_ippo_source_hash(),
                },
                target_state=template["runner_state"],
            )
        )
        resume_parent_hash = parent_checkpoint_hash
        validate_resume_target(resume_metadata, int(payload["transitions"]))
        completed = int(resume_metadata["completed_transitions"])
    try:
        result = None
        checkpoint_paths = []
        checkpoint_hashes = {}
        checkpoint_interval = int(payload.get("checkpoint_interval", 1_000_000))
        transitions_per_update = int(config["model"]["NUM_STEPS"]) * int(
            config["model"]["NUM_ENVS"]
        )
        attainable_target = (
            int(payload["transitions"]) // transitions_per_update
        ) * transitions_per_update
        if attainable_target <= completed:
            raise ValueError("requested target does not contain a new complete PPO update")
        while completed < attainable_target:
            chunk_target = min(
                attainable_target,
                (
                    (((completed // checkpoint_interval) + 1) * checkpoint_interval)
                    // transitions_per_update
                )
                * transitions_per_update,
            )
            if chunk_target <= completed:
                chunk_target = min(attainable_target, completed + transitions_per_update)
            result = _single_run(
                config,
                initial_runner_state=initial_runner_state,
                completed_transitions=completed,
                stop_transitions=chunk_target,
            )
            result = jax.block_until_ready(result)
            initial_runner_state = result["runner_state"]
            completed = chunk_target
            metadata = {
                "suite_id": request["suite_id"],
                "method_id": method,
                "layout_id": payload["layout_id"],
                "seed": int(payload["seed"]),
                "component_id": "task_policy",
                "completed_transitions": completed,
                "target_transitions": attainable_target,
                "configuration_hash": configuration_hash,
                "dataset_hashes": tuple(payload.get("dataset_hashes", ())),
                "upstream_commit": request["upstreams"]["overcookedv2"]["commit"],
                "upstream_source_hash": pinned_ippo_source_hash(),
                "device": _device_description(),
                "parent_checkpoint_hash": parent_checkpoint_hash,
                "exact_continuation": all(device.platform == "cpu" for device in jax.devices()),
            }
            checkpoint_path, checkpoint_hash = save_training_checkpoint(
                output / "training-state",
                initial_runner_state,
                metadata,
            )
            checkpoint_paths.append(str(checkpoint_path))
            checkpoint_hashes[str(checkpoint_path)] = checkpoint_hash
            parent_checkpoint_hash = checkpoint_hash
    finally:
        jaxmarl.make = original_make
        wandb_run.finish()
    if result is None:
        raise RuntimeError("training completed without producing a runner state")
    written, parameter_hashes = _export_policy_checkpoints(
        output, config, result["runner_state"]
    )
    return {
        "method_id": method,
        "layout_id": payload["layout_id"],
        "seed": int(payload["seed"]),
        "requested_transitions": int(payload["transitions"]),
        "completed_transitions": completed,
        "checkpoint_paths": written,
        "checkpoint_parameter_hashes": parameter_hashes,
        "device": _device_description(),
        "configuration_hash": configuration_hash,
        "resumed": resumed,
        "recovered": False,
        "resume_checkpoint_path": None if resume_path is None else str(resume_path),
        "parent_checkpoint_hash": resume_parent_hash,
        "training_state_paths": checkpoint_paths,
        "training_state_hashes": checkpoint_hashes,
        "policy_kind": "ppo",
        "component_transitions": {"task_policy": completed},
        "aggregate_training_transitions": completed,
    }


def recover_official_training(request, project_root):
    """Re-export a completed policy without executing an optimizer update."""

    payload = request["payload"]
    method = payload["method_id"]
    experiment = {
        "partner_ippo": "rnn-sp",
        "rnn_ippo": "rnn-sp",
        "other_play": "rnn-op",
        "fcp": "rnn-fcp",
    }.get(method)
    if experiment is None:
        raise ValueError(f"recovery-only export is unsupported for {method}")
    resume_path = payload.get("resume_checkpoint")
    if resume_path is None:
        raise ValueError("recovery-only export requires resume_checkpoint")
    upstream = Path(project_root) / request["upstreams"]["overcookedv2"]["path"]
    config_dir = upstream / "experiments/overcooked_v2_experiments/ppo/config"
    schedule_target = int(payload.get("schedule_target_transitions", payload["transitions"]))
    overrides = [
        f"+experiment={experiment}",
        f"+env={payload['layout_id']}",
        f"SEED={int(payload['seed'])}",
        "NUM_SEEDS=1",
        f"model.TOTAL_TIMESTEPS={schedule_target}",
        f"model.REW_SHAPING_HORIZON={schedule_target // 2}",
        f"model.LR={float(payload['learning_rate'])}",
        f"model.ENT_COEF={float(payload['entropy_coefficient'])}",
        "NUM_CHECKPOINTS=3",
    ]
    if bool(payload.get("smoke", False)):
        overrides.extend(
            (
                "model.NUM_ENVS=32",
                "model.NUM_STEPS=32",
                "model.NUM_MINIBATCHES=8",
                "model.UPDATE_EPOCHS=2",
                "NUM_CHECKPOINTS=2",
            )
        )
    if method == "fcp":
        overrides.append(f"+FCP={payload['population_path']}")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = OmegaConf.to_container(compose(config_name="base", overrides=overrides))
    output = Path(payload["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["RUN_BASE_DIR"] = output
    configuration_hash = _configuration_hash(config)

    import jaxmarl

    original_make = jaxmarl.make
    preferences = payload.get("behavior_preferences", {})

    def wrapped_make(env_id, **kwargs):
        env = original_make(env_id, **kwargs)
        return BehaviorPreferenceWrapper(env, preferences) if preferences else env

    jaxmarl.make = wrapped_make
    try:
        template = jax.block_until_ready(_single_run(config, initialize_only=True))
        runner_state, metadata, resolved, checkpoint_hash = restore_training_checkpoint(
            resume_path,
            expected={
                "suite_id": request["suite_id"],
                "method_id": method,
                "layout_id": payload["layout_id"],
                "seed": int(payload["seed"]),
                "configuration_hash": configuration_hash,
                "upstream_commit": request["upstreams"]["overcookedv2"]["commit"],
                "upstream_source_hash": pinned_ippo_source_hash(),
            },
            target_state=template["runner_state"],
        )
    finally:
        jaxmarl.make = original_make
    transitions_per_update = int(config["model"]["NUM_STEPS"]) * int(
        config["model"]["NUM_ENVS"]
    )
    attainable_target = (
        int(payload["transitions"]) // transitions_per_update
    ) * transitions_per_update
    validate_completed_target(metadata, attainable_target)
    written, parameter_hashes = _export_policy_checkpoints(
        output, config, runner_state, overwrite=True
    )
    return {
        "method_id": method,
        "layout_id": payload["layout_id"],
        "seed": int(payload["seed"]),
        "requested_transitions": int(payload["transitions"]),
        "completed_transitions": attainable_target,
        "checkpoint_paths": written,
        "checkpoint_parameter_hashes": parameter_hashes,
        "device": _device_description(),
        "configuration_hash": configuration_hash,
        "resumed": True,
        "recovered": True,
        "resume_checkpoint_path": str(resolved),
        "parent_checkpoint_hash": checkpoint_hash,
        "training_state_paths": [str(resolved)],
        "training_state_hashes": {str(resolved): checkpoint_hash},
        "policy_kind": "ppo",
        "component_transitions": {"task_policy": attainable_target},
        "aggregate_training_transitions": attainable_target,
    }


def _export_policy_checkpoints(output, config, runner_state, *, overwrite=False):
    train_states = runner_state[0]
    registered_checkpoints = runner_state[1]
    run_count = jax.tree_util.tree_flatten(train_states.params)[0][0].shape[0]
    written = []
    parameter_hashes = {}
    for run_number in range(run_count):
        checkpoint_count = jax.tree_util.tree_flatten(registered_checkpoints)[0][0].shape[1]
        for checkpoint_number in range(checkpoint_count):
            checkpoint_params = jax.tree_util.tree_map(
                lambda value, run=run_number, checkpoint=checkpoint_number: value[run, checkpoint],
                registered_checkpoints,
            )
            _store_checkpoint(
                output,
                config,
                checkpoint_params,
                run_number=run_number,
                checkpoint=checkpoint_number,
                final=False,
                overwrite=overwrite,
            )
            checkpoint_path = output / f"run_{run_number}" / f"ckpt_{checkpoint_number}"
            written.append(str(checkpoint_path))
            parameter_hashes[str(checkpoint_path)] = _parameter_tree_hash(checkpoint_params)
        params = jax.tree_util.tree_map(
            lambda value, run=run_number: value[run],
            train_states.params,
        )
        _store_checkpoint(
            output,
            config,
            params,
            run_number=run_number,
            checkpoint=0,
            final=True,
            overwrite=overwrite,
        )
        written.append(str(output / f"run_{run_number}" / "ckpt_final"))
        parameter_hashes[written[-1]] = _parameter_tree_hash(params)
    return written, parameter_hashes


def _device_description():
    devices = jax.devices()
    return ",".join(
        sorted(
            {
                f"{device.platform}:{getattr(device, 'device_kind', type(device).__name__)}"
                for device in devices
            }
        )
    )


def _configuration_hash(config):
    portable = copy.deepcopy(config)
    portable.pop("RUN_BASE_DIR", None)
    root = Path(__file__).resolve().parent
    local_sources = hashlib.sha256()
    for name in ("checkpointing.py", "resumable_upstream.py", "training.py"):
        path = root / name
        local_sources.update(name.encode())
        local_sources.update(path.read_bytes())
    payload = {
        "config": portable,
        "runtime_source_hash": local_sources.hexdigest(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _parameter_tree_hash(params):
    digest = hashlib.sha256()
    leaves, tree_definition = jax.tree_util.tree_flatten_with_path(params)
    digest.update(str(tree_definition).encode())
    for path, value in leaves:
        array = np.asarray(jax.device_get(value))
        digest.update(str(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _store_checkpoint(
    output, config, params, *, run_number, checkpoint, final, overwrite=False
):
    """Store an official PPO policy tree with a JSON-safe runtime config."""

    import orbax.checkpoint as ocp
    from flax.training import orbax_utils

    name = "ckpt_final" if final else f"ckpt_{checkpoint}"
    checkpoint_dir = output / f"run_{run_number}" / name
    if overwrite and checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    payload = {"config": _json_safe(config), "params": params}
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(payload)
    checkpointer.save(checkpoint_dir.resolve(), payload, save_args=save_args)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_json_safe(item) for item in value)
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _single_run(
    config,
    *,
    initial_runner_state=None,
    completed_transitions=0,
    stop_transitions=None,
    initialize_only=False,
):
    """Invoke the pinned official PPO core without its unused BC import.

    The upstream ``ppo.run`` module imports the optional ``overcooked_ai_py``
    human-data stack at module import time even when behavioral cloning is not
    requested.  That package is absent from the upstream dependency manifest.
    Stage 6 calls the same pinned ``make_train`` and mapping utilities directly,
    so no upstream source is patched and the learned backbone is unchanged.
    """

    import jax.numpy as jnp

    # The pinned module contains one script-style absolute import
    # (``from models.rnn``).  Alias its installed package before importing the
    # trainer instead of editing the immutable upstream checkout.
    sys.modules.setdefault(
        "models", importlib.import_module("overcooked_v2_experiments.ppo.models")
    )
    from overcooked_v2_experiments.ppo.policy import PPOParams
    from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints
    from overcooked_v2_experiments.ppo.utils.utils import get_num_devices
    from overcooked_v2_experiments.utils.utils import mini_batch_pmap

    # The upstream trainer enables global NaN debugging at import time.  Keep
    # the exact loss checks in smoke tests, but disable the very expensive JAX
    # debug re-execution path for registered training runs.
    jax.config.update("jax_debug_nans", False)
    from stage6_overcooked_runtime.resumable_upstream import load_resumable_make_train

    make_train = load_resumable_make_train()

    num_runs = int(config["NUM_SEEDS"])
    population = None
    population_config = None
    if "FCP" in config:
        population, population_config = _load_fcp_population(
            Path(config["FCP"]), load_all_checkpoints, PPOParams, jnp
        )
        num_runs = int(jax.tree_util.tree_flatten(population)[0][0].shape[0])
        population = population.params

    rngs = jax.random.split(jax.random.PRNGKey(int(config["SEED"])), num_runs)
    transitions_per_update = int(config["model"]["NUM_STEPS"]) * int(config["model"]["NUM_ENVS"])
    stop = int(config["model"]["TOTAL_TIMESTEPS"] if stop_transitions is None else stop_transitions)
    remaining_updates = (stop - int(completed_transitions)) // transitions_per_update
    if remaining_updates <= 0 and not initialize_only:
        raise ValueError("training chunk does not contain a complete PPO update")
    update_offset = int(completed_transitions) // transitions_per_update
    train = jax.jit(
        make_train(
            copy.deepcopy(config),
            update_step_offset=update_offset,
            update_step_num_overwrite=remaining_updates,
            population_config=population_config,
            initialize_only=initialize_only,
        )
    )
    extra = {} if population is None else {"population": population}
    if initial_runner_state is not None:
        extra["initial_runner_state"] = initial_runner_state
    return mini_batch_pmap(train, get_num_devices())(rngs, **extra)


def _load_fcp_population(population_dir, load_all_checkpoints, ppo_params_type, jnp):
    populations = []
    first_config = None
    for directory in sorted(population_dir.iterdir()):
        if not directory.is_dir() or "fcp_" not in directory.name:
            continue
        checkpoints, checkpoint_config = load_all_checkpoints(
            directory, final_only=False, skip_initial=True
        )
        params, _ = jax.tree_util.tree_flatten(
            checkpoints, is_leaf=lambda value: type(value) is ppo_params_type
        )
        if not params:
            continue
        populations.append(jax.tree_util.tree_map(lambda *values: jnp.stack(values), *params))
        if first_config is None:
            first_config = checkpoint_config
    if not populations or first_config is None:
        raise ValueError(f"no usable FCP checkpoints found under {population_dir}")
    return (
        jax.tree_util.tree_map(lambda *values: jnp.stack(values), *populations),
        first_config,
    )
