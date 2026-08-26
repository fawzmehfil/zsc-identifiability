"""Official OvercookedV2 IPPO/Other-Play/FCP training adapter."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

import jax
import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stage6_overcooked_runtime.events import BehaviorPreferenceWrapper


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
    overrides = [
        f"+experiment={experiment}",
        f"+env={payload['layout_id']}",
        f"SEED={int(payload['seed'])}",
        "NUM_SEEDS=1",
        f"model.TOTAL_TIMESTEPS={int(payload['transitions'])}",
        f"model.REW_SHAPING_HORIZON={int(payload['transitions']) // 2}",
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
    try:
        result = _single_run(config)
        result = jax.block_until_ready(result)
    finally:
        jaxmarl.make = original_make
        wandb_run.finish()
    checkpoints = result["runner_state"][1]
    run_count = jax.tree_util.tree_flatten(checkpoints)[0][0].shape[0]
    checkpoint_count = int(config["NUM_CHECKPOINTS"])
    written = []
    parameter_hashes = {}
    for run_number in range(run_count):
        for checkpoint in range(checkpoint_count):
            params = jax.tree_util.tree_map(
                lambda value, run=run_number, item=checkpoint: value[run][item],
                checkpoints,
            )
            _store_checkpoint(
                output,
                config,
                params,
                run_number=run_number,
                checkpoint=checkpoint,
                final=checkpoint == checkpoint_count - 1,
            )
            checkpoint_name = (
                "ckpt_final"
                if checkpoint == checkpoint_count - 1
                else f"ckpt_{checkpoint}"
            )
            written.append(str(output / f"run_{run_number}" / checkpoint_name))
            parameter_hashes[written[-1]] = _parameter_tree_hash(params)
    return {
        "method_id": method,
        "layout_id": payload["layout_id"],
        "seed": int(payload["seed"]),
        "requested_transitions": int(payload["transitions"]),
        "completed_transitions": int(payload["transitions"]),
        "checkpoint_paths": written,
        "checkpoint_parameter_hashes": parameter_hashes,
        "device": _device_description(),
        "configuration_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }


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


def _store_checkpoint(output, config, params, *, run_number, checkpoint, final):
    """Store an official PPO policy tree with a JSON-safe runtime config."""

    import orbax.checkpoint as ocp
    from flax.training import orbax_utils

    name = "ckpt_final" if final else f"ckpt_{checkpoint}"
    checkpoint_dir = output / f"run_{run_number}" / name
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


def _single_run(config):
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
    from overcooked_v2_experiments.ppo.ippo import make_train
    from overcooked_v2_experiments.ppo.policy import PPOParams
    from overcooked_v2_experiments.ppo.utils.store import load_all_checkpoints
    from overcooked_v2_experiments.ppo.utils.utils import get_num_devices
    from overcooked_v2_experiments.utils.utils import mini_batch_pmap

    # The upstream trainer enables global NaN debugging at import time.  Keep
    # the exact loss checks in smoke tests, but disable the very expensive JAX
    # debug re-execution path for registered training runs.
    jax.config.update("jax_debug_nans", False)

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
    train = jax.jit(make_train(copy.deepcopy(config), population_config=population_config))
    extra = {} if population is None else {"population": population}
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
