"""Minimal audited extension of the pinned IPPO closure for exact continuation.

The upstream trainer already carries every state required for continuation in
``runner_state`` but exposes only ``initial_train_state``.  Reimplementing the
850-line PPO loop would make paper-faithfulness harder to audit.  Instead this
module verifies the pinned source hash and applies two narrow source edits:
accept an initial runner state and use it at the scan boundary.
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import sys
from collections.abc import Callable
from pathlib import Path

PINNED_IPPO_SOURCE_HASH = "da9076b7404ba1317a0804165a5e35c1170541fc277830591259296bcf3424fe"


def load_resumable_make_train() -> Callable:
    sys.modules.setdefault(
        "models", importlib.import_module("overcooked_v2_experiments.ppo.models")
    )
    from overcooked_v2_experiments.ppo import ippo

    source_file = inspect.getsourcefile(ippo.make_train)
    if source_file is None:
        raise RuntimeError("cannot locate pinned IPPO source")
    observed = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    if observed != PINNED_IPPO_SOURCE_HASH:
        raise RuntimeError("pinned IPPO source changed; resumable adapter must be re-audited")
    source = inspect.getsource(ippo.make_train)
    outer_signature = """def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
):
"""
    resumed_outer_signature = """def make_train(
    config,
    update_step_offset=None,
    update_step_num_overwrite=None,
    population_config=None,
    initialize_only=False,
):
"""
    if source.count(outer_signature) != 1:
        raise RuntimeError("could not identify the pinned make_train signature")
    source = source.replace(outer_signature, resumed_outer_signature)
    signature = """    def train(
        rng,
        population: Optional[Union[AbstractPolicy, core.FrozenDict[str, Any]]] = None,
        initial_train_state=None,
    ):
"""
    replacement = """    def train(
        rng,
        population: Optional[Union[AbstractPolicy, core.FrozenDict[str, Any]]] = None,
        initial_train_state=None,
        initial_runner_state=None,
    ):
"""
    if source.count(signature) != 1:
        raise RuntimeError("could not identify the pinned train closure signature")
    source = source.replace(signature, replacement)
    initialization = """        runner_state = (
            train_state,
            initial_checkpoints,
            env_state,
            obsv,
            jnp.zeros((model_config[\"NUM_ACTORS\"]), dtype=bool),
            initial_update_step,
            init_hstate,
            init_population_hstate,
            init_population_annealing_mask,
            init_fcp_pop_idxs,
            _rng,
        )
        num_update_steps = model_config[\"NUM_UPDATES\"]
"""
    resumed_initialization = """        fresh_runner_state = (
            train_state,
            initial_checkpoints,
            env_state,
            obsv,
            jnp.zeros((model_config[\"NUM_ACTORS\"]), dtype=bool),
            initial_update_step,
            init_hstate,
            init_population_hstate,
            init_population_annealing_mask,
            init_fcp_pop_idxs,
            _rng,
        )
        runner_state = (
            fresh_runner_state if initial_runner_state is None else initial_runner_state
        )
        num_update_steps = model_config[\"NUM_UPDATES\"]
"""
    if source.count(initialization) != 1:
        raise RuntimeError("could not identify the pinned runner-state initialization")
    source = source.replace(initialization, resumed_initialization)
    scan = """        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_update_steps
        )

        # jax.debug.print("Runner state {x}", x=runner_state)
"""
    resumable_scan = """        if initialize_only:
            return {"runner_state": runner_state, "metrics": None}
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, num_update_steps
        )

        # jax.debug.print("Runner state {x}", x=runner_state)
"""
    if source.count(scan) != 1:
        raise RuntimeError("could not identify the pinned update scan")
    source = source.replace(scan, resumable_scan)
    namespace = dict(vars(ippo))
    exec(compile(source, source_file, "exec"), namespace)
    return namespace["make_train"]


def pinned_ippo_source_hash() -> str:
    return PINNED_IPPO_SOURCE_HASH
