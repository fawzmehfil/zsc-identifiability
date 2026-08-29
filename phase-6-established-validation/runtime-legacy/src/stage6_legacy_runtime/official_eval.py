"""CPU-only inference adapter for pinned official ZSC-Eval checkpoints.

This module intentionally imports no trainer and exposes no optimizer path.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import random
import resource
import subprocess
import sys
import tempfile
from pathlib import Path


def run_official_operation(request):
    if request.get("policy_training_allowed") is not False:
        raise ValueError("official runtime accepts inference-only requests")
    source = Path(os.environ["ZSC_EVAL_SOURCE"]).resolve()
    pool = Path(os.environ["ZSC_EVAL_POLICY_POOL"]).resolve()
    _verify_source(source, request["repository_commit"])
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    os.environ["POLICY_POOL"] = str(pool)
    operation = request["operation"]
    payload = request["payload"]
    layout = request["layout_id"]
    if operation == "official_parity":
        result = _parity(layout, payload, pool, source)
    elif operation == "official_response_rollout":
        result = _pair_rollout(layout, payload, pool, kind="response")
    elif operation == "official_method_rollout":
        result = _pair_rollout(layout, payload, pool, kind="method")
    elif operation == "official_trace_rollout":
        result = _pair_rollout(layout, payload, pool, kind="trace")
    else:
        raise ValueError(f"unsupported official operation: {operation!r}")
    result["official_source_commit"] = request["repository_commit"]
    result["policy_pool_revision"] = request["policy_pool_revision"]
    result["dependency_versions"] = _dependency_versions()
    return result


def _pair_rollout(layout, payload, pool, kind):
    import numpy as np

    ego_path = Path(
        payload[
            "response_checkpoint_path"
            if kind == "response"
            else "method_checkpoint_path"
            if kind == "method"
            else "reference_checkpoint_path"
        ]
    ).resolve()
    partner_path = Path(payload["partner_checkpoint_path"]).resolve()
    _require_pool_asset(ego_path, pool)
    _require_pool_asset(partner_path, pool)
    ego_deterministic = bool(payload.get("deterministic", False))
    evidence_policy = payload.get("evidence_policy", "ordinary_progress")
    ego_kind = (
        "mlp"
        if kind == "response"
        else str(payload["policy_architecture"])
        if kind == "method"
        else "rnn"
    )
    if ego_kind not in {"mlp", "rnn"}:
        raise ValueError(f"unsupported official policy architecture: {ego_kind!r}")
    ego_policy, ego_args = _load_policy(ego_path, layout, pool_kind=ego_kind)
    partner_policy, _ = _load_policy(partner_path, layout, pool_kind="mlp")
    loaded = (ego_policy, ego_args, partner_policy)
    episodes = []
    event_totals = []
    for episode_index, environment_key in enumerate(payload["episode_keys"]):
        seat = episode_index % 2 if payload.get("balanced_seats", True) else 0
        episode = _run_episode(
            layout,
            ego_path,
            partner_path,
            int(environment_key),
            seat,
            ego_deterministic,
            False,
            evidence_policy,
            int(payload.get("maximum_option_steps", 16)),
            retain_history=kind in {"trace", "method"},
            retain_observations=kind == "trace",
            loaded=loaded,
        )
        episodes.append(episode)
        event_totals.append(episode.pop("ego_event_features"))
    returns = [float(episode["sparse_return"]) for episode in episodes]
    episode_lengths = [int(episode["steps_elapsed"]) for episode in episodes]
    feature_mean = np.mean(np.asarray(event_totals, dtype=np.float64), axis=0).tolist()
    base = {
        "status": "complete",
        "policy_training_performed": False,
        "device": "cpu",
        "layout_id": layout,
        "episode_returns": returns,
        "episode_lengths": episode_lengths,
        "mean_ego_event_features": feature_mean,
        "episodes": episodes if kind in {"trace", "method"} else [],
        "ego_deployment": "greedy" if ego_deterministic else "stochastic",
        "partner_deployment": "stochastic",
        "peak_memory_bytes": _peak_rss_bytes(),
    }
    if kind == "response":
        base["partner_id"] = _checkpoint_identity(partner_path, layout, partner=True)
        base["response_id"] = _checkpoint_identity(ego_path, layout, partner=False)
    elif kind == "method":
        base["partner_id"] = _checkpoint_identity(partner_path, layout, partner=True)
        base["method_id"] = payload["method_id"]
        base["method_seed"] = int(payload["method_seed"])
        base["deployment"] = payload["deployment"]
    else:
        base["partner_id"] = _checkpoint_identity(partner_path, layout, partner=True)
        base["evidence_policy"] = evidence_policy
        base["split"] = payload["split"]
    return base


def _run_episode(
    layout,
    ego_checkpoint,
    partner_checkpoint,
    environment_key,
    ego_seat,
    ego_deterministic,
    partner_deterministic,
    evidence_policy,
    maximum_option_steps,
    retain_history,
    retain_observations,
    loaded=None,
):
    import numpy as np
    import torch

    np.random.seed(environment_key % (2**32 - 1))
    random.seed(environment_key)
    torch.manual_seed(environment_key)
    if loaded is None:
        ego_kind = "mlp" if ego_checkpoint.name.endswith("_w1_actor.pt") else "rnn"
        ego_policy, ego_args = _load_policy(ego_checkpoint, layout, pool_kind=ego_kind)
        partner_policy, _ = _load_policy(partner_checkpoint, layout, pool_kind="mlp")
    else:
        ego_policy, ego_args, partner_policy = loaded
    env = _make_environment(layout, ego_args, environment_key)
    env.agent_idx = int(ego_seat)
    physical_ego = int(ego_seat)
    observation, _, available = env.reset()
    # The wrapper orders the controlled ego first regardless of physical seat.
    ego_policy.reset(1, 2)
    partner_policy.reset(1, 2)
    ego_policy.register_control_agent(0, 0)
    partner_policy.register_control_agent(0, 1)
    controller = _make_controller(layout, evidence_policy, env, ego_seat)
    sparse_return = 0.0
    commitment_step = None
    first_delivery_step = None
    intervention_completed_step = None
    controller_failed = False
    controller_active = controller is not None
    step_rows = []
    event_features = None
    previous_cumulative_features = None
    for step in range(400):
        ego_model_action = _policy_action(
            ego_policy,
            observation[0],
            available[0],
            (0, 0),
            ego_deterministic,
        )
        partner_action = _policy_action(
            partner_policy,
            observation[1],
            available[1],
            (0, 1),
            partner_deterministic,
        )
        ego_action = ego_model_action
        if controller_active and step < maximum_option_steps and not controller.done(env):
            ego_action = controller.action(env)
        elif controller_active and not controller.done(env) and step >= maximum_option_steps:
            controller_failed = True
            controller_active = False
        if controller_active and controller.done(env):
            intervention_completed_step = step
            controller_active = False
        previous_positions = tuple(player.position for player in env.base_env.state.players)
        partner_index = 1 - physical_ego
        previous_partner_orientation = env.base_env.state.players[partner_index].orientation
        previous_partner_object = _held_object_signature(env.base_env.state.players[partner_index])
        next_observation, _, reward, done, info, next_available = env.step(
            ((int(ego_action),), (int(partner_action),))
        )
        shaped = info.get("shaped_info_by_agent", ({}, {}))
        if event_features is None:
            event_features = np.zeros(len(shaped[physical_ego]), dtype=np.float64)
        shaped_keys = tuple(shaped[physical_ego])
        current_features = np.asarray(
            [float(shaped[physical_ego][key]) for key in shaped_keys], dtype=np.float64
        )
        if layout == "random3_m":
            prior_features = (
                np.zeros_like(current_features)
                if previous_cumulative_features is None
                else previous_cumulative_features
            )
            step_features = np.maximum(0.0, current_features - prior_features)
            previous_cumulative_features = current_features
        else:
            step_features = current_features
        event_features += step_features
        placement = sum(float(agent.get("PLACEMENT_IN_POT", 0)) for agent in shaped) > 0
        if placement and commitment_step is None:
            commitment_step = step
        sparse = _sparse_reward(info)
        sparse_return += sparse
        if sparse > 0 and first_delivery_step is None:
            first_delivery_step = step
        inferred_partner_action = _infer_partner_action(
            env,
            previous_positions[partner_index],
            previous_partner_orientation,
            previous_partner_object,
            partner_index,
        )
        inferred_events = [
            f"ego_event:{key}"
            for key, value in zip(shaped_keys, step_features)  # noqa: B905 - Python 3.9 runtime
            if float(value) > 0
        ]
        if retain_history:
            step_row = {
                "step": step,
                "ego_action": int(ego_action),
                "visible_partner_action": inferred_partner_action,
                "reward": sparse,
                "events": inferred_events,
            }
            if retain_observations:
                step_row["ego_observation"] = (
                    np.asarray(observation[0], dtype=np.float32).reshape(-1).tolist()
                )
            step_rows.append(step_row)
        observation = next_observation
        available = next_available
        if all(done):
            break
    if event_features is None:
        event_features = np.zeros(1, dtype=np.float64)
    result = {
        "environment_key": environment_key,
        "ego_seat": int(ego_seat),
        "sparse_return": sparse_return,
        "commitment_reached": commitment_step is not None,
        "commitment_step": commitment_step,
        "first_delivery_step": first_delivery_step,
        "intervention_completed_step": intervention_completed_step,
        "intervention_failed": controller_failed,
        "steps": step_rows,
        "observation_width": (
            len(step_rows[0]["ego_observation"]) if step_rows and retain_observations else 0
        ),
        "ego_event_features": event_features.tolist(),
        "steps_elapsed": step + 1,
    }
    env.close()
    return result


def _make_environment(layout, policy_args, environment_key):
    args = copy.deepcopy(policy_args)
    args.layout_name = layout
    args.episode_length = 400
    args.num_agents = 2
    args.algorithm_name = "population"
    args.algorithm_type = "co-play"
    args.agent0_policy_name = "ego"
    args.agent1_policy_name = "partner"
    args.random_index = False
    args.use_render = False
    args.store_traj = False
    args.use_wandb = False
    args.cuda = False
    args.n_training_threads = 1
    args.use_hsp = False
    args.use_phi = False
    args.reward_shaping_factor = 0.0
    args.initial_reward_shaping_factor = 0.0
    args.use_available_actions = True
    args.use_random_player_pos = False
    args.use_random_terrain_state = False
    args.num_initial_state = 5
    args.replay_return_threshold = 0.75
    args.random_start_prob = 0.0
    args.use_timestep_feature = bool(getattr(args, "use_timestep_feature", False))
    args.use_identity_feature = bool(getattr(args, "use_identity_feature", False))
    args.use_agent_policy_id = False
    args.old_dynamics = layout == "small_corridor"
    args.overcooked_version = "old" if args.old_dynamics else "new"
    run_dir = Path(tempfile.gettempdir()) / "zsc-identifiability-official-runtime"
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.old_dynamics:
        from zsceval.envs.overcooked.Overcooked_Env import Overcooked
    else:
        from zsceval.envs.overcooked_new.Overcooked_Env import Overcooked
    env = Overcooked(args, run_dir, rank=0, evaluation=True)
    env.seed(environment_key % (2**31 - 1))
    return env


def _load_policy(checkpoint, layout, pool_kind):
    import torch

    config_path = (
        Path(os.environ["ZSC_EVAL_POLICY_POOL"])
        / layout
        / "policy_config"
        / ("rnn_policy_config.pkl" if pool_kind == "rnn" else "mlp_policy_config.pkl")
    )
    with config_path.open("rb") as handle:
        config = list(pickle.load(handle))
    policy_args = config[0]
    from zsceval.algorithms.population.utils import EvalPolicy

    if policy_args.algorithm_name not in {"mappo", "rmappo"}:
        raise ValueError(
            f"unsupported official policy architecture: {policy_args.algorithm_name!r}"
        )
    if policy_args.use_single_network:
        raise ValueError("official audit does not support single-network training artifacts")
    from zsceval.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

    policy_class = R_MAPPOPolicy
    policy = policy_class(*config, device=torch.device("cpu"))
    policy.load_checkpoint({"actor": str(checkpoint)})
    policy.prep_rollout()
    return EvalPolicy(policy_args, policy), policy_args


def _policy_action(policy, observation, available, agent, deterministic):
    import numpy as np

    action = policy.step(
        np.asarray(observation)[None],
        [agent],
        deterministic=deterministic,
        available_actions=np.asarray(available)[None],
    )
    return int(np.asarray(action).reshape(-1)[0])


class _Controller:
    def __init__(self, period, physical_ego, maximum_stay=0):
        self.period = period
        self.physical_ego = physical_ego
        self.maximum_stay = maximum_stay
        self.calls = 0
        self.was_reset = False

    def _reset(self, env):
        if not self.was_reset and self.period is not None:
            mdp = _script_controller_mdp(env.base_env.mdp)
            self.period.reset(mdp, env.base_env.state, self.physical_ego)
        self.was_reset = True

    def action(self, env):
        self._reset(env)
        self.calls += 1
        if self.period is None:
            return 4  # Action.STAY in the official six-action mapping.
        mdp = _script_controller_mdp(env.base_env.mdp)
        action = self.period.step(mdp, env.base_env.state, self.physical_ego)
        return _action_index(env, action)

    def done(self, env):
        self._reset(env)
        if self.period is None:
            return self.calls >= self.maximum_stay
        mdp = _script_controller_mdp(env.base_env.mdp)
        return bool(self.period.done(mdp, env.base_env.state, self.physical_ego))


def _script_controller_mdp(mdp):
    """Bridge the pinned new-environment recipe-capacity rename.

    ZSC-Eval's bundled ``overcooked_new`` scripted controller still reads the
    legacy ``num_items_for_soup`` field, while its matching MDP exposes the
    same capacity as ``max_num_items_for_soup``.  Add the legacy read alias to
    the per-episode MDP instance only when a scripted diagnostic controller is
    used.  Ordinary official-policy inference never passes through this path.
    """

    legacy_value = getattr(mdp, "num_items_for_soup", None)
    current_value = getattr(mdp, "max_num_items_for_soup", None)
    if legacy_value is not None:
        if (
            not isinstance(legacy_value, int)
            or isinstance(legacy_value, bool)
            or legacy_value <= 0
        ):
            raise AttributeError(
                "scripted diagnostic controller requires a positive "
                "num_items_for_soup or max_num_items_for_soup capacity"
            )
        if current_value is not None and (
            not isinstance(current_value, int)
            or isinstance(current_value, bool)
            or current_value <= 0
        ):
            raise AttributeError(
                "scripted diagnostic controller requires a positive "
                "num_items_for_soup or max_num_items_for_soup capacity"
            )
        if current_value is not None and legacy_value != current_value:
            raise ValueError(
                "script-controller soup-capacity fields disagree: "
                f"num_items_for_soup={legacy_value!r}, "
                f"max_num_items_for_soup={current_value!r}"
            )
        return mdp
    if (
        not isinstance(current_value, int)
        or isinstance(current_value, bool)
        or current_value <= 0
    ):
        raise AttributeError(
            "scripted diagnostic controller requires a positive "
            "num_items_for_soup or max_num_items_for_soup capacity"
        )
    mdp.num_items_for_soup = current_value
    return mdp


def _make_controller(layout, option, env, ego_seat):
    if option == "ordinary_progress":
        return None
    if option == "corridor_yield":
        return _Controller(None, ego_seat, maximum_stay=1)
    module = (
        "zsceval.envs.overcooked.script_agent.script_period"
        if layout == "small_corridor"
        else "zsceval.envs.overcooked_new.script_agent.script_period"
    )
    periods = __import__(module, fromlist=["SCRIPT_PERIODS_CLASSES"])
    if option == "stage_onion":
        if hasattr(periods, "Pickup_Ingredient_and_Place_Random"):
            period = periods.Pickup_Ingredient_and_Place_Random(obj="onion")
        else:
            period = periods.SCRIPT_PERIODS_CLASSES["pickup_onion_and_place_random"]()
    elif option == "stage_tomato":
        if not hasattr(periods, "Pickup_Ingredient_and_Place_Random"):
            raise ValueError("tomato staging is unavailable in this official layout")
        period = periods.Pickup_Ingredient_and_Place_Random(obj="tomato")
    elif option == "temporary_role_takeover":
        period = periods.SCRIPT_PERIODS_CLASSES["pickup_object"]("onion")
    else:
        raise ValueError(f"unsupported official diagnostic option: {option!r}")
    return _Controller(period, ego_seat)


def _action_index(env, action):
    module = env.__class__.__module__
    prefix = module.split(".Overcooked_Env")[0]
    source_segment = ".src" if prefix.endswith("overcooked_new") else ""
    mdp = __import__(
        prefix + source_segment + ".overcooked_ai_py.mdp.actions",
        fromlist=["Action"],
    )
    return int(mdp.Action.ACTION_TO_INDEX[action])


def _held_object_signature(player):
    if not player.has_object():
        return None
    held = player.get_object()
    return (held.name, repr(getattr(held, "state", None)))


def _infer_partner_action(
    env,
    previous_position,
    previous_orientation,
    previous_object,
    partner_index,
):
    player = env.base_env.state.players[partner_index]
    next_position = player.position
    next_orientation = player.orientation
    if next_position != previous_position:
        delta = (
            next_position[0] - previous_position[0],
            next_position[1] - previous_position[1],
        )
        return _action_index(env, delta)
    if next_orientation != previous_orientation:
        return _action_index(env, next_orientation)
    if _held_object_signature(player) != previous_object:
        module = env.__class__.__module__
        prefix = module.split(".Overcooked_Env")[0]
        source_segment = ".src" if prefix.endswith("overcooked_new") else ""
        actions = __import__(
            prefix + source_segment + ".overcooked_ai_py.mdp.actions",
            fromlist=["Action"],
        )
        return int(actions.Action.ACTION_TO_INDEX[actions.Action.INTERACT])
    return None


def _sparse_reward(info):
    values = info.get("sparse_r_by_agent")
    if values is not None:
        return float(max(values))
    episode = info.get("episode")
    if episode and "ep_sparse_r" in episode:
        return float(episode["ep_sparse_r"])
    return 0.0


def _parity(layout, payload, pool, source):
    checkpoint = Path(payload["checkpoint_path"]).resolve()
    partner_checkpoint = Path(payload["partner_checkpoint_path"]).resolve()
    _require_pool_asset(checkpoint, pool)
    _require_pool_asset(partner_checkpoint, pool)
    architecture = str(payload["checkpoint_architecture"])
    if architecture not in {"mlp", "rnn"}:
        raise ValueError(f"unsupported parity policy architecture: {architecture!r}")
    ego_policy, ego_args = _load_policy(checkpoint, layout, pool_kind=architecture)
    partner_policy, _ = _load_policy(partner_checkpoint, layout, pool_kind="mlp")
    loaded = (ego_policy, ego_args, partner_policy)
    seat_hashes = {}
    for seat in (0, 1):
        runs = [
            _run_episode(
                layout,
                checkpoint,
                partner_checkpoint,
                int(payload["episode_keys"][0]),
                seat,
                True,
                True,
                "ordinary_progress",
                16,
                True,
                True,
                loaded,
            )
            for _ in range(2)
        ]
        hashes = [_hash_json(item) for item in runs]
        if hashes[0] != hashes[1]:
            raise RuntimeError(
                f"official deterministic evaluator replay was not reproducible in seat {seat}"
            )
        seat_hashes[str(seat)] = hashes[0]
    return {
        "status": "complete",
        "policy_training_performed": False,
        "device": "cpu",
        "layout_id": layout,
        "official_source_commit": _git_commit(source),
        "deterministic_replay_equal": True,
        "seat_observation_action_reward_hashes": seat_hashes,
        "observation_action_reward_hash": _hash_json(seat_hashes),
        "checkpoint_hash": _sha256(checkpoint),
        "policy_loader": "zsceval.algorithms.r_mappo.algorithm.R_MAPPOPolicy",
        "environment_class": "official_zsceval_overcooked",
        "peak_memory_bytes": _peak_rss_bytes(),
    }


def _checkpoint_identity(path, layout, partner):
    name = path.name
    if partner:
        import re

        match = re.search(r"(hsp\d+)_(mid|final)_w0_actor\.pt$", name)
        if match:
            return f"{layout}:{match.group(1)}:{match.group(2)}"
    if name.endswith("_w1_actor.pt"):
        import re

        match = re.search(r"(hsp\d+)_(mid|final)_w1_actor\.pt$", name)
        if match:
            return f"{layout}:{match.group(1)}:{match.group(2)}"
    return f"{layout}:{name}"


def _require_pool_asset(path, pool):
    if not path.is_file() or pool != path and pool not in path.parents:
        raise ValueError("official checkpoint is absent or outside the locked policy pool")


def _verify_source(source, expected):
    if _git_commit(source) != expected:
        raise ValueError("official ZSC-Eval source commit mismatch")


def _git_commit(source):
    return subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _peak_rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _dependency_versions():
    import gym
    import numpy
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "gym": gym.__version__,
    }
