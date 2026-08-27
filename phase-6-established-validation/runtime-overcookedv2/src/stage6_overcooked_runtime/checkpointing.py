"""Atomic full-state checkpoints and compact deployment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import jax
import numpy as np
import orbax.checkpoint as ocp
from flax.training import orbax_utils

CHECKPOINT_SCHEMA_VERSION = 2


def pytree_hash(tree) -> str:
    digest = hashlib.sha256()
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    for path, value in leaves:
        array = np.asarray(jax.device_get(value))
        # Do not hash the PyTreeDef repr.  Flax TrainState embeds function
        # objects in its static fields and their repr contains process-local
        # memory addresses.  Leaf paths, types, shapes and bytes are stable
        # across a fresh Python process and still detect structural changes.
        digest.update(str(path).encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def source_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_training_checkpoint(
    checkpoint_root: str | Path,
    state,
    metadata: dict,
    *,
    keep_latest: int = 2,
    is_best: bool = False,
) -> tuple[Path, str]:
    """Save a verified update-boundary state and atomically publish latest.json."""

    root = Path(checkpoint_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    completed = int(metadata["completed_transitions"])
    state_hash = pytree_hash(state)
    complete_metadata = {
        **metadata,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "state_tree_hash": state_hash,
    }
    destination = root / f"step-{completed}"
    if destination.exists():
        shutil.rmtree(destination)
    # Orbax cannot portably reconstruct PmapSharding from this pinned JAX
    # stack.  Persist host-valued leaves while preserving the exact PyTree;
    # target-aware restore below re-applies the current process topology.
    host_state = jax.device_get(state)
    payload = {"state": host_state, "metadata": complete_metadata}
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(payload)
    checkpointer.save(destination, payload, save_args=save_args)
    restored = checkpointer.restore(destination, item=payload)
    if pytree_hash(restored["state"]) != state_hash:
        raise RuntimeError("checkpoint verification failed after write")
    checkpoint_hash = directory_hash(destination)
    latest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_path": str(destination),
        "checkpoint_hash": checkpoint_hash,
        "completed_transitions": completed,
        "state_tree_hash": state_hash,
    }
    temporary = root / ".latest.json.tmp"
    temporary.write_text(json.dumps(latest, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, root / "latest.json")
    if is_best:
        best = {
            **latest,
            "validation_metric": metadata.get("validation_metric"),
        }
        temporary_best = root / ".best.json.tmp"
        temporary_best.write_text(json.dumps(best, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_best, root / "best.json")
    preserve = {destination}
    best_path = _published_checkpoint(root / "best.json")
    if best_path is not None:
        preserve.add(best_path)
    _prune_checkpoints(root, keep_latest=keep_latest, preserve=preserve)
    return destination, checkpoint_hash


def restore_training_checkpoint(
    checkpoint: str | Path,
    *,
    expected: dict | None = None,
    target_state=None,
):
    path = resolve_checkpoint_path(checkpoint)
    checkpointer = ocp.PyTreeCheckpointer()
    raw = checkpointer.restore(path, item=None)
    if not isinstance(raw, dict) or "state" not in raw or "metadata" not in raw:
        raise ValueError("training checkpoint does not contain full resumable state")
    metadata = raw["metadata"]
    if int(metadata.get("schema_version", -1)) != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported training checkpoint schema")
    if target_state is None:
        raise ValueError(
            "a freshly initialized target_state is required to restore the exact "
            "TrainState and environment-state PyTree types"
        )
    restored = checkpointer.restore(
        path,
        item={"state": target_state, "metadata": metadata},
    )
    if pytree_hash(restored["state"]) != metadata["state_tree_hash"]:
        raise ValueError("training checkpoint state hash mismatch")
    if (
        bool(metadata.get("exact_continuation", False))
        and metadata.get("device") != _device_description()
    ):
        raise ValueError("exact checkpoint continuation requires the original device topology")
    for key, value in (expected or {}).items():
        observed = metadata.get(key)
        if _canonical_metadata(observed) != _canonical_metadata(value):
            raise ValueError(f"resume checkpoint mismatch for {key}: {observed!r} != {value!r}")
    return restored["state"], metadata, path, directory_hash(path)


def resolve_checkpoint_path(checkpoint: str | Path) -> Path:
    path = Path(checkpoint).resolve()
    if path.is_file() and path.name == "latest.json":
        payload = json.loads(path.read_text())
        path = Path(payload["checkpoint_path"]).resolve()
    elif path.is_dir() and (path / "latest.json").is_file():
        payload = json.loads((path / "latest.json").read_text())
        path = Path(payload["checkpoint_path"]).resolve()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {path}")
    return path


def write_deployment_artifact(path: str | Path, payload: dict) -> tuple[Path, str]:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(serialized)
    os.replace(temporary, destination)
    return destination, hashlib.sha256(serialized.encode()).hexdigest()


def directory_hash(path: str | Path) -> str:
    root = Path(path).resolve()
    digest = hashlib.sha256()
    for item in sorted(root.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(root)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def validate_resume_target(metadata: dict, target_transitions: int) -> None:
    completed = int(metadata["completed_transitions"])
    if target_transitions <= completed:
        raise ValueError(
            f"resume target {target_transitions} must exceed completed transitions {completed}"
        )


def _prune_checkpoints(root: Path, *, keep_latest: int, preserve: set[Path]) -> None:
    checkpoints = sorted(
        (item for item in root.iterdir() if item.is_dir() and item.name.startswith("step-")),
        key=lambda item: int(item.name.split("-", 1)[1]),
    )
    keep = set(checkpoints[-keep_latest:]) | preserve
    for item in checkpoints:
        if item not in keep:
            shutil.rmtree(item)


def _published_checkpoint(index_path: Path) -> Path | None:
    if not index_path.is_file():
        return None
    candidate = Path(json.loads(index_path.read_text())["checkpoint_path"]).resolve()
    return candidate if candidate.is_dir() else None


def _canonical_metadata(value):
    if isinstance(value, dict):
        return {str(key): _canonical_metadata(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_metadata(item) for item in value]
    return value


def _device_description():
    return ",".join(
        sorted(
            {
                f"{device.platform}:{getattr(device, 'device_kind', type(device).__name__)}"
                for device in jax.devices()
            }
        )
    )
