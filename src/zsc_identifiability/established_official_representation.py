"""Frozen history representations for the Stage 6 v3 decision-risk audit."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zsc_identifiability.established_gru import GRUSequenceBatchSource
from zsc_identifiability.established_official_decision import signed_hash_event_features
from zsc_identifiability.established_official_trace_store import (
    CompactTraceEpisode,
    IndexedTraceSequenceSource,
    SparseTraceSequenceSource,
)


@dataclass(frozen=True)
class FrozenIdentityEncoder:
    checkpoint_path: Path
    checkpoint_hash: str
    feature_width: int
    hidden_size: int
    mode_count: int
    seed: int
    validation_cross_entropy: float
    epochs_completed: int


@dataclass(frozen=True)
class EncodedHistories:
    embeddings: np.ndarray
    identity_logits: np.ndarray
    labels: np.ndarray
    row_indices: np.ndarray


def deterministic_stratified_calibration_split(
    source: SparseTraceSequenceSource,
    *,
    fraction: float = 0.75,
    salt: str,
) -> tuple[IndexedTraceSequenceSource, IndexedTraceSequenceSource]:
    """Split every partner's calibration rows by a stable environment-key hash."""

    if fraction <= 0 or fraction >= 1:
        raise ValueError("calibration training fraction must lie strictly inside (0, 1)")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(source.labels):
        groups.setdefault(int(label), []).append(index)
    train: list[int] = []
    early_stop: list[int] = []
    for label, indices in sorted(groups.items()):
        if len(indices) < 2:
            raise ValueError("each partner needs at least two calibration histories")
        ordered = sorted(
            indices,
            key=lambda index: hashlib.sha256(
                f"{salt}:{label}:{source.episodes[index].environment_key}".encode()
            ).hexdigest(),
        )
        training_count = min(len(ordered) - 1, max(1, int(np.floor(fraction * len(ordered)))))
        train.extend(ordered[:training_count])
        early_stop.extend(ordered[training_count:])
    return source.subset(sorted(train)), source.subset(sorted(early_stop))


def fit_gru_identity_representation(
    training: GRUSequenceBatchSource,
    early_stopping: GRUSequenceBatchSource,
    *,
    mode_count: int,
    hidden_size: int,
    seed: int,
    signature: str,
    checkpoint_path: str | Path,
    learning_rate: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 12,
    batch_size: int = 16,
) -> FrozenIdentityEncoder:
    """Train an identity objective only to freeze a reusable history encoder."""

    torch, nn = _torch_modules()
    if training.size < 2 or early_stopping.size < 2:
        raise ValueError("identity representation requires train and early-stopping histories")
    if training.feature_width != early_stopping.feature_width:
        raise ValueError("identity representation sources have inconsistent widths")
    if mode_count < 2 or hidden_size < 1:
        raise ValueError("identity representation dimensions must be positive")
    target = Path(checkpoint_path).resolve()
    if target.is_file():
        payload = torch.load(target, map_location="cpu", weights_only=False)
        if payload.get("signature") != signature:
            raise ValueError("existing identity encoder belongs to a different frozen unit")
        return FrozenIdentityEncoder(
            checkpoint_path=target,
            checkpoint_hash=_sha256(target),
            feature_width=int(payload["feature_width"]),
            hidden_size=int(payload["hidden_size"]),
            mode_count=int(payload["mode_count"]),
            seed=int(payload["seed"]),
            validation_cross_entropy=float(payload["validation_cross_entropy"]),
            epochs_completed=int(payload["epochs_completed"]),
        )
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    model = _make_model(torch, nn, training.feature_width, hidden_size, mode_count)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale = 0
    epochs_completed = 0
    for epoch in range(max_epochs):
        model.train()
        for features, lengths, labels, _indices in training.iter_batches(
            batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            optimizer.zero_grad(set_to_none=True)
            _embedding, logits = model(
                torch.from_numpy(features),
                torch.from_numpy(lengths.astype(np.int64, copy=False)),
            )
            loss = loss_fn(logits, torch.from_numpy(labels.astype(np.int64, copy=False)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = encode_with_identity_model(
            model,
            early_stopping,
            batch_size=batch_size,
            torch=torch,
        )
        validation_loss = float(
            loss_fn(
                torch.from_numpy(validation.identity_logits),
                torch.from_numpy(validation.labels),
            ).item()
        )
        epochs_completed = epoch + 1
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("identity representation training produced no checkpoint")
    payload = {
        "schema_version": 1,
        "signature": signature,
        "feature_width": training.feature_width,
        "hidden_size": hidden_size,
        "mode_count": mode_count,
        "seed": seed,
        "validation_cross_entropy": best_loss,
        "epochs_completed": epochs_completed,
        "state_dict": best_state,
    }
    _atomic_torch_save(target, payload, torch)
    return FrozenIdentityEncoder(
        checkpoint_path=target,
        checkpoint_hash=_sha256(target),
        feature_width=training.feature_width,
        hidden_size=hidden_size,
        mode_count=mode_count,
        seed=seed,
        validation_cross_entropy=best_loss,
        epochs_completed=epochs_completed,
    )


def encode_with_frozen_identity_representation(
    encoder: FrozenIdentityEncoder,
    source: GRUSequenceBatchSource,
    *,
    signature: str,
    batch_size: int = 16,
) -> EncodedHistories:
    torch, nn = _torch_modules()
    payload = torch.load(encoder.checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("signature") != signature or _sha256(encoder.checkpoint_path) != (
        encoder.checkpoint_hash
    ):
        raise ValueError("frozen identity encoder integrity check failed")
    model = _make_model(
        torch,
        nn,
        encoder.feature_width,
        encoder.hidden_size,
        encoder.mode_count,
    )
    model.load_state_dict(payload["state_dict"])
    return encode_with_identity_model(model, source, batch_size=batch_size, torch=torch)


def load_frozen_identity_encoder(
    checkpoint_path: str | Path,
    expected_hash: str,
) -> FrozenIdentityEncoder:
    """Load compact encoder metadata after verifying its immutable checkpoint hash."""

    path = Path(checkpoint_path).resolve()
    if not path.is_file() or _sha256(path) != expected_hash:
        raise ValueError("frozen identity encoder hash mismatch")
    torch, _nn = _torch_modules()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return FrozenIdentityEncoder(
        checkpoint_path=path,
        checkpoint_hash=expected_hash,
        feature_width=int(payload["feature_width"]),
        hidden_size=int(payload["hidden_size"]),
        mode_count=int(payload["mode_count"]),
        seed=int(payload["seed"]),
        validation_cross_entropy=float(payload["validation_cross_entropy"]),
        epochs_completed=int(payload["epochs_completed"]),
    )


def encode_with_identity_model(
    model: Any,
    source: GRUSequenceBatchSource,
    *,
    batch_size: int,
    torch: Any,
) -> EncodedHistories:
    model.eval()
    embeddings: list[np.ndarray | None] = [None] * source.size
    logits: list[np.ndarray | None] = [None] * source.size
    labels_by_row = np.empty(source.size, dtype=np.int64)
    with torch.no_grad():
        for features, lengths, labels, indices in source.iter_batches(
            batch_size,
            shuffle=False,
            seed=0,
        ):
            hidden, output = model(
                torch.from_numpy(features),
                torch.from_numpy(lengths.astype(np.int64, copy=False)),
            )
            hidden_array = hidden.detach().cpu().numpy()
            output_array = output.detach().cpu().numpy()
            for local, global_index in enumerate(indices):
                index = int(global_index)
                embeddings[index] = hidden_array[local].copy()
                logits[index] = output_array[local].copy()
                labels_by_row[index] = int(labels[local])
    if any(value is None for value in embeddings) or any(value is None for value in logits):
        raise RuntimeError("identity encoder omitted one or more history rows")
    return EncodedHistories(
        embeddings=np.stack([value for value in embeddings if value is not None]),
        identity_logits=np.stack([value for value in logits if value is not None]),
        labels=labels_by_row,
        row_indices=np.arange(source.size, dtype=np.int64),
    )


def event_representation_matrix(
    episodes: Sequence[CompactTraceEpisode],
    prefix: int | str,
    *,
    width: int = 512,
    salt: str = "zsc-dri-v3-event-features",
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for episode in episodes:
        observed_length, cumulative_reward, visibility_rate = episode.history_scalars(prefix)
        rows.append(
            signed_hash_event_features(
                episode.timed_event_tokens(prefix),
                observed_length=observed_length,
                cumulative_reward=cumulative_reward,
                partner_visibility_rate=visibility_rate,
                width=width,
                salt=salt,
            )
        )
    if not rows:
        raise ValueError("event representation requires at least one history")
    return np.stack(rows)


def save_encoded_histories(
    path: str | Path,
    encoded: EncodedHistories,
    *,
    environment_keys: Sequence[int],
    partner_ids: Sequence[str],
    commitment_reached: Sequence[bool],
) -> str:
    target = Path(path).resolve()
    if (
        len(environment_keys) != len(encoded.labels)
        or len(partner_ids) != len(encoded.labels)
        or len(commitment_reached) != len(encoded.labels)
    ):
        raise ValueError("encoded-history metadata must align with representation rows")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            embeddings=encoded.embeddings.astype(np.float32),
            identity_logits=encoded.identity_logits.astype(np.float32),
            labels=encoded.labels.astype(np.int32),
            environment_keys=np.asarray(environment_keys, dtype=np.int64),
            partner_ids=np.asarray(partner_ids, dtype=np.str_),
            commitment_reached=np.asarray(commitment_reached, dtype=np.bool_),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return _sha256(target)


def load_encoded_histories(
    path: str | Path,
) -> tuple[EncodedHistories, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        encoded = EncodedHistories(
            embeddings=data["embeddings"].astype(np.float64),
            identity_logits=data["identity_logits"].astype(np.float64),
            labels=data["labels"].astype(np.int64),
            row_indices=np.arange(len(data["labels"]), dtype=np.int64),
        )
        return (
            encoded,
            data["environment_keys"].copy(),
            data["partner_ids"].copy(),
            data["commitment_reached"].copy(),
        )


def representation_source_hashes(
    sources: Mapping[str, SparseTraceSequenceSource],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for name, source in sources.items():
        rows = sorted(
            (source.episodes[index].partner_id, source.episodes[index].environment_key)
            for index in range(source.size)
        )
        output[name] = hashlib.sha256(repr(rows).encode()).hexdigest()
    return output


def _make_model(torch: Any, nn: Any, width: int, hidden_size: int, mode_count: int) -> Any:
    class IdentityGRU(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(width, hidden_size, batch_first=True)
            self.identity_head = nn.Linear(hidden_size, mode_count)

        def forward(self, batch: Any, lengths: Any) -> tuple[Any, Any]:
            packed = nn.utils.rnn.pack_padded_sequence(
                batch,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            _output, hidden = self.gru(packed)
            representation = hidden[-1]
            return representation, self.identity_head(representation)

    return IdentityGRU()


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as error:  # pragma: no cover - optional established extra
        raise RuntimeError("v3 GRU measurement requires: uv sync --extra established") from error
    return torch, nn


def _atomic_torch_save(path: Path, payload: Mapping[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(raw_path)
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
