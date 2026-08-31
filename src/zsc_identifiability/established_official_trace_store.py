"""Bounded-memory compact storage for the Stage 6 official trace audit."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from zsc_identifiability.established_official_models import (
    OfficialTraceIndex,
    OfficialTraceIndexEntry,
)

_CACHE_SCHEMA_VERSION = 1
_MISSING_STEP = -1


@dataclass(frozen=True)
class CompactTraceCacheEntry:
    trace_id: str
    layout_id: str
    partner_id: str
    evidence_policy: str
    split: str
    source_hash: str
    cache_path: Path
    cache_hash: str
    episodes: int
    observation_width: int


@dataclass(frozen=True)
class _CompactShard:
    entry: CompactTraceCacheEntry
    step_offsets: np.ndarray
    observation_offsets: np.ndarray
    observation_indices: np.ndarray
    observation_values: np.ndarray
    ego_actions: np.ndarray
    partner_actions: np.ndarray
    rewards: np.ndarray
    step_numbers: np.ndarray
    event_offsets: np.ndarray
    event_ids: np.ndarray
    event_vocab: tuple[str, ...]
    environment_keys: np.ndarray
    sparse_returns: np.ndarray
    commitment_steps: np.ndarray
    first_delivery_steps: np.ndarray
    intervention_steps: np.ndarray
    ego_seats: np.ndarray


@dataclass(frozen=True)
class CompactTraceEpisode:
    """One trace without its large dense observation vectors."""

    shard: _CompactShard
    episode_index: int
    partner_id: str

    @property
    def environment_key(self) -> int:
        return int(self.shard.environment_keys[self.episode_index])

    @property
    def sparse_return(self) -> float:
        return float(self.shard.sparse_returns[self.episode_index])

    @property
    def ego_seat(self) -> int:
        return int(self.shard.ego_seats[self.episode_index])

    @property
    def commitment_step(self) -> int | None:
        value = int(self.shard.commitment_steps[self.episode_index])
        return None if value == _MISSING_STEP else value

    @property
    def first_delivery_step(self) -> int | None:
        value = int(self.shard.first_delivery_steps[self.episode_index])
        return None if value == _MISSING_STEP else value

    @property
    def intervention_completed_step(self) -> int | None:
        value = int(self.shard.intervention_steps[self.episode_index])
        return None if value == _MISSING_STEP else value

    @property
    def commitment_reached(self) -> bool:
        return self.commitment_step is not None

    @property
    def step_count(self) -> int:
        start, end = self._step_bounds()
        return end - start

    def prefix_length(self, prefix: int | str) -> int:
        length = self.step_count
        if isinstance(prefix, int):
            return min(length, prefix)
        steps = self.step_numbers
        if prefix == "pre_commitment":
            commitment = self.commitment_step
            return 0 if commitment is None else int(np.count_nonzero(steps < commitment))
        if prefix == "eventual":
            delivery = self.first_delivery_step
            return length if delivery is None else int(np.count_nonzero(steps <= delivery))
        raise ValueError(f"unknown official trace prefix: {prefix!r}")

    @property
    def ego_actions(self) -> np.ndarray:
        start, end = self._step_bounds()
        return self.shard.ego_actions[start:end]

    @property
    def partner_actions(self) -> np.ndarray:
        start, end = self._step_bounds()
        return self.shard.partner_actions[start:end]

    @property
    def rewards(self) -> np.ndarray:
        start, end = self._step_bounds()
        return self.shard.rewards[start:end]

    @property
    def step_numbers(self) -> np.ndarray:
        start, end = self._step_bounds()
        return self.shard.step_numbers[start:end]

    def event_tokens(self, prefix: int | str) -> tuple[str, ...]:
        length = self.prefix_length(prefix)
        start, _end = self._step_bounds()
        tokens: list[str] = []
        for local_step in range(length):
            row = start + local_step
            tokens.append(f"ego_action:{int(self.shard.ego_actions[row])}")
            partner_action = int(self.shard.partner_actions[row])
            if partner_action >= 0:
                tokens.append(f"partner_action:{partner_action}")
            event_start = int(self.shard.event_offsets[row])
            event_end = int(self.shard.event_offsets[row + 1])
            tokens.extend(
                self.shard.event_vocab[int(event_id)]
                for event_id in self.shard.event_ids[event_start:event_end]
            )
            reward = float(self.shard.rewards[row])
            tokens.append(
                "reward:positive"
                if reward > 0
                else "reward:negative"
                if reward < 0
                else "reward:zero"
            )
        return tuple(tokens) or ("zero_step",)

    def visible_action_targets(self) -> list[tuple[tuple[int, int, int], int]]:
        output: list[tuple[tuple[int, int, int], int]] = []
        previous_ego = -1
        previous_reward = 0
        length = self.prefix_length("pre_commitment")
        for index in range(length):
            action = int(self.partner_actions[index])
            step = int(self.step_numbers[index])
            if action >= 0:
                output.append(((min(step // 8, 7), previous_ego, previous_reward), action))
            previous_ego = int(self.ego_actions[index])
            reward = float(self.rewards[index])
            previous_reward = 1 if reward > 0 else -1 if reward < 0 else 0
        return output

    def precommitment_partner_actions(self) -> tuple[int, ...]:
        length = self.prefix_length("pre_commitment")
        return tuple(int(value) for value in self.partner_actions[:length] if int(value) >= 0)

    def precommitment_has_diagnostic_pattern(self) -> bool:
        length = self.prefix_length("pre_commitment")
        start, _end = self._step_bounds()
        markers = ("PLACEMENT_ON_COUNTER", "POT", "PICKUP")
        for local_step in range(length):
            row = start + local_step
            event_start = int(self.shard.event_offsets[row])
            event_end = int(self.shard.event_offsets[row + 1])
            for event_id in self.shard.event_ids[event_start:event_end]:
                event = self.shard.event_vocab[int(event_id)]
                if any(marker in event for marker in markers):
                    return True
        return False

    def _step_bounds(self) -> tuple[int, int]:
        return (
            int(self.shard.step_offsets[self.episode_index]),
            int(self.shard.step_offsets[self.episode_index + 1]),
        )


class SparseTraceSequenceSource:
    """In-memory sparse shards that densify only one GRU mini-batch."""

    def __init__(
        self,
        shards: Sequence[_CompactShard],
        partner_labels: Mapping[str, int],
        prefix: int | str,
    ) -> None:
        self._shards = tuple(shards)
        self._prefix = prefix
        episodes: list[CompactTraceEpisode] = []
        labels: list[int] = []
        locations: list[tuple[int, int]] = []
        widths = {shard.entry.observation_width for shard in shards}
        if len(widths) != 1:
            raise ValueError("compact trace shards have inconsistent observation widths")
        for shard_index, shard in enumerate(shards):
            if shard.entry.partner_id not in partner_labels:
                continue
            label = partner_labels[shard.entry.partner_id]
            for episode_index in range(shard.entry.episodes):
                episodes.append(
                    CompactTraceEpisode(shard, episode_index, shard.entry.partner_id)
                )
                labels.append(label)
                locations.append((shard_index, episode_index))
        self._episodes = tuple(episodes)
        self._labels = tuple(labels)
        self._locations = tuple(locations)
        self._feature_width = next(iter(widths)) + 5 if widths else 0

    @property
    def size(self) -> int:
        return len(self._episodes)

    @property
    def feature_width(self) -> int:
        return self._feature_width

    @property
    def labels(self) -> Sequence[int]:
        return self._labels

    @property
    def episodes(self) -> Sequence[CompactTraceEpisode]:
        return self._episodes

    def iter_batches(
        self,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
    ) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        order = np.arange(self.size, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for offset in range(0, self.size, batch_size):
            indices = order[offset : offset + batch_size]
            episodes = [self._episodes[int(index)] for index in indices]
            lengths = np.asarray(
                [max(1, episode.prefix_length(self._prefix)) for episode in episodes],
                dtype=np.int64,
            )
            features = np.zeros(
                (len(episodes), int(lengths.max()), self.feature_width), dtype=np.float32
            )
            for batch_row, episode in enumerate(episodes):
                actual_length = episode.prefix_length(self._prefix)
                if actual_length == 0:
                    continue
                shard = episode.shard
                step_start, _step_end = episode._step_bounds()
                for local_step in range(actual_length):
                    step_row = step_start + local_step
                    sparse_start = int(shard.observation_offsets[step_row])
                    sparse_end = int(shard.observation_offsets[step_row + 1])
                    columns = shard.observation_indices[sparse_start:sparse_end]
                    features[batch_row, local_step, columns] = shard.observation_values[
                        sparse_start:sparse_end
                    ]
                    extra_start = self.feature_width - 5
                    features[batch_row, local_step, extra_start:] = (
                        float(shard.ego_actions[step_row]),
                        float(shard.partner_actions[step_row]),
                        float(shard.rewards[step_row]),
                        float(shard.step_numbers[step_row]) / 400.0,
                        1.0,
                    )
            yield (
                features,
                lengths,
                np.asarray([self._labels[int(index)] for index in indices], dtype=np.int64),
                indices,
            )


class OfficialCompactTraceStore:
    """Resumable cache that removes dense JSON observations from working memory."""

    def __init__(self, root: Path, entries: Sequence[CompactTraceCacheEntry]) -> None:
        self.root = root
        self.entries = tuple(entries)

    @classmethod
    def prepare(
        cls,
        index: OfficialTraceIndex,
        root: str | Path,
        *,
        progress: Callable[[str], None] | None = None,
    ) -> OfficialCompactTraceStore:
        cache_root = Path(root).resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
        manifest_path = cache_root / "compact-trace-cache.json"
        existing = _read_manifest_entries(manifest_path)
        completed: dict[str, CompactTraceCacheEntry] = {}
        total = len(index.entries)
        for ordinal, source_entry in enumerate(index.entries, start=1):
            cached = existing.get(source_entry.trace_id)
            if cached is not None and _cache_entry_is_valid(cached, source_entry):
                completed[source_entry.trace_id] = cached
                continue
            if progress is not None:
                progress(f"trace-cache {ordinal}/{total}: {source_entry.trace_id}")
            cache_entry = _compact_trace_entry(source_entry, cache_root)
            completed[source_entry.trace_id] = cache_entry
            _write_cache_manifest(manifest_path, index.suite_id, completed.values())
        ordered = tuple(completed[entry.trace_id] for entry in index.entries)
        _write_cache_manifest(manifest_path, index.suite_id, ordered)
        return cls(cache_root, ordered)

    def select(
        self,
        layout_id: str,
        evidence_policy: str,
        split: str,
        *,
        partners: set[str] | None = None,
    ) -> tuple[CompactTraceCacheEntry, ...]:
        return tuple(
            entry
            for entry in self.entries
            if entry.layout_id == layout_id
            and entry.evidence_policy == evidence_policy
            and entry.split == split
            and (partners is None or entry.partner_id in partners)
        )

    def sequence_source(
        self,
        layout_id: str,
        evidence_policy: str,
        split: str,
        partner_labels: Mapping[str, int],
        prefix: int | str,
        *,
        partners: set[str] | None = None,
    ) -> SparseTraceSequenceSource:
        entries = self.select(
            layout_id, evidence_policy, split, partners=partners
        )
        shards = tuple(_load_compact_shard(entry) for entry in entries)
        return SparseTraceSequenceSource(shards, partner_labels, prefix)

    def episodes(
        self,
        layout_id: str,
        evidence_policy: str,
        split: str,
        *,
        partners: set[str] | None = None,
    ) -> tuple[CompactTraceEpisode, ...]:
        selected = self.select(layout_id, evidence_policy, split, partners=partners)
        output: list[CompactTraceEpisode] = []
        for entry in selected:
            shard = _load_compact_shard(entry)
            output.extend(
                CompactTraceEpisode(shard, episode, entry.partner_id)
                for episode in range(entry.episodes)
            )
        return tuple(output)


def _compact_trace_entry(
    source_entry: OfficialTraceIndexEntry,
    cache_root: Path,
) -> CompactTraceCacheEntry:
    source = Path(source_entry.path)
    if _sha256(source) != source_entry.content_hash:
        raise ValueError(f"trace source hash changed: {source_entry.trace_id}")
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        result: Any = json.load(handle)
    if not isinstance(result, dict) or result.get("operation") != "official_trace_rollout":
        raise ValueError(f"invalid official trace result: {source}")
    if result.get("policy_training_performed") is not False:
        raise ValueError("compact trace cache accepts inference-only results")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != source_entry.episodes:
        raise ValueError(f"trace episode count changed: {source_entry.trace_id}")
    widths = {int(episode["observation_width"]) for episode in episodes}
    if len(widths) != 1 or next(iter(widths)) <= 0:
        raise ValueError("official trace observations require one positive width")
    width = next(iter(widths))
    step_offsets = [0]
    observation_offsets = [0]
    observation_indices: list[int] = []
    observation_values: list[float] = []
    ego_actions: list[int] = []
    partner_actions: list[int] = []
    rewards: list[float] = []
    step_numbers: list[int] = []
    event_offsets = [0]
    event_ids: list[int] = []
    event_vocab: list[str] = []
    event_lookup: dict[str, int] = {}
    environment_keys: list[int] = []
    sparse_returns: list[float] = []
    commitment_steps: list[int] = []
    first_delivery_steps: list[int] = []
    intervention_steps: list[int] = []
    ego_seats: list[int] = []
    for episode in episodes:
        steps = episode["steps"]
        environment_keys.append(int(episode["environment_key"]))
        sparse_returns.append(float(episode["sparse_return"]))
        commitment_steps.append(_optional_step(episode.get("commitment_step")))
        first_delivery_steps.append(_optional_step(episode.get("first_delivery_step")))
        intervention_steps.append(_optional_step(episode.get("intervention_completed_step")))
        ego_seats.append(int(episode["ego_seat"]))
        for step in steps:
            observation = np.asarray(step["ego_observation"], dtype=np.float32)
            if observation.shape != (width,):
                raise ValueError("official ego-observation width changed within a trace")
            nonzero = np.flatnonzero(observation)
            observation_indices.extend(int(value) for value in nonzero)
            observation_values.extend(float(value) for value in observation[nonzero])
            observation_offsets.append(len(observation_indices))
            ego_actions.append(int(step["ego_action"]))
            partner_action = step.get("visible_partner_action")
            partner_actions.append(-1 if partner_action is None else int(partner_action))
            rewards.append(float(step.get("reward", 0.0)))
            step_numbers.append(int(step["step"]))
            for raw_event in step.get("events", ()):
                event = str(raw_event)
                event_id = event_lookup.get(event)
                if event_id is None:
                    event_id = len(event_vocab)
                    event_lookup[event] = event_id
                    event_vocab.append(event)
                event_ids.append(event_id)
            event_offsets.append(len(event_ids))
        step_offsets.append(len(ego_actions))
    identifier = hashlib.sha256(source_entry.trace_id.encode()).hexdigest()[:20]
    cache_path = cache_root / f"{identifier}.npz"
    with tempfile.NamedTemporaryFile(dir=cache_root, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            schema_version=np.asarray([_CACHE_SCHEMA_VERSION], dtype=np.int16),
            source_hash=np.asarray([source_entry.content_hash]),
            step_offsets=np.asarray(step_offsets, dtype=np.int32),
            observation_offsets=np.asarray(observation_offsets, dtype=np.int64),
            observation_indices=np.asarray(observation_indices, dtype=np.int16),
            observation_values=np.asarray(observation_values, dtype=np.float32),
            ego_actions=np.asarray(ego_actions, dtype=np.int8),
            partner_actions=np.asarray(partner_actions, dtype=np.int8),
            rewards=np.asarray(rewards, dtype=np.float32),
            step_numbers=np.asarray(step_numbers, dtype=np.int16),
            event_offsets=np.asarray(event_offsets, dtype=np.int64),
            event_ids=np.asarray(event_ids, dtype=np.int16),
            event_vocab=np.asarray(event_vocab, dtype=np.str_),
            environment_keys=np.asarray(environment_keys, dtype=np.int64),
            sparse_returns=np.asarray(sparse_returns, dtype=np.float32),
            commitment_steps=np.asarray(commitment_steps, dtype=np.int16),
            first_delivery_steps=np.asarray(first_delivery_steps, dtype=np.int16),
            intervention_steps=np.asarray(intervention_steps, dtype=np.int16),
            ego_seats=np.asarray(ego_seats, dtype=np.int8),
            observation_width=np.asarray([width], dtype=np.int32),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, cache_path)
    return CompactTraceCacheEntry(
        trace_id=source_entry.trace_id,
        layout_id=source_entry.layout_id,
        partner_id=source_entry.partner_id,
        evidence_policy=source_entry.evidence_policy,
        split=source_entry.split,
        source_hash=source_entry.content_hash,
        cache_path=cache_path,
        cache_hash=_sha256(cache_path),
        episodes=source_entry.episodes,
        observation_width=width,
    )


def _load_compact_shard(entry: CompactTraceCacheEntry) -> _CompactShard:
    with np.load(entry.cache_path, allow_pickle=False) as arrays:
        if int(arrays["schema_version"][0]) != _CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported compact trace cache: {entry.cache_path}")
        if str(arrays["source_hash"][0]) != entry.source_hash:
            raise ValueError(f"compact trace source hash mismatch: {entry.trace_id}")
        return _CompactShard(
            entry=entry,
            step_offsets=arrays["step_offsets"].copy(),
            observation_offsets=arrays["observation_offsets"].copy(),
            observation_indices=arrays["observation_indices"].copy(),
            observation_values=arrays["observation_values"].copy(),
            ego_actions=arrays["ego_actions"].copy(),
            partner_actions=arrays["partner_actions"].copy(),
            rewards=arrays["rewards"].copy(),
            step_numbers=arrays["step_numbers"].copy(),
            event_offsets=arrays["event_offsets"].copy(),
            event_ids=arrays["event_ids"].copy(),
            event_vocab=tuple(str(value) for value in arrays["event_vocab"]),
            environment_keys=arrays["environment_keys"].copy(),
            sparse_returns=arrays["sparse_returns"].copy(),
            commitment_steps=arrays["commitment_steps"].copy(),
            first_delivery_steps=arrays["first_delivery_steps"].copy(),
            intervention_steps=arrays["intervention_steps"].copy(),
            ego_seats=arrays["ego_seats"].copy(),
        )


def _read_manifest_entries(path: Path) -> dict[str, CompactTraceCacheEntry]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != _CACHE_SCHEMA_VERSION:
        return {}
    output: dict[str, CompactTraceCacheEntry] = {}
    for item in raw.get("entries", ()):
        entry = CompactTraceCacheEntry(
            trace_id=str(item["trace_id"]),
            layout_id=str(item["layout_id"]),
            partner_id=str(item["partner_id"]),
            evidence_policy=str(item["evidence_policy"]),
            split=str(item["split"]),
            source_hash=str(item["source_hash"]),
            cache_path=Path(item["cache_path"]),
            cache_hash=str(item["cache_hash"]),
            episodes=int(item["episodes"]),
            observation_width=int(item["observation_width"]),
        )
        output[entry.trace_id] = entry
    return output


def _write_cache_manifest(
    path: Path,
    suite_id: str,
    entries: Sequence[CompactTraceCacheEntry] | Any,
) -> None:
    ordered = sorted(entries, key=lambda item: item.trace_id)
    payload = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "suite_id": suite_id,
        "entries": [
            {
                "trace_id": entry.trace_id,
                "layout_id": entry.layout_id,
                "partner_id": entry.partner_id,
                "evidence_policy": entry.evidence_policy,
                "split": entry.split,
                "source_hash": entry.source_hash,
                "cache_path": str(entry.cache_path),
                "cache_hash": entry.cache_hash,
                "episodes": entry.episodes,
                "observation_width": entry.observation_width,
            }
            for entry in ordered
        ],
    }
    _atomic_json(path, payload)


def _cache_entry_is_valid(
    cached: CompactTraceCacheEntry,
    source: OfficialTraceIndexEntry,
) -> bool:
    return bool(
        cached.source_hash == source.content_hash
        and cached.episodes == source.episodes
        and cached.cache_path.is_file()
        and _sha256(cached.cache_path) == cached.cache_hash
    )


def _optional_step(value: Any) -> int:
    return _MISSING_STEP if value is None else int(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
