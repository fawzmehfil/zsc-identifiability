from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from zsc_identifiability.established_gru import (
    fit_streaming_cross_fitted_gru_posterior,
)
from zsc_identifiability.established_official_analysis import (
    _load_gru_seed_checkpoint,
    _write_gru_seed_checkpoint,
)
from zsc_identifiability.established_official_models import (
    OfficialTraceIndex,
    OfficialTraceIndexEntry,
)
from zsc_identifiability.established_official_trace_store import (
    OfficialCompactTraceStore,
)


def test_compact_trace_store_is_resumable_and_densifies_only_a_batch(
    tmp_path: Path,
) -> None:
    index = _trace_index(tmp_path)
    progress: list[str] = []
    store = OfficialCompactTraceStore.prepare(
        index, tmp_path / "cache", progress=progress.append
    )
    assert len(progress) == len(index.entries)
    assert len(store.entries) == len(index.entries)

    resumed_progress: list[str] = []
    resumed = OfficialCompactTraceStore.prepare(
        index, tmp_path / "cache", progress=resumed_progress.append
    )
    assert resumed_progress == []

    labels = {"random3_m:hsp1:mid": 0, "random3_m:hsp2:mid": 1}
    source = resumed.sequence_source(
        "random3_m", "ordinary_progress", "confirmatory", labels, 2
    )
    assert source.size == 4
    assert source.feature_width == 9
    batches = list(source.iter_batches(2, shuffle=False, seed=0))
    assert len(batches) == 2
    features, lengths, batch_labels, indices = batches[0]
    assert features.shape == (2, 2, 9)
    assert lengths.tolist() == [2, 2]
    assert batch_labels.tolist() == [0, 0]
    assert indices.tolist() == [0, 1]
    assert features[0, 0, :4].tolist() == [255.0, 0.0, 0.0, 0.0]
    assert source.episodes[0].event_tokens("pre_commitment")[0] == "ego_action:1"


def test_streaming_gru_and_seed_checkpoint_round_trip(tmp_path: Path) -> None:
    index = _trace_index(tmp_path)
    store = OfficialCompactTraceStore.prepare(index, tmp_path / "cache")
    labels = {"random3_m:hsp1:mid": 0, "random3_m:hsp2:mid": 1}
    sources = {
        split: store.sequence_source(
            "random3_m", "ordinary_progress", split, labels, "pre_commitment"
        )
        for split in ("calibration", "validation", "confirmatory")
    }
    result = fit_streaming_cross_fitted_gru_posterior(
        sources["calibration"],
        sources["validation"],
        sources["confirmatory"],
        (0.5, 0.5),
        ((0.0, 1.0), (1.0, 0.0)),
        hidden_size=4,
        max_epochs=3,
        patience=2,
        batch_size=2,
        seed=73,
    )
    assert result.posteriors.shape == (4, 2)
    assert np.allclose(result.posteriors.sum(axis=1), 1.0)

    checkpoint = tmp_path / "seed.npz"
    _write_gru_seed_checkpoint(checkpoint, "a" * 64, result.posteriors)
    restored = _load_gru_seed_checkpoint(checkpoint, "a" * 64, 4, 2)
    assert restored is not None
    assert np.array_equal(restored, result.posteriors)
    assert _load_gru_seed_checkpoint(checkpoint, "b" * 64, 4, 2) is None


def _trace_index(tmp_path: Path) -> OfficialTraceIndex:
    entries: list[OfficialTraceIndexEntry] = []
    partners = ("random3_m:hsp1:mid", "random3_m:hsp2:mid")
    for split_number, split in enumerate(("calibration", "validation", "confirmatory")):
        for partner_number, partner in enumerate(partners):
            path = tmp_path / f"{split}-{partner_number}.json.gz"
            episodes = [
                _episode(
                    environment_key=split_number * 100 + episode,
                    partner_number=partner_number,
                )
                for episode in range(2)
            ]
            payload = {
                "operation": "official_trace_rollout",
                "policy_training_performed": False,
                "partner_deployment": "stochastic",
                "layout_id": "random3_m",
                "partner_id": partner,
                "evidence_policy": "ordinary_progress",
                "split": split,
                "episodes": episodes,
            }
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                json.dump(payload, handle)
            entries.append(
                OfficialTraceIndexEntry(
                    trace_id=f"{split}-{partner_number}",
                    layout_id="random3_m",
                    partner_id=partner,
                    evidence_policy="ordinary_progress",
                    split=split,  # type: ignore[arg-type]
                    path=str(path),
                    content_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
                    episodes=2,
                )
            )
    return OfficialTraceIndex(suite_id="streaming-test", entries=tuple(entries))


def _episode(environment_key: int, partner_number: int) -> dict[str, object]:
    steps: list[dict[str, object]] = []
    for step in range(3):
        observation = [0.0, 0.0, 0.0, 0.0]
        observation[partner_number] = 255.0
        steps.append(
            {
                "step": step,
                "ego_action": 1 + partner_number,
                "visible_partner_action": 2 + partner_number,
                "reward": 0.0,
                "events": [f"partner_event:{partner_number}"],
                "ego_observation": observation,
            }
        )
    return {
        "environment_key": environment_key,
        "sparse_return": 1.0,
        "ego_seat": environment_key % 2,
        "observation_width": 4,
        "commitment_reached": True,
        "commitment_step": 2,
        "first_delivery_step": 2,
        "intervention_completed_step": None,
        "steps": steps,
    }
