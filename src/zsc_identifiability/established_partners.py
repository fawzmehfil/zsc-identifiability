"""Leakage-safe reward-vector splitting and partner-pool manifests."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from zsc_identifiability.established_models import (
    EstablishedValidationSuite,
    PartnerCheckpoint,
    PartnerPoolManifest,
    SplitName,
)


def canonical_reward_vector(vector: Mapping[str, float]) -> str:
    cleaned = {key: float(value) for key, value in sorted(vector.items()) if value != 0}
    return json.dumps(cleaned, sort_keys=True, separators=(",", ":"), allow_nan=False)


def reward_vector_hash(vector: Mapping[str, float]) -> str:
    return hashlib.sha256(canonical_reward_vector(vector).encode()).hexdigest()


def enumerate_reward_vectors(suite: EstablishedValidationSuite) -> tuple[dict[str, float], ...]:
    """Enumerate hash-orderable behavior preferences with at most three terms."""

    spec = suite.partner_generation
    vectors: list[dict[str, float]] = []
    maximum = min(spec.maximum_nonzero_behavior_preferences, len(spec.behavior_events))
    for width in range(1, maximum + 1):
        for events in itertools.combinations(spec.behavior_events, width):
            for coefficients in itertools.product(spec.preference_values, repeat=width):
                vectors.append(dict(zip(events, coefficients, strict=True)))
    vectors.sort(key=reward_vector_hash)
    return tuple(vectors)


def split_for_reward_vector(
    vector: Mapping[str, float], suite: EstablishedValidationSuite
) -> SplitName:
    digest = int(
        hashlib.sha256(
            (
                suite.partner_generation.split_hash_salt
                + ":"
                + canonical_reward_vector(vector)
            ).encode()
        ).hexdigest(),
        16,
    )
    proportions = suite.partner_generation.split_proportions
    position = digest % sum(proportions)
    if position < proportions[0]:
        return "train"
    if position < proportions[0] + proportions[1]:
        return "validation"
    return "evaluation"


def vectors_for_split(
    suite: EstablishedValidationSuite, split: SplitName
) -> tuple[dict[str, float], ...]:
    return tuple(
        vector
        for vector in enumerate_reward_vectors(suite)
        if split_for_reward_vector(vector, suite) == split
    )


def generate_partner_pool_manifest(
    suite: EstablishedValidationSuite,
    split: SplitName,
    checkpoints: Iterable[PartnerCheckpoint] = (),
) -> PartnerPoolManifest:
    """Validate completed external jobs without pretending missing jobs exist."""

    items = tuple(sorted(checkpoints, key=lambda item: item.partner_id))
    if any(item.split != split for item in items):
        raise ValueError("checkpoint split does not match requested pool split")
    reward_hashes = tuple(item.reward_vector_hash for item in items)
    if len(reward_hashes) != len(set(reward_hashes)):
        # Two seeds per vector are expected.  Duplicates are valid only when the
        # seed and normalized checkpoint hash are distinct.
        keys = tuple((item.reward_vector_hash, item.seed) for item in items)
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate reward-vector/seed checkpoint")
    normalized = tuple(item.normalized_checkpoint_hash for item in items)
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate normalized checkpoint in partner pool")
    expected_hashes = {reward_vector_hash(vector) for vector in vectors_for_split(suite, split)}
    unknown = set(reward_hashes) - expected_hashes
    if unknown:
        raise ValueError(f"reward-vector hashes assigned to the wrong split: {sorted(unknown)}")
    competent = tuple(item for item in items if item.competent)
    quota = {
        "train": suite.partner_generation.training_partner_quota,
        "validation": suite.partner_generation.validation_partner_quota,
        "evaluation": suite.partner_generation.evaluation_candidate_quota,
    }[split]
    source_payload = {
        "suite": suite.model_dump(mode="json"),
        "split": split,
        "reward_vector_hashes": sorted(expected_hashes),
    }
    source_hash = hashlib.sha256(
        json.dumps(source_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PartnerPoolManifest(
        suite_id=suite.suite_id,
        split=split,
        checkpoints=items,
        quota=quota,
        quota_met=len(competent) >= quota,
        reward_vector_hashes=tuple(sorted(set(reward_hashes))),
        source_request_hash=source_hash,
        generated_at_utc=datetime.now(UTC).isoformat(),
    )


def load_partner_checkpoints(path: str | Path) -> tuple[PartnerCheckpoint, ...]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("checkpoints", [])
    if not isinstance(payload, list):
        raise ValueError("checkpoint index must be a list or contain a checkpoints list")
    return tuple(PartnerCheckpoint.model_validate(item) for item in payload)


def audit_split_leakage(manifests: Iterable[PartnerPoolManifest]) -> dict[str, Any]:
    seen_reward: dict[str, str] = {}
    seen_checkpoint: dict[str, str] = {}
    collisions: list[str] = []
    for manifest in manifests:
        for checkpoint in manifest.checkpoints:
            old_split = seen_reward.setdefault(checkpoint.reward_vector_hash, manifest.split)
            if old_split != manifest.split:
                collisions.append(
                    f"reward vector {checkpoint.reward_vector_hash} appears in "
                    f"{old_split} and {manifest.split}"
                )
            old_checkpoint_split = seen_checkpoint.setdefault(
                checkpoint.normalized_checkpoint_hash, manifest.split
            )
            if old_checkpoint_split != manifest.split:
                collisions.append(
                    "normalized checkpoint "
                    f"{checkpoint.normalized_checkpoint_hash} crosses "
                    f"{old_checkpoint_split}/{manifest.split}"
                )
    return {"passed": not collisions, "collisions": sorted(set(collisions))}
