"""Frozen event-decoder diagnostics for curated replay histories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from zsc_identifiability.demo.models import DemoCurationEntry, DiagnosticPoint
from zsc_identifiability.established_official_decision import (
    decoder_probabilities,
    signed_hash_event_features,
)
from zsc_identifiability.established_official_redesign_models import (
    PairwiseDecisionDecoder,
)

Prefix = int | str
REGISTERED_PREFIXES: tuple[Prefix, ...] = (0, 8, 16, 32, "pre_commitment", "eventual")


class FrozenDiagnosticSources:
    """Resolve only the frozen analysis rows needed by the public collection."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self._decoder_units = self._index_decoder_units()
        matrix_payload = _read_json(
            self.root
            / "phase-6-established-validation/artifacts/official-checkpoint-audit/"
            "response-value-matrices.json"
        )
        self._matrices = {
            str(item["layout_id"]): item for item in matrix_payload["matrices"]
        }
        rows = _read_json(
            self.root
            / "phase-6-established-validation/artifacts/official-measurement-v3/"
            "pairwise-decision-value.json"
        )
        self._decision_rows = {
            (
                str(item["layout_id"]),
                str(item["left_partner_id"]),
                str(item["right_partner_id"]),
                str(item["evidence_policy"]),
                str(item["representation"]),
                str(item["prefix"]),
            ): item
            for item in rows
        }

    def diagnostics(
        self,
        entry: DemoCurationEntry,
        episode: Mapping[str, Any],
    ) -> tuple[DiagnosticPoint, ...]:
        matrix = self._matrices[entry.layout_id]
        partners = [str(value) for value in matrix["partner_ids"]]
        responses = [str(value) for value in matrix["response_ids"]]
        losses = np.asarray(matrix["normalized_losses"], dtype=np.float64)
        left_row = losses[partners.index(entry.left_partner_id)]
        right_row = losses[partners.index(entry.right_partner_id)]
        pair_losses = np.stack((left_row, right_row))
        true_mode = 0 if entry.partner_id == entry.left_partner_id else 1
        points: list[DiagnosticPoint] = []
        for prefix in REGISTERED_PREFIXES:
            decoder = self._decoder(entry, prefix)
            features = _event_features(episode, prefix)
            probability_right = float(decoder_probabilities(decoder, features[None, :])[0])
            posterior = np.asarray([1.0 - probability_right, probability_right])
            expected_losses = posterior @ pair_losses
            decision_index = int(np.argmin(expected_losses))
            aggregate = self.decision_row(entry, "event", prefix)
            frame = _prefix_frame(episode, prefix)
            commitment = episode.get("commitment_step")
            phase = (
                "before"
                if commitment is None or frame <= int(commitment)
                else "after"
            )
            if commitment is not None and frame == int(commitment) + 1:
                phase = "commitment"
            points.append(
                DiagnosticPoint(
                    key=str(prefix).replace("_", "-"),
                    prefix=prefix,  # type: ignore[arg-type]
                    frame=frame,
                    phase=phase,  # type: ignore[arg-type]
                    probability_left=1.0 - probability_right,
                    probability_right=probability_right,
                    recommended_response_id=responses[decision_index],
                    recommended_response_alias=_response_alias(
                        decision_index,
                        left_row,
                        right_row,
                    ),
                    estimated_expected_risk=max(0.0, float(expected_losses[decision_index])),
                    selected_response_loss_for_true_mode=max(
                        0.0, float(pair_losses[true_mode, decision_index])
                    ),
                    aggregate_event_dri=(
                        None if aggregate is None else aggregate.get("dri")
                    ),
                )
            )
        return tuple(points)

    def decision_row(
        self,
        entry: DemoCurationEntry,
        representation: str,
        prefix: Prefix,
    ) -> dict[str, Any] | None:
        direct = (
            entry.layout_id,
            entry.left_partner_id,
            entry.right_partner_id,
            entry.evidence_policy,
            representation,
            str(prefix),
        )
        row = self._decision_rows.get(direct)
        if row is not None:
            return dict(row)
        reverse = (
            entry.layout_id,
            entry.right_partner_id,
            entry.left_partner_id,
            entry.evidence_policy,
            representation,
            str(prefix),
        )
        reverse_row = self._decision_rows.get(reverse)
        return None if reverse_row is None else dict(reverse_row)

    def response_contract(
        self, entry: DemoCurationEntry
    ) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        matrix = self._matrices[entry.layout_id]
        partners = [str(value) for value in matrix["partner_ids"]]
        responses = [str(value) for value in matrix["response_ids"]]
        losses = np.asarray(matrix["normalized_losses"], dtype=np.float64)
        left_best = int(np.argmin(losses[partners.index(entry.left_partner_id)]))
        right_best = int(np.argmin(losses[partners.index(entry.right_partner_id)]))
        if left_best == right_best:
            raise ValueError("curated partner pair is not response-conflicting")
        return (
            (responses[left_best], "Response A", ("left",)),
            (responses[right_best], "Response B", ("right",)),
        )

    def _decoder(self, entry: DemoCurationEntry, prefix: Prefix) -> PairwiseDecisionDecoder:
        key = (entry.layout_id, entry.evidence_policy, str(prefix), "event")
        path = self._decoder_units.get(key)
        if path is None:
            raise ValueError(f"no frozen event-decoder unit for {key!r}")
        payload = _read_json(path)
        for item in payload["decoders"]:
            if {
                str(item["left_partner_id"]),
                str(item["right_partner_id"]),
            } == {entry.left_partner_id, entry.right_partner_id}:
                decoder = PairwiseDecisionDecoder.model_validate(item)
                if decoder.left_partner_id != entry.left_partner_id:
                    decoder = _reverse_decoder(decoder)
                return decoder
        raise ValueError(f"frozen decoder omits curated pair {entry.replay_id}")

    def _index_decoder_units(self) -> dict[tuple[str, str, str, str], Path]:
        directory = (
            self.root
            / "phase-6-established-validation/runs/official-measurement-v3/fit/"
            "decoder-work-units"
        )
        result: dict[tuple[str, str, str, str], Path] = {}
        for path in sorted(directory.glob("*.json")):
            payload = _read_json(path)
            selection = payload["selection"]
            key = (
                str(selection["layout_id"]),
                str(selection["evidence_policy"]),
                str(selection["prefix"]),
                str(selection["representation"]),
            )
            result[key] = path
        return result


def _reverse_decoder(decoder: PairwiseDecisionDecoder) -> PairwiseDecisionDecoder:
    payload = decoder.model_dump(mode="json")
    payload["left_partner_id"], payload["right_partner_id"] = (
        payload["right_partner_id"],
        payload["left_partner_id"],
    )
    payload["coefficient"] = [-float(value) for value in payload["coefficient"]]
    payload["intercept"] = -float(payload["intercept"])
    return PairwiseDecisionDecoder.model_validate(payload)


def _event_features(episode: Mapping[str, Any], prefix: Prefix) -> np.ndarray:
    steps = list(episode["steps"])
    if isinstance(prefix, int):
        selected = steps[:prefix]
    elif prefix == "pre_commitment":
        commitment = episode.get("commitment_step")
        selected = [] if commitment is None else [
            item for item in steps if int(item["step"]) < int(commitment)
        ]
    elif prefix == "eventual":
        delivery = episode.get("first_delivery_step")
        selected = steps if delivery is None else [
            item for item in steps if int(item["step"]) <= int(delivery)
        ]
    else:
        raise ValueError(f"unsupported diagnostic prefix: {prefix!r}")
    tokens: list[tuple[int, str]] = []
    visible = 0
    reward = 0.0
    for item in selected:
        step = int(item["step"])
        tokens.append((step, f"ego_action:{int(item['ego_action'])}"))
        partner_action = item.get("visible_partner_action")
        if partner_action is not None and int(partner_action) >= 0:
            tokens.append((step, f"partner_action:{int(partner_action)}"))
            visible += 1
        tokens.extend((step, str(event)) for event in item.get("events", ()))
        step_reward = float(item.get("reward", 0.0))
        reward += step_reward
        tokens.append(
            (
                step,
                "reward:positive"
                if step_reward > 0
                else "reward:negative"
                if step_reward < 0
                else "reward:zero",
            )
        )
    length = len(selected)
    return signed_hash_event_features(
        tokens,
        observed_length=length,
        cumulative_reward=reward,
        partner_visibility_rate=0.0 if length == 0 else visible / length,
    )


def _prefix_frame(episode: Mapping[str, Any], prefix: Prefix) -> int:
    step_count = int(episode["steps_elapsed"])
    if isinstance(prefix, int):
        return min(prefix, step_count)
    if prefix == "pre_commitment":
        commitment = episode.get("commitment_step")
        return 0 if commitment is None else min(int(commitment), step_count)
    delivery = episode.get("first_delivery_step")
    return step_count if delivery is None else min(int(delivery) + 1, step_count)


def _response_alias(index: int, left_losses: np.ndarray, right_losses: np.ndarray) -> str:
    if index == int(np.argmin(left_losses)):
        return "Response A"
    if index == int(np.argmin(right_losses)):
        return "Response B"
    return "Library response"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
