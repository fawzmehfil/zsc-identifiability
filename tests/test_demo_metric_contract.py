import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_replays_separate_episode_diagnostics_from_aggregate_dri() -> None:
    replay_root = ROOT / "web/public/static-api/replays"
    for package in replay_root.iterdir():
        manifest = json.loads((package / "manifest.json").read_text())
        diagnostics = json.loads((package / "diagnostics.json").read_text())
        assert manifest["source"]["policy_training_performed"] is False
        assert "aggregate_dri_primary" in manifest["summary"]
        assert all("probability_left" in row for row in diagnostics)
        assert all("estimated_expected_risk" in row for row in diagnostics)
        commitment = manifest["commitment"]["frame"]
        if commitment is not None:
            precommitment = next(row for row in diagnostics if row["prefix"] == "pre_commitment")
            assert precommitment["frame"] <= commitment
