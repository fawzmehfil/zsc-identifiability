from pathlib import Path

from zsc_identifiability.demo.curation import load_demo_curation

ROOT = Path(__file__).resolve().parents[1]


def test_public_curation_is_complete_and_unique() -> None:
    curation = load_demo_curation(ROOT / "demo/curation/official-v1.yaml")
    assert len(curation.entries) == 6
    assert len({entry.replay_id for entry in curation.entries}) == 6
    assert sum(entry.featured for entry in curation.entries) == 1
    assert all(not entry.representative for entry in curation.entries)
