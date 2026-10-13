from pathlib import Path

from zsc_identifiability.demo.validator import validate_site_data

ROOT = Path(__file__).resolve().parents[1]


def test_committed_static_site_data_validates() -> None:
    result = validate_site_data(ROOT / "web/public/static-api")
    assert result == {
        "valid": True,
        "collection_id": "beforecommit-official-v1",
        "replay_count": 6,
    }
