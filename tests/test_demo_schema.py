from zsc_identifiability.demo.models import ReplayManifest


def test_replay_schema_has_versioned_required_contract() -> None:
    schema = ReplayManifest.model_json_schema()
    assert schema["properties"]["schema_version"]["const"] == "beforecommit_replay_1.0.0"
    assert {"source", "commitment", "summary", "artifacts"}.issubset(schema["required"])
