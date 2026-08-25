from pathlib import Path

import numpy as np

from zsc_identifiability.learning_env import VectorConventionEnvironment
from zsc_identifiability.learning_models import load_learning_suite_file
from zsc_identifiability.learning_pools import generate_learning_pools

ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "phase-4-learned-audit" / "suites" / "canonical.json"


def pools():
    suite = load_learning_suite_file(SUITE_PATH)
    return generate_learning_pools(suite, suite_path=SUITE_PATH)


def test_runtime_encoding_contains_no_hidden_partner_metadata() -> None:
    cell = pools().by_cell()["active_only"]
    environment = VectorConventionEnvironment(cell.train, 7, 4)
    forbidden = ("mode", "profile", "cell", "dri", "response_signature")
    assert not any(
        token in feature for feature in environment.layout.feature_names for token in forbidden
    )
    batch = environment.current_batch()
    assert batch.observations.shape == (
        4,
        environment.layout.observation_size,
    )
    assert batch.action_masks.shape == (4, environment.layout.action_size)


def test_passive_mask_removes_staging_but_preserves_commitment() -> None:
    cell = pools().by_cell()["active_only"]
    environment = VectorConventionEnvironment(cell.train, 3, 1, action_class="passive")
    batch = environment.current_batch()
    stage = environment.layout.action_ids.index("stage_shared_item")
    commit = environment.layout.action_ids.index("commit:take_role_a")
    assert not batch.action_masks[0, stage]
    assert batch.action_masks[0, commit]


def test_commit_reward_matches_scaled_confusion_loss() -> None:
    cell = pools().by_cell()["active_only"]
    environment = VectorConventionEnvironment(cell.test, 11, 1, loss_scale=40)
    environment.game_indices[0] = 0
    environment.mode_indices[0] = 0
    mode = cell.test[0].game.mode_ids[0]
    correct = cell.source_population.descriptor.response_signature_by_mode[mode]
    wrong = next(item for item in cell.test[0].game.decisions if item != correct)
    wrong_index = environment.layout.action_ids.index(f"commit:{wrong}")
    result = environment.step(np.asarray([wrong_index]))
    assert result.rewards[0] == -1
    assert result.infos[0]["confusion_loss"] == 40
    assert result.terminated[0]


def test_environment_state_round_trip_preserves_rng_and_episode_state() -> None:
    cell = pools().by_cell()["passive_early"]
    left = VectorConventionEnvironment(cell.train, 17, 3)
    state = left.state_dict()
    right = VectorConventionEnvironment(cell.train, 99, 3)
    right.load_state_dict(state)
    assert right.state_dict() == state
