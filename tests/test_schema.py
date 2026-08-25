from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from zsc_identifiability.models import FiniteConventionGame, load_game_file

GAMES = Path(__file__).resolve().parents[1] / "phase-2-exact-model" / "games"


def raw_game(name: str) -> dict:
    return load_game_file(GAMES / f"{name}.json").model_dump(mode="json")


def test_all_canonical_games_validate() -> None:
    games = [load_game_file(path) for path in sorted(GAMES.glob("*.json"))]
    assert len(games) == 10
    assert len({game.game_id for game in games}) == 10


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda data: data["modes"][0].update(probability="3/4"), "prior"),
        (lambda data: data["kernels"][2]["outcomes"][0].update(probability="9/10"), "sum"),
        (lambda data: data["kernels"][2]["outcomes"][0].update(cost="-1"), "negative"),
        (lambda data: data["decision_losses"][1].update(loss="-1"), "nonnegative"),
        (lambda data: data["actions"][0].update(id="stage_shared_item"), "duplicate"),
    ],
)
def test_invalid_game_rows_are_rejected(mutation, message: str) -> None:
    data = deepcopy(raw_game("active-separable"))
    mutation(data)
    with pytest.raises(ValidationError, match=message):
        FiniteConventionGame.model_validate(data)


def test_non_string_probability_is_rejected() -> None:
    data = deepcopy(raw_game("active-separable"))
    data["modes"][0]["probability"] = 0.5
    with pytest.raises(ValidationError, match="string"):
        FiniteConventionGame.model_validate(data)


def test_special_query_action_is_rejected() -> None:
    data = deepcopy(raw_game("active-separable"))
    data["actions"][1]["task_semantics"] = "query the partner type"
    with pytest.raises(ValidationError, match="special query"):
        FiniteConventionGame.model_validate(data)


def test_missing_available_kernel_is_rejected() -> None:
    data = deepcopy(raw_game("active-separable"))
    data["kernels"].pop()
    with pytest.raises(ValidationError, match="missing kernel"):
        FiniteConventionGame.model_validate(data)


def test_mode_requires_a_zero_loss_decision() -> None:
    data = deepcopy(raw_game("active-separable"))
    data["decision_losses"][0]["loss"] = "1"
    with pytest.raises(ValidationError, match="no zero-loss"):
        FiniteConventionGame.model_validate(data)
