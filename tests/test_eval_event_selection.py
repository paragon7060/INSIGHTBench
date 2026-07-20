from __future__ import annotations

import pytest

from insightbench.envs.builder import resolve_eval_event_key


@pytest.mark.parametrize("scene_key", ["5a", "5g"])
def test_fixed_position_squeeze_eval_keeps_orientation_randomization(scene_key: str) -> None:
    assert resolve_eval_event_key("bottle", scene_key, pos_rand=False) == "5squeeze_test"


@pytest.mark.parametrize("scene_key", ["5b", "5c", "5d", "5e", "5f", "5h"])
def test_fixed_position_non_squeeze_bottle_eval_uses_standard_test_events(scene_key: str) -> None:
    assert resolve_eval_event_key("bottle", scene_key, pos_rand=False) == "5test"


@pytest.mark.parametrize("scene_key", ["5a", "5g"])
def test_position_randomized_squeeze_eval_uses_full_bottle_events(scene_key: str) -> None:
    assert resolve_eval_event_key("bottle", scene_key, pos_rand=True) == "5"
