from __future__ import annotations

import json

from insightbench.utils.results import aggregate_results, save_result


def test_save_result_writes_expected_json(tmp_path) -> None:
    path = save_result(
        str(tmp_path),
        obj_name="cabinet",
        asset_path="asset_a",
        task_idx=1,
        scene_key="scene_1",
        successes=3,
        attempts=4,
    )

    with open(path) as f:
        result = json.load(f)

    assert result == {
        "object": "cabinet",
        "asset": "asset_a",
        "task_idx": 1,
        "scene_key": "scene_1",
        "attempts": 4,
        "successes": 3,
        "rate": 0.75,
    }


def test_aggregate_results_returns_total_and_per_object_stats(tmp_path) -> None:
    save_result(str(tmp_path), "cabinet", "asset_a", 0, "scene_1", successes=3, attempts=4)
    save_result(str(tmp_path), "cabinet", "asset_b", 1, "scene_2", successes=1, attempts=2)
    save_result(str(tmp_path), "door", "asset_c", 0, "scene_3", successes=0, attempts=1)

    summary = aggregate_results(str(tmp_path))

    assert summary["total"] == {"successes": 4, "attempts": 7, "rate": 0.5714}
    assert summary["per_object"] == {
        "cabinet": {"successes": 4, "attempts": 6, "rate": 0.6667},
        "door": {"successes": 0, "attempts": 1, "rate": 0.0},
    }
