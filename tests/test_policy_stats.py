from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from insightbench.policies.pi0 import ACTION, OBS_STATE, Pi0Wrapper


def test_trim_stats_slices_state_and_action_stats_without_touching_count() -> None:
    cfg = SimpleNamespace(state_start_idx=2, state_dim=4, action_dim=3)
    stats = {
        OBS_STATE: {
            "mean": torch.arange(10, dtype=torch.float32),
            "std": torch.arange(10, dtype=torch.float32) + 10,
            "min": torch.arange(10, dtype=torch.float32) + 20,
            "max": torch.arange(10, dtype=torch.float32) + 30,
            "count": 5,
        },
        ACTION: {
            "mean": torch.arange(6, dtype=torch.float32),
            "std": torch.arange(6, dtype=torch.float32) + 10,
            "min": torch.arange(6, dtype=torch.float32) + 20,
            "max": torch.arange(6, dtype=torch.float32) + 30,
            "count": 7,
        },
    }

    Pi0Wrapper._trim_stats(stats, cfg)

    assert torch.equal(stats[OBS_STATE]["mean"], torch.tensor([2, 3, 4, 5], dtype=torch.float32))
    assert torch.equal(stats[OBS_STATE]["std"], torch.tensor([12, 13, 14, 15], dtype=torch.float32))
    assert torch.equal(stats[OBS_STATE]["min"], torch.tensor([22, 23, 24, 25], dtype=torch.float32))
    assert torch.equal(stats[OBS_STATE]["max"], torch.tensor([32, 33, 34, 35], dtype=torch.float32))
    assert stats[OBS_STATE]["count"] == 5

    assert torch.equal(stats[ACTION]["mean"], torch.tensor([0, 1, 2], dtype=torch.float32))
    assert torch.equal(stats[ACTION]["std"], torch.tensor([10, 11, 12], dtype=torch.float32))
    assert torch.equal(stats[ACTION]["min"], torch.tensor([20, 21, 22], dtype=torch.float32))
    assert torch.equal(stats[ACTION]["max"], torch.tensor([30, 31, 32], dtype=torch.float32))
    assert stats[ACTION]["count"] == 7
