"""Public evaluation config contracts that run without Isaac Sim."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from insightbench.utils.eval_config import load_eval_config, validate_required_eval_inputs


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SMOLVLA_CONFIG = _REPO_ROOT / "configs/eval/smolvla.yaml"


def test_shipped_smolvla_config_has_no_unverified_stats_default() -> None:
    cfg = load_eval_config(str(_SMOLVLA_CONFIG), [])

    assert OmegaConf.select(cfg, "policy.dataset_stats_repo") == ""
    assert OmegaConf.select(cfg, "policy.dataset_stats_root") == ""
    assert "paragon7060/INSIGHTfixpos" not in _SMOLVLA_CONFIG.read_text()


def test_smolvla_requires_explicit_stats_before_isaac_launch() -> None:
    cfg = load_eval_config(
        str(_SMOLVLA_CONFIG),
        ["policy.checkpoint=your-hf-user-or-org/smolvla-policy"],
    )

    with pytest.raises(SystemExit, match="SmolVLA requires dataset statistics"):
        validate_required_eval_inputs(cfg, str(_SMOLVLA_CONFIG))


@pytest.mark.parametrize(
    "stats_override",
    [
        "policy.dataset_stats_repo=your-hf-user-or-org/dataset-repo",
        "policy.dataset_stats_root=path/to/lerobot-dataset",
    ],
)
def test_smolvla_accepts_one_explicit_stats_source(stats_override: str) -> None:
    cfg = load_eval_config(
        str(_SMOLVLA_CONFIG),
        ["policy.checkpoint=your-hf-user-or-org/smolvla-policy", stats_override],
    )

    validate_required_eval_inputs(cfg, str(_SMOLVLA_CONFIG))
