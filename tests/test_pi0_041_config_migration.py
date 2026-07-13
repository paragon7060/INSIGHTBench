"""Regression coverage for 0.3.4 Pi0 checkpoint config decoding on LeRobot 0.4.1."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "pi0_034_checkpoint_config.json"


def _load_fixture() -> dict:
    with FIXTURE.open() as file:
        return json.load(file)


def test_legacy_pi0_config_decodes_without_mutating_source(tmp_path) -> None:
    lerobot = pytest.importorskip("lerobot")
    if lerobot.__version__ != "0.4.1":
        pytest.skip("requires the documented LeRobot 0.4.1 baseline")

    pi0 = importlib.import_module("insightbench.policies.pi0")
    source = _load_fixture()
    original = json.loads(json.dumps(source))

    checkpoint_dir = tmp_path / "pretrained_model"
    checkpoint_dir.mkdir()
    config_path = checkpoint_dir / "config.json"
    config_path.write_text(json.dumps(source))

    decoded = pi0._load_pi0_checkpoint_config(checkpoint_dir)

    assert decoded.image_resolution == (224, 224)
    assert decoded.num_inference_steps == 10
    assert decoded.action_expert_variant == "gemma_300m"
    assert json.loads(config_path.read_text()) == original


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("adapt_to_pi_aloha", True),
        ("use_delta_joint_actions_aloha", True),
        ("proj_width", 2048),
        ("use_cache", False),
        ("attention_implementation", "fa2"),
        ("resize_imgs_with_padding", None),
    ],
)
def test_legacy_pi0_config_rejects_unproven_semantic_changes(field, value) -> None:
    pytest.importorskip("lerobot")
    pi0 = importlib.import_module("insightbench.policies.pi0")
    config = _load_fixture()
    config[field] = value

    with pytest.raises(pi0.Pi0CheckpointConfigCompatibilityError):
        pi0._normalize_pi0_checkpoint_config(config)


def test_legacy_pi0_config_rejects_unclassified_fields() -> None:
    pytest.importorskip("lerobot")
    pi0 = importlib.import_module("insightbench.policies.pi0")
    config = _load_fixture()
    config["future_checkpoint_field"] = "unclassified"

    with pytest.raises(pi0.Pi0CheckpointConfigCompatibilityError, match="unclassified"):
        pi0._normalize_pi0_checkpoint_config(config)
