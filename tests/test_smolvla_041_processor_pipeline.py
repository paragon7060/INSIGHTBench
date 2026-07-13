"""Processor-pipeline coverage for the pinned LeRobot 0.4.1 SmolVLA runtime."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _require_041():
    lerobot = pytest.importorskip("lerobot")
    if lerobot.__version__ != "0.4.1":
        pytest.skip("requires the documented LeRobot 0.4.1 baseline")
    return importlib.import_module("insightbench.policies.smolvla")


def _processor_config():
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    config = SmolVLAConfig(device="cpu", tokenizer_max_length=12, vlm_model_name="fixture-tokenizer")
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(16,)),
        "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.right_shoulder": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        "observation.images.guide": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(8,)),
        "action.skill_id": PolicyFeature(type=FeatureType.ACTION, shape=(1,)),
    }
    return config


def _fixture_stats() -> dict:
    return {
        "observation.state": {
            "mean": torch.ones(16),
            "std": torch.full((16,), 2.0),
        },
        "action": {
            "mean": torch.full((8,), 3.0),
            "std": torch.full((8,), 4.0),
        },
        "action.skill_id": {
            "mean": torch.zeros(1),
            "std": torch.ones(1),
        },
    }


def test_official_smolvla_processors_preserve_batched_input_contract(monkeypatch) -> None:
    smolvla = _require_041()
    tokenizer_processor = importlib.import_module("lerobot.processor.tokenizer_processor")

    class FakeTokenizer:
        def __call__(self, prompts, *, max_length, **kwargs):
            batch_size = len(prompts)
            return {
                "input_ids": torch.ones((batch_size, max_length), dtype=torch.long),
                "attention_mask": torch.ones((batch_size, max_length), dtype=torch.long),
            }

    monkeypatch.setattr(
        tokenizer_processor,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda _name: FakeTokenizer()),
    )

    config = _processor_config()
    preprocessor, postprocessor = smolvla.make_smolvla_pre_post_processors(config, _fixture_stats())
    batch_size = 2
    batch = {
        "observation.state": torch.full((batch_size, 16), 3.0),
        "observation.images.wrist": torch.rand(batch_size, 3, 224, 224),
        "observation.images.right_shoulder": torch.rand(batch_size, 3, 224, 224),
        "observation.images.guide": torch.rand(batch_size, 3, 224, 224),
        "task": ["open the door", "close the door"],
    }

    processed = preprocessor(batch)
    action = postprocessor(torch.zeros(batch_size, 8))

    assert processed["observation.state"].shape == (batch_size, 16)
    assert torch.allclose(processed["observation.state"], torch.ones(batch_size, 16))
    assert processed["observation.language.tokens"].shape == (batch_size, config.tokenizer_max_length)
    assert processed["observation.language.attention_mask"].dtype is torch.bool
    assert processed["task"] == ["open the door\n", "close the door\n"]
    assert torch.allclose(action, torch.full((batch_size, 8), 3.0))


def test_smolvla_loader_sends_stats_only_to_processors(monkeypatch) -> None:
    smolvla = _require_041()
    loader = {}
    factory = {}

    class DatasetMetadataSpy:
        def __init__(self, repo_id, root=None):
            loader["stats_source"] = (repo_id, root)
            self.stats = _fixture_stats()

    class FakeLoadedPolicy:
        config = "loaded-config"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

    class PolicySpy:
        @classmethod
        def from_pretrained(cls, checkpoint):
            loader["checkpoint"] = checkpoint
            return FakeLoadedPolicy()

    def make_processors(config, dataset_stats):
        factory["config"] = config
        factory["stats"] = dataset_stats
        return lambda batch: batch, lambda action: action

    monkeypatch.setattr(smolvla, "LeRobotDatasetMetadata", DatasetMetadataSpy)
    monkeypatch.setattr(smolvla, "SmolVLAPolicy", PolicySpy)
    monkeypatch.setattr(smolvla, "make_smolvla_pre_post_processors", make_processors)

    cfg = SimpleNamespace(
        checkpoint="fixture-checkpoint",
        dataset_stats_repo="fixture-stats",
        dataset_stats_root="",
        state_start_idx=0,
        state_dim=16,
        action_dim=8,
    )
    wrapper = smolvla.SmolVLAWrapper(cfg, torch.device("cpu"))
    wrapper.load()

    assert loader == {"stats_source": ("fixture-stats", None), "checkpoint": "fixture-checkpoint"}
    assert factory["config"] == "loaded-config"
    assert factory["stats"]["observation.state"]["mean"].shape == (16,)
    assert factory["stats"]["action"]["mean"].shape == (8,)
    assert wrapper.policy.is_eval


def test_smolvla_wrapper_routes_wrist_shoulder_and_guide_through_processors() -> None:
    smolvla = _require_041()
    events = []

    def preprocessor(batch):
        events.append("pre")
        assert set(batch).issuperset(
            {
                "observation.state",
                "observation.images.wrist",
                "observation.images.right_shoulder",
                "observation.images.guide",
                "task",
            }
        )
        batch["prepared"] = True
        return batch

    class PolicySpy:
        def select_action(self, batch):
            events.append("policy")
            assert batch["prepared"] is True
            return torch.zeros(1, 8)

    def postprocessor(action):
        events.append("post")
        return action + 2.0

    wrapper = smolvla.SmolVLAWrapper(SimpleNamespace(action_dim=8), torch.device("cpu"))
    wrapper.policy = PolicySpy()
    wrapper._preprocessor = preprocessor
    wrapper._postprocessor = postprocessor

    action = wrapper.select_action(
        torch.zeros(1, 16),
        {
            "observation.images.wrist": torch.zeros(1, 3, 224, 224),
            "observation.images.right_shoulder": torch.zeros(1, 3, 224, 224),
            "observation.images.guide": torch.zeros(1, 3, 224, 224),
        },
        ["open the door"],
    )

    assert events == ["pre", "policy", "post"]
    assert torch.allclose(action, torch.full((1, 8), 2.0))


def test_smolvla_fails_fast_outside_pinned_processor_api(monkeypatch) -> None:
    smolvla = _require_041()
    monkeypatch.setattr(smolvla.lerobot, "__version__", "0.3.4")

    with pytest.raises(RuntimeError, match="requires the pinned LeRobot 0.4.x processor API"):
        smolvla._require_smolvla_processor_pipeline()
