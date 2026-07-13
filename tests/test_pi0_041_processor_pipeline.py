"""Processor-pipeline coverage for the pinned LeRobot 0.4.1 Pi0 runtime."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")
FIXTURE = Path(__file__).parent / "fixtures" / "pi0_034_checkpoint_config.json"


def _require_041():
    lerobot = pytest.importorskip("lerobot")
    if lerobot.__version__ != "0.4.1":
        pytest.skip("requires the documented LeRobot 0.4.1 baseline")
    return importlib.import_module("insightbench.policies.pi0")


def _fixture_config(pi0):
    with FIXTURE.open() as file:
        raw_config = json.load(file)
    normalized, _ = pi0._normalize_pi0_checkpoint_config(raw_config)
    return pi0._decode_pi0_checkpoint_config(normalized)


def _fixture_stats() -> dict:
    return {
        "observation.state": {
            "mean": torch.ones(8),
            "std": torch.full((8,), 2.0),
        },
        "action": {
            "mean": torch.full((8,), 3.0),
            "std": torch.full((8,), 4.0),
        },
    }


def test_official_pi0_processors_preserve_batched_input_contract(monkeypatch) -> None:
    pi0 = _require_041()
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

    config = _fixture_config(pi0)
    preprocessor, postprocessor = pi0.make_pi0_pre_post_processors(config, _fixture_stats())
    batch_size = 2
    batch = {
        "observation.state": torch.full((batch_size, 8), 3.0),
        "observation.images.guide": torch.rand(batch_size, 3, 224, 224),
        "observation.images.right_shoulder": torch.rand(batch_size, 3, 224, 224),
        "observation.images.wrist": torch.rand(batch_size, 3, 224, 224),
        "task": ["open the door", "close the door"],
    }

    processed = preprocessor(batch)
    action = postprocessor(torch.zeros(batch_size, 8))

    assert processed["observation.state"].shape == (batch_size, 8)
    assert torch.allclose(processed["observation.state"], torch.ones(batch_size, 8))
    assert processed["observation.language.tokens"].shape == (batch_size, config.tokenizer_max_length)
    assert processed["observation.language.attention_mask"].dtype is torch.bool
    assert processed["task"] == ["open the door\n", "close the door\n"]
    assert torch.allclose(action, torch.full((batch_size, 8), 3.0))


def test_wrapper_routes_action_through_pre_policy_post() -> None:
    pi0 = _require_041()
    events = []

    def preprocessor(batch):
        events.append("pre")
        assert batch["task"] == ["open the door"]
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

    cfg = SimpleNamespace(action_dim=8)
    wrapper = pi0.Pi0Wrapper(cfg, torch.device("cpu"))
    wrapper.policy = PolicySpy()
    wrapper._uses_processor_pipeline = True
    wrapper._preprocessor = preprocessor
    wrapper._postprocessor = postprocessor

    action = wrapper.select_action(
        torch.zeros(1, 8),
        {"observation.images.guide": torch.zeros(1, 3, 224, 224)},
        ["open the door"],
    )

    assert events == ["pre", "policy", "post"]
    assert torch.allclose(action, torch.full((1, 8), 2.0))
