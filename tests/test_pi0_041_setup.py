"""Runtime import coverage for the documented LeRobot 0.4.1 Pi0 layout."""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def test_pi0_wrapper_load_setup_uses_embedded_041_paligemma(monkeypatch) -> None:
    """Exercise Pi0Wrapper.load without weights, network, or a GPU."""
    lerobot = pytest.importorskip("lerobot")
    if lerobot.__version__ != "0.4.1":
        pytest.skip("requires the documented LeRobot 0.4.1 baseline")

    pi0 = importlib.import_module("insightbench.policies.pi0")
    modeling = importlib.import_module("lerobot.policies.pi0.modeling_pi0")
    assert hasattr(modeling, "PaliGemmaWithExpertModel")
    assert importlib.util.find_spec("lerobot.policies.pi0.paligemma_with_expert") is None

    class FakeLoadedPolicy:
        config = "loaded-config"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.is_eval = True
            return self

    class FakePI0Policy:
        _load_as_safetensor = object()
        loaded = None

        @classmethod
        def from_pretrained(cls, checkpoint, config):
            cls.loaded = (checkpoint, config)
            return FakeLoadedPolicy()

    processor_factory = {}

    def make_processors(config, dataset_stats):
        processor_factory["config"] = config
        processor_factory["stats"] = dataset_stats
        return lambda batch: batch, lambda action: action

    monkeypatch.setattr(pi0, "PI0Policy", FakePI0Policy)
    monkeypatch.setattr(pi0.Pi0Wrapper, "_load_stats", staticmethod(lambda cfg: {}))
    monkeypatch.setattr(pi0, "_load_pi0_checkpoint_config", lambda checkpoint: "decoded-config")
    monkeypatch.setattr(pi0, "make_pi0_pre_post_processors", make_processors)

    cfg = SimpleNamespace(
        checkpoint="fixture-checkpoint",
        checkpoint_subfolder="",
        state_start_idx=0,
        state_dim=8,
        action_dim=8,
    )
    wrapper = pi0.Pi0Wrapper(cfg, torch.device("cpu"))
    wrapper.load()

    assert FakePI0Policy.loaded == ("fixture-checkpoint", "decoded-config")
    assert processor_factory == {"config": "loaded-config", "stats": {}}
    assert wrapper.policy.is_eval
    assert getattr(FakePI0Policy, "_insightbench_compat_installed")


def test_pi0_transformers_bridge_preserves_target_train_contract() -> None:
    lerobot = pytest.importorskip("lerobot")
    if lerobot.__version__ != "0.4.1":
        pytest.skip("requires the documented LeRobot 0.4.1 baseline")

    pi0 = importlib.import_module("insightbench.policies.pi0")
    modeling = importlib.import_module("lerobot.policies.pi0.modeling_pi0")

    pi0._install_pi0_transformers_compat()

    # 0.4.1 does not expose the 0.3.4 freeze/train configuration on this
    # embedded class, so evaluation must retain nn.Module.train unchanged.
    assert modeling.PaliGemmaWithExpertModel.train is torch.nn.Module.train


def test_pi0_03x_keeps_internal_stats_loading(monkeypatch) -> None:
    """The processor migration must not change the supported 0.3.x load path."""
    pi0 = importlib.import_module("insightbench.policies.pi0")

    class FakeLoadedPolicy:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FakePI0Policy:
        loaded = None

        @classmethod
        def from_pretrained(cls, checkpoint, dataset_stats):
            cls.loaded = (checkpoint, dataset_stats)
            return FakeLoadedPolicy()

    monkeypatch.setattr(pi0, "PI0Policy", FakePI0Policy)
    monkeypatch.setattr(pi0, "_LEROBOT_VERSION", pi0.Version("0.3.4"))
    monkeypatch.setattr(pi0.Pi0Wrapper, "_load_stats", staticmethod(lambda cfg: {"state": {}}))
    monkeypatch.setattr(
        pi0,
        "_load_pi0_checkpoint_config",
        lambda checkpoint: (_ for _ in ()).throw(AssertionError("0.3.x must not run 0.4.x config migration")),
    )

    cfg = SimpleNamespace(
        checkpoint="fixture-checkpoint",
        checkpoint_subfolder="",
        state_start_idx=0,
        state_dim=8,
        action_dim=8,
    )
    wrapper = pi0.Pi0Wrapper(cfg, torch.device("cpu"))
    wrapper.load()

    assert FakePI0Policy.loaded == ("fixture-checkpoint", {"state": {}})
    assert wrapper._preprocessor is None
    assert wrapper._postprocessor is None
