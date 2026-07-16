"""Tests for the opt-in evaluation camera pipeline."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from insightbench.envs.builder import configure_eval_camera_pipeline


def _env_cfg():
    policy = SimpleNamespace(
        wrist=object(),
        wrist_depth=object(),
        wrist_semantic=object(),
        right_shoulder=object(),
        right_shoulder_depth=object(),
        right_shoulder_semantic=object(),
        left_shoulder=object(),
        left_shoulder_depth=object(),
        left_shoulder_semantic=object(),
        guide=object(),
        guide_semantic=object(),
    )
    scene = SimpleNamespace(
        camera_wrist=SimpleNamespace(data_types=["rgb", "depth"]),
        camera_right_shoulder=SimpleNamespace(data_types=["rgb", "depth"]),
        camera_left_shoulder=SimpleNamespace(data_types=["rgb", "depth"]),
        camera_guide=SimpleNamespace(data_types=["rgb", "depth"]),
    )
    return SimpleNamespace(
        scene=scene,
        observations=SimpleNamespace(policy=policy),
        sim=SimpleNamespace(render_interval=1),
        decimation=12,
    )


def test_eval_camera_pipeline_keeps_only_required_rgb_views() -> None:
    cfg = _env_cfg()

    configure_eval_camera_pipeline(cfg, {"wrist", "right_shoulder", "guide"})

    assert cfg.scene.camera_wrist.data_types == ["rgb"]
    assert cfg.scene.camera_right_shoulder.data_types == ["rgb"]
    assert cfg.scene.camera_guide.data_types == ["rgb"]
    assert cfg.scene.camera_left_shoulder is None
    assert cfg.observations.policy.left_shoulder is None
    assert cfg.observations.policy.wrist_depth is None
    assert cfg.observations.policy.right_shoulder_semantic is None
    assert cfg.observations.policy.guide_semantic is None
    assert cfg.sim.render_interval == cfg.decimation


def test_eval_camera_pipeline_rejects_unknown_views() -> None:
    with pytest.raises(ValueError, match="Unknown eval camera views"):
        configure_eval_camera_pipeline(_env_cfg(), {"overhead"})
