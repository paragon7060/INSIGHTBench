"""Unified environment builder — replaces build_test_env() in all evaluation scripts."""

from __future__ import annotations
from copy import deepcopy
from typing import Tuple


EVAL_GUIDE_CAMERA_HEIGHT = 224
EVAL_GUIDE_CAMERA_WIDTH = 224
_EVAL_CAMERA_VIEWS = ("wrist", "right_shoulder", "left_shoulder", "guide")
_EVAL_NON_RGB_TERMS = (
    "wrist_depth",
    "wrist_semantic",
    "right_shoulder_depth",
    "right_shoulder_semantic",
    "left_shoulder_depth",
    "left_shoulder_semantic",
    "guide_semantic",
)

TASK_LIBS: dict[str, list[str]] = {
    "door":   ["3a", "3b", "3c", "3d"],
    "bottle": ["5a", "5b", "5c", "5d", "5e", "5f", "5g", "5h"],
}

ASSET_DIRS: dict[str, str] = {
    "cabinet": "./Assets/TestSuite/cabinet_suite",
    "door":    "./Assets/TestSuite/door_suite",
    "bottle":  "./Assets/TestSuite/bottle_suite",
}


def _set_eval_guide_camera_resolution(scene_cfg) -> None:
    """Use 224x224 guide-camera observations for evaluation."""
    camera_cfg = getattr(scene_cfg, "camera_guide", None)
    if camera_cfg is None:
        return

    scene_cfg.camera_guide = deepcopy(camera_cfg)
    scene_cfg.camera_guide.height = EVAL_GUIDE_CAMERA_HEIGHT
    scene_cfg.camera_guide.width = EVAL_GUIDE_CAMERA_WIDTH


def configure_eval_camera_pipeline(env_cfg, required_views: set[str]) -> None:
    """Keep only required RGB cameras and render once per environment action."""
    unknown_views = required_views.difference(_EVAL_CAMERA_VIEWS)
    if unknown_views:
        raise ValueError(f"Unknown eval camera views: {sorted(unknown_views)}")

    for view in _EVAL_CAMERA_VIEWS:
        camera_name = f"camera_{view}"
        camera_cfg = getattr(env_cfg.scene, camera_name, None)
        if camera_cfg is None:
            continue
        if view not in required_views:
            setattr(env_cfg.scene, camera_name, None)
            setattr(env_cfg.observations.policy, view, None)
            continue
        camera_cfg.data_types = ["rgb"]

    for term_name in _EVAL_NON_RGB_TERMS:
        setattr(env_cfg.observations.policy, term_name, None)

    env_cfg.sim.render_interval = env_cfg.decimation


def build_env(
    obj_name: str,
    task_idx: int,
    asset_path: str,
    no_guide: bool,
    num_envs: int,
    seed: int = 42,
    pos_rand: bool = False,
) -> Tuple:
    """Build (env_cfg, scene_key, info) for the given task.

    Args:
        obj_name:   Object category — "cabinet" | "door" | "bottle".
        task_idx:   Integer index into the task library for this object.
        asset_path: Asset identifier string (e.g. "99660059960030l").
        no_guide:   When True, the guide asset is not spawned in the scene.
        num_envs:   Number of parallel Isaac Lab environments.
        seed:       RNG seed for reproducibility.
        pos_rand:   When True, apply object position randomization on each reset.
                    When False (default), use fixed positions for reproducibility.

    Returns:
        (env_cfg, scene_key, info) where info is a dict with asset metadata.
    """
    from cfg.BaseCfg import TestTerminationsCfg, CommandsCfg, ObsCfg, ContinuousJointActionsCfg
    from cfg.BaseTaskCfg import EVENT_CLASSES
    from cfg.BaseEnvCfg import DynamicEnvCfg
    from cfg.helper import get_info_test
    from cfg.scene1ExtCfg import make_cabinet_scene_and_reward_cfg
    from cfg.scene3ExtCfg import make_door_scene_cfg
    from cfg.Scene5ExtCfg import make_bottle_scene_and_reward_cfg

    if obj_name not in ASSET_DIRS:
        raise ValueError(f"Unknown object '{obj_name}'. Choose from: {list(ASSET_DIRS)}")

    dir_path = ASSET_DIRS[obj_name]

    if obj_name == "cabinet":
        task_name = str(task_idx)
        info = get_info_test(dir_path, obj_name, asset_path, task_name)
        scene_cfg, reward_cfg = make_cabinet_scene_and_reward_cfg(
            usd_path=info["usd_path"],
            asset_init_pos=info["asset_init_pos"],
            cabinet_scale=info["scale"],
            joint_id=info["joint_id"],
            link_id=info["link_id"],
            guide_init_pos=info["guide_init_pos"],
            guide_init_quat=info["guide_init_quat"],
            joint_type=info["joint_type"],
            no_guide=no_guide,
        )

    elif obj_name == "door":
        task_name = TASK_LIBS["door"][task_idx]
        info = get_info_test(dir_path, obj_name, asset_path, task_name)
        scene_cfg, reward_cfg = make_door_scene_cfg(
            usd_path=info["usd_path"],
            guide_path=info["guide_path"],
            asset_init_pos=info["asset_init_pos"],
            guide_init_pos=info["guide_init_pos"],
            door_scale=info["scale"],
            no_guide=no_guide,
        )

    elif obj_name == "bottle":
        task_name = TASK_LIBS["bottle"][task_idx]
        info = get_info_test(dir_path, obj_name, asset_path, task_name)
        scene_cfg, reward_cfg = make_bottle_scene_and_reward_cfg(
            usd_path=info["usd_path"],
            guide_path=info["guide_path"],
            guide_scale=info["guide_scale"],
            asset_init_pos=info["asset_init_pos"],
            guide_init_pos=info["guide_init_pos"],
            close_mode=info["close_mode"],
            no_guide=no_guide,
        )

    scene_key = info["scene_key"]
    _set_eval_guide_camera_resolution(scene_cfg)
    scene_cfg.num_envs = num_envs
    scene_cfg.env_spacing = 3.0

    env_cfg = DynamicEnvCfg(
        scene=scene_cfg,
        observations=ObsCfg(),
        commands=CommandsCfg(),
        actions=ContinuousJointActionsCfg(),
        rewards=reward_cfg,
        terminations=TestTerminationsCfg(),
        events=EVENT_CLASSES[scene_key[0] + ("" if pos_rand else "test")](),
    )
    env_cfg.seed = seed
    env_cfg.episode_length_s = 100
    env_cfg.decimation = 12
    env_cfg.sim.dt = 1 / 120
    env_cfg.scene.robot.actuators["panda_hand"].stiffness = 2e3
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**18

    return env_cfg, scene_key, info
