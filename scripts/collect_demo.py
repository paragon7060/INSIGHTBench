"""InsightBench demonstration data collection script.

Collects expert demonstrations using CuRobo motion planning and saves them
as a LeRobot dataset (local or HuggingFace Hub).

Usage:
    python scripts/collect_demo.py \\
        --object door --asset_id <ASSET_ID> --scene_key 3ext \\
        --dataset_name YOUR_HF_USERNAME/INSIGHT-demo \\
        --num_envs 8 --target_episodes 200 \\
        --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher
from cfg.ActionRangeCfg import SCENE_ACTION_CONFIG

parser = argparse.ArgumentParser(description="InsightBench demo collection.")
parser.add_argument("--object",          type=str, required=True, choices=["cabinet", "door", "bottle"])
parser.add_argument("--asset_id",        type=str, required=True, help="Asset identifier (e.g. 'Table-19179-...').")
parser.add_argument("--scene_key",       type=str, required=True, help="Scene key (e.g. '1ext', '3ext', '5ext').")
parser.add_argument("--dataset_name",    type=str, required=True, help="HuggingFace repo-id or local name.")
parser.add_argument("--num_envs",        type=int, default=8)
parser.add_argument("--fps",             type=int, default=10)
parser.add_argument("--target_episodes", type=int, default=100, help="Stop after this many successful episodes.")
parser.add_argument("--max_loops",       type=int, default=500,  help="Hard stop after this many collection loops.")
parser.add_argument("--no_guide",        action="store_true",    help="Collect without guide asset in scene.")
pos_rand_group = parser.add_mutually_exclusive_group()
pos_rand_group.add_argument("--pos_rand", dest="pos_rand", action="store_true", default=True, help="Enable object base position randomization during collection (default).")
pos_rand_group.add_argument("--no_pos_rand", "--fixed_pos", dest="pos_rand", action="store_false", help="Disable object base position randomization during collection.")
parser.add_argument("--push_to_hub",     action="store_true",    help="Push dataset to HuggingFace Hub when done.")
parser.add_argument("--save_video",      action="store_true",    help="Save a mosaic overview video per asset.")
parser.add_argument("--video_dir",       type=str, default="./outputs/collect_videos")
parser.add_argument("--asset_dir",       type=str, default=None, help="Override collect asset root dir.")
parser.add_argument("--debug_collect",   action="store_true",    help="Print detailed collect smoke/debug logs.")
parser.add_argument("--progress_interval", type=int, default=10, help="Print trajectory execution progress every N steps when --debug_collect is enabled.")
parser.add_argument("--skill_timeout_s", type=float, default=0.0, help="Abort one skill execution after this many wall-clock seconds; 0 disables the guard.")
parser.add_argument("--no_frame_write", "--dry_run_frames", dest="no_frame_write", action="store_true", help="Run env.step but skip LeRobot frame/image writes for smoke debugging.")
parser.add_argument("--collect_decimation", type=int, default=300, help="Physics substeps per collect action step.")
parser.add_argument("--collect_render_interval", type=int, default=0, help="Render interval in physics steps; 0 matches collect_decimation for one camera render per action step.")
parser.add_argument("--smoke_action_steps", type=int, default=0, help="Smoke-only: execute this many planned action steps, then finish without continuing the trajectory. 0 disables smoke mode.")
parser.add_argument("--smoke_decimation", type=int, default=10, help="Smoke-only physics substeps per action step (1-20; default: 10).")
parser.add_argument("--smoke_step_timeout_s", type=float, default=60.0, help="Smoke-only per env.step deadline checked inside the physics loop.")
parser.add_argument("--smoke_save_episode", action="store_true", help="Smoke-only: save the partial frame-write episode after the action-step limit. Requires frame writing.")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
app_launcher  = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-launch imports ───────────────────────────────────────────────────────
import math

import cv2
import numpy as np
import PIL.Image
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import disable_caching
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from insightbench.utils.lerobot_compat import add_frame as _lerobot_add_frame, lerobot_version
from insightbench.utils.collect_smoke import CollectTiming, resolve_collect_timing
print(f"[collect_demo] lerobot version: {lerobot_version()}")

from cfg.BaseCfg import ActionsCfg, ContinuousJointActionsCfg, TerminationsCfg, CommandsCfg, ObsCfg
from cfg.BaseEnvCfg import DynamicEnvCfg
from cfg.BaseTaskCfg import EVENT_CLASSES, REWARD_FOR_SCENE_KEY
from cfg.helper import get_info_collect
from cfg.scene1ExtCfg import make_cabinet_scene_and_reward_cfg
from cfg.scene3ExtCfg import make_door_scene_cfg
from cfg.Scene5ExtCfg import make_bottle_scene_and_reward_cfg
from custom_lab.envs.manager_based_rl_step_env import ManagerBasedContinuousEnv
from demo_gen.action_sampling.sample_actions import (
    sample_actions_for_cabinet,
    sample_actions_from_bbox,
    sample_actions_from_body_pose,
)
from interact.motion_generator import TrajectoryGenerator

disable_caching()


# ─── Video mosaic recorder ─────────────────────────────────────────────────────

class _MosaicVideoRecorder:
    """Records a grid of all env viewpoints into a single overview video."""

    def __init__(self, path: str, fps: int, num_envs: int):
        self._path = path
        self._fps  = fps
        self._cols = math.ceil(math.sqrt(num_envs))
        self._rows = math.ceil(num_envs / self._cols)
        self._writer = None

    def add_frame(self, env_images: list[np.ndarray]) -> None:
        imgs = [np.clip(i, 0, 255).astype(np.uint8) if i.dtype != np.uint8 else i for i in env_images]
        h, w = imgs[0].shape[:2]
        while len(imgs) < self._rows * self._cols:
            imgs.append(np.zeros((h, w, 3), dtype=np.uint8))
        grid = np.vstack([np.hstack(imgs[r * self._cols:(r + 1) * self._cols]) for r in range(self._rows)])
        if self._writer is None:
            gh, gw = grid.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"avc1")
            self._writer = cv2.VideoWriter(self._path, fourcc, self._fps, (gw, gh))
            if not self._writer.isOpened():
                self._writer = cv2.VideoWriter(self._path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (gw, gh))
        self._writer.write(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"[Video] Saved overview: {self._path}")


# ─── Dataset helpers ───────────────────────────────────────────────────────────

def _save_image(image_array: np.ndarray, fpath: str) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    if image_array.ndim != 3:
        return
    channels = image_array.shape[-1]
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels)
    if mode == "L":
        image_array = image_array.squeeze(-1)
    if mode:
        PIL.Image.fromarray(image_array, mode=mode).save(fpath)


def _add_frame(dataset: LeRobotDataset, episode_buffer: dict, frame: dict, task: str) -> None:
    """Write one frame into a per-env episode buffer.

    Uses lerobot_compat.add_frame() to handle v0.3.x / v0.4.x API differences.
    Multi-env parallel collection requires managing explicit episode_buffer objects
    because the built-in dataset.episode_buffer is single-instance.
    """
    for k in frame:
        if isinstance(frame[k], torch.Tensor):
            frame[k] = frame[k].numpy()

    idx = episode_buffer["size"]
    timestamp = idx / dataset.fps

    # Manually populate the episode_buffer (same internal structure in both versions)
    episode_buffer["frame_index"].append(idx)
    episode_buffer["timestamp"].append(timestamp)
    episode_buffer["task"].append(task)

    for key in frame:
        if key not in dataset.features:
            raise ValueError(f"Frame key '{key}' not in dataset features.")
        if dataset.features[key]["dtype"] in ("image", "video"):
            img_path = dataset._get_image_file_path(episode_buffer["episode_index"], key, idx)
            if idx == 0:
                img_path.parent.mkdir(parents=True, exist_ok=True)
            _save_image(frame[key], str(img_path))
            episode_buffer[key].append(str(img_path))
        else:
            episode_buffer[key].append(frame[key])

    episode_buffer["size"] += 1


def _round_list(value, digits: int = 4) -> list[float]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    elif isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    return [round(float(v), digits) for v in value]


def _object_root_state_env0(env: ManagerBasedContinuousEnv, obj_name: str) -> list[float] | None:
    scene_name = {"cabinet": "cabinet", "door": "door", "bottle": "bottle"}[obj_name]
    try:
        obj = env.scene[scene_name]
    except Exception:
        return None
    return _round_list(obj.data.root_state_w[0])


def _print_collect_action_debug(
    env: ManagerBasedContinuousEnv,
    obj_name: str,
    asset_id: str,
    scene_key: str,
    event_key: str,
    pos_rand: bool,
    step_idx: int,
    actions_per_env: torch.Tensor,
) -> None:
    env0 = actions_per_env[0].detach().cpu()
    skill_id = int(env0[0].item())
    print(
        f"[CollectDebug] object={obj_name} asset_id={asset_id} scene_key={scene_key} "
        f"event_key={event_key} pos_rand={pos_rand} step_idx={step_idx}",
        flush=True,
    )
    print(
        f"[CollectDebug] action env0 skill={skill_id} "
        f"target_pos={_round_list(env0[1:4])} target_quat={_round_list(env0[4:8])}",
        flush=True,
    )
    target_pos = actions_per_env[:, 1:4].detach().cpu()
    print(
        f"[CollectDebug] action target_pos_min={_round_list(target_pos.min(dim=0).values)} "
        f"target_pos_max={_round_list(target_pos.max(dim=0).values)}",
        flush=True,
    )
    print(f"[CollectDebug] object_root env0={_object_root_state_env0(env, obj_name)}", flush=True)
    print(f"[CollectDebug] robot_joint_pos env0={_round_list(env.scene['robot'].data.joint_pos[0, :9])}", flush=True)


def _clear_episode(dataset: LeRobotDataset, episode_buffer: dict) -> None:
    for cam_key in dataset.meta.camera_keys:
        img_dir = dataset._get_image_file_path(episode_buffer["episode_index"], cam_key, 0).parent
        if img_dir.is_dir():
            shutil.rmtree(img_dir)


def _finish_smoke_episode(
    dataset: LeRobotDataset,
    episode_buffers: list[dict],
    *,
    no_frame_write: bool,
    smoke_save_episode: bool,
    executed_action_steps: int,
) -> int:
    """Finalize a bounded smoke rollout without applying production success rules."""
    if no_frame_write:
        for episode_buffer in episode_buffers:
            _clear_episode(dataset, episode_buffer)
        print(
            f"[CollectSmoke] completed action_steps={executed_action_steps}; "
            "no_frame_write=True, dataset.save_episode skipped",
            flush=True,
        )
        return 0

    if smoke_save_episode:
        for episode_buffer in episode_buffers:
            dataset.save_episode(episode_buffer)
        print(
            f"[CollectSmoke] completed action_steps={executed_action_steps}; "
            f"saved {len(episode_buffers)} partial frame-write episode(s)",
            flush=True,
        )
        return len(episode_buffers)

    for episode_buffer in episode_buffers:
        _clear_episode(dataset, episode_buffer)
    print(
        f"[CollectSmoke] completed action_steps={executed_action_steps}; "
        "frame writes validated, dataset.save_episode skipped",
        flush=True,
    )
    return 0


def _step_env_with_collect_trace(
    env: ManagerBasedContinuousEnv,
    action_batch: torch.Tensor,
    *,
    phase: str,
    step_idx: int,
    action_idx: int,
    trace_steps: bool,
    smoke_step_timeout_s: float | None,
):
    """Run one env step with optional smoke deadline and before/after evidence."""
    if trace_steps:
        print(
            f"[EnvStepBegin] phase={phase} step_idx={step_idx} action_idx={action_idx} "
            f"shape={tuple(action_batch.shape)} decimation={env.cfg.decimation} "
            f"render_interval={env.cfg.sim.render_interval}",
            flush=True,
        )

    if smoke_step_timeout_s is not None:
        env._collect_step_deadline_s = time.perf_counter() + smoke_step_timeout_s
    env_step_start = time.perf_counter()
    try:
        step_result = env.step(action_batch)
    except TimeoutError as exc:
        print(
            f"[CollectSmokeAbort] phase={phase} step_idx={step_idx} action_idx={action_idx} "
            f"reason={exc}",
            flush=True,
        )
        raise
    finally:
        env._collect_step_deadline_s = None

    elapsed = time.perf_counter() - env_step_start
    if trace_steps:
        print(
            f"[EnvStepEnd] phase={phase} step_idx={step_idx} action_idx={action_idx} "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )
    return step_result, elapsed


def _create_or_load_dataset(name: str, fps: int, root: str) -> LeRobotDataset:
    features = {
        "observation.images.wrist":          {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
        "observation.images.right_shoulder":  {"dtype": "image", "shape": (224, 224, 3), "names": ["height", "width", "channel"]},
        "observation.images.guide":           {"dtype": "image", "shape": (672, 672, 3), "names": ["height", "width", "channel"]},
        "observation.state":                  {"dtype": "float32", "shape": (16,),        "names": ["state"]},
        "action":                             {"dtype": "float32", "shape": (8,),         "names": ["action"]},
    }
    dataset_path = Path(root)
    if (dataset_path / "meta" / "tasks.jsonl").exists():
        print(f"[Dataset] Resuming existing dataset at {root}")
        return LeRobotDataset(name, root=root)
    # Remove incomplete dataset dir if it exists but lacks required meta files
    if dataset_path.exists():
        import shutil as _shutil
        print(f"[Dataset] Removing incomplete dataset at {root}")
        _shutil.rmtree(dataset_path)
    print(f"[Dataset] Creating new dataset: {name}")
    return LeRobotDataset.create(name, fps=fps, root=root, features=features, image_writer_threads=4)


# ─── Environment builder ───────────────────────────────────────────────────────

def _resolve_collect_asset_dir(obj_name: str, asset_dir: str | None) -> str:
    if asset_dir:
        return asset_dir

    candidates = {
        "cabinet": ("./Assets/TrainSuite/cabinet_suite", "./Assets/PartManip/drawer/train"),
        "door":    ("./Assets/TrainSuite/door_suite", "./Assets/AdaManip/door"),
        "bottle":  ("./Assets/TrainSuite/bottle_suite", "./Assets/AdaManip/bottle"),
    }[obj_name]

    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _collect_event_key(resolved_scene_key: str, pos_rand: bool) -> str:
    if pos_rand:
        return resolved_scene_key[0]
    return f"{resolved_scene_key[0]}nopos"


def _build_collect_env(
    obj_name: str,
    asset_id: str,
    scene_key: str,
    num_envs: int,
    no_guide: bool,
    pos_rand: bool,
    collect_decimation: int,
    collect_render_interval: int,
    asset_dir: str | None = None,
):
    dir_path = _resolve_collect_asset_dir(obj_name, asset_dir)

    if obj_name == "cabinet":
        info = get_info_collect(dir_path, obj_name, asset_id, scene_key)
        scene_cfg, reward_cfg = make_cabinet_scene_and_reward_cfg(
            usd_path=info["usd_path"], asset_init_pos=info["asset_init_pos"],
            cabinet_scale=info["scale"], joint_id=info["joint_id"],
            link_id=info["link_id"], guide_init_pos=info["guide_init_pos"],
            guide_init_quat=info["guide_init_quat"], joint_type=info["joint_type"],
            no_guide=no_guide,
        )
    elif obj_name == "door":
        info = get_info_collect(dir_path, obj_name, asset_id, scene_key)
        scene_cfg, reward_cfg = make_door_scene_cfg(
            usd_path=info["usd_path"], guide_path=info["guide_path"],
            asset_init_pos=info["asset_init_pos"], guide_init_pos=info["guide_init_pos"],
            door_scale=info["scale"], no_guide=no_guide,
        )
    elif obj_name == "bottle":
        info = get_info_collect(dir_path, obj_name, asset_id, scene_key)
        scene_cfg, reward_cfg = make_bottle_scene_and_reward_cfg(
            usd_path=info["usd_path"], guide_path=info["guide_path"],
            guide_scale=info["guide_scale"], asset_init_pos=info["asset_init_pos"],
            guide_init_pos=info["guide_init_pos"], close_mode=info["close_mode"],
            no_guide=no_guide,
        )
    else:
        raise ValueError(f"Unknown object '{obj_name}'")

    resolved_scene_key = info["scene_key"]
    event_key = _collect_event_key(resolved_scene_key, pos_rand)
    if event_key not in EVENT_CLASSES:
        raise KeyError(f"Unknown collect event key '{event_key}' for scene_key '{resolved_scene_key}'")
    print(f"[Collect] event_key={event_key} pos_rand={pos_rand}", flush=True)

    scene_cfg.num_envs  = num_envs
    scene_cfg.env_spacing = 3.0

    env_cfg = DynamicEnvCfg(
        scene=scene_cfg,
        observations=ObsCfg(),
        commands=CommandsCfg(),
        actions=ContinuousJointActionsCfg(),
        rewards=reward_cfg,
        terminations=TerminationsCfg(),
        events=EVENT_CLASSES[event_key](),
    )
    if collect_decimation <= 0:
        raise ValueError(f"collect_decimation must be positive, got {collect_decimation}")
    if collect_render_interval < 0:
        raise ValueError(f"collect_render_interval must be non-negative, got {collect_render_interval}")

    env_cfg.episode_length_s = 50
    env_cfg.decimation = collect_decimation
    env_cfg.sim.render_interval = collect_render_interval or collect_decimation
    env_cfg.sim.dt = 1 / 60
    env_cfg.scene.robot.actuators["panda_hand"].stiffness = 2e3
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**18
    print(
        f"[Collect] decimation={env_cfg.decimation} render_interval={env_cfg.sim.render_interval}",
        flush=True,
    )

    return env_cfg, resolved_scene_key, info


# ─── Episode collection ────────────────────────────────────────────────────────

def _collect_one_episode(
    env: ManagerBasedContinuousEnv,
    obj_name: str,
    asset_id: str,
    scene_key: str,
    event_key: str,
    pos_rand: bool,
    curobo_mg: TrajectoryGenerator,
    info: dict,
    dataset: LeRobotDataset,
    task_name: str,
    video_recorder: _MosaicVideoRecorder | None,
    debug_collect: bool = False,
    progress_interval: int = 10,
    skill_timeout_s: float | None = None,
    no_frame_write: bool = False,
    smoke_action_steps: int = 0,
    smoke_step_timeout_s: float | None = None,
    smoke_save_episode: bool = False,
) -> int:
    obs_batch, _ = env.reset()
    episode_buffers = [dataset.create_episode_buffer(dataset.meta.total_episodes + i) for i in range(env.num_envs)]
    success_tracker = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    trace_steps = debug_collect or smoke_action_steps > 0

    skill_seq = SCENE_ACTION_CONFIG[scene_key]["skill_sequence"]
    reward_threshold = REWARD_FOR_SCENE_KEY[scene_key[0]]

    curobo_mg.gripper_reset()
    # Hold current joint positions for initial settle steps (no CuRobo needed)
    noop_batch = env.scene["robot"].data.joint_pos[:, :9].clone()
    for settle_idx in range(5):
        try:
            (obs_batch, _, _, _, _), _ = _step_env_with_collect_trace(
                env,
                noop_batch,
                phase="settle",
                step_idx=-1,
                action_idx=settle_idx + 1,
                trace_steps=trace_steps,
                smoke_step_timeout_s=smoke_step_timeout_s,
            )
        except TimeoutError:
            for episode_buffer in episode_buffers:
                _clear_episode(dataset, episode_buffer)
            return 0

    executed_action_steps = 0
    for step_idx in range(len(skill_seq)):
        if obj_name == "cabinet":
            action_list = info.get("action_list")
            handle_type = info.get("handle_type")
            actions_per_env = sample_actions_for_cabinet(env, scene_key, step_idx, env.num_envs, action_list, env.device, handle_type)
            skill_mask = (actions_per_env[:, 0] == 1) | (actions_per_env[:, 0] == 0)
            if skill_mask.any():
                asset_root_pos = env.scene["cabinet"].data.root_state_w[:, :3] - env.scene.env_origins
                actions_per_env = actions_per_env.clone()
                actions_per_env[skill_mask, 1:4] += asset_root_pos[skill_mask]
        elif obj_name == "door":
            actions_per_env = sample_actions_from_body_pose(env, scene_key, step_idx, env.num_envs, env.device)
        else:
            bbox_info = info.get("cover_points")["cover"]
            actions_per_env = sample_actions_from_bbox(env, scene_key, bbox_info, step_idx, env.num_envs, env.device)
            skill_mask = (actions_per_env[:, 0] == 1) | (actions_per_env[:, 0] == 0)
            if skill_mask.any():
                asset_root_pos = env.scene["bottle"].data.root_state_w[:, :3] - env.scene.env_origins
                actions_per_env = actions_per_env.clone()
                actions_per_env[skill_mask, 1:4] += asset_root_pos[skill_mask]

        if debug_collect:
            _print_collect_action_debug(env, obj_name, asset_id, scene_key, event_key, pos_rand, step_idx, actions_per_env)

        print(f"[DBG] skill_seq step_idx={step_idx} skill={skill_seq[step_idx]} calling curobo...", flush=True)
        curobo_start = time.perf_counter()
        cmd_plan = curobo_mg.command(actions_per_env)
        curobo_elapsed = time.perf_counter() - curobo_start

        # Compute max trajectory length across all envs
        max_T = max(
            (cmd_plan[i].position.shape[0] if cmd_plan[i] is not None else 1)
            for i in range(env.num_envs)
        )
        none_count = sum(1 for i in range(env.num_envs) if cmd_plan[i] is None)
        print(f"[DBG] max_T={max_T} none_envs={none_count}/{env.num_envs}", flush=True)
        plan_lengths = [(cmd_plan[i].position.shape[0] if cmd_plan[i] is not None else 0) for i in range(env.num_envs)]
        if debug_collect:
            print(
                f"[CollectDebug] curobo_done elapsed={curobo_elapsed:.2f}s "
                f"lengths={plan_lengths} none={none_count}/{env.num_envs}",
                flush=True,
            )
            print(f"[CollectDebug] plan_debug={getattr(curobo_mg, 'last_plan_debug', {})}", flush=True)
        traj_counters = [0] * env.num_envs

        # Execute all envs in lockstep: each env advances its own time index
        _max_rew_this_skill = 0.0
        exec_start = time.perf_counter()
        last_env_step_s = 0.0
        last_frame_write_s = 0.0
        for t in range(max_T):
            elapsed = time.perf_counter() - exec_start
            if skill_timeout_s and skill_timeout_s > 0 and elapsed > skill_timeout_s:
                print(
                    f"[CollectTimeout] step_idx={step_idx} skill={skill_seq[step_idx]} "
                    f"t={t}/{max_T} elapsed={elapsed:.1f}s timeout={skill_timeout_s:g}s aborting episode",
                    flush=True,
                )
                for episode_buffer in episode_buffers:
                    _clear_episode(dataset, episode_buffer)
                return 0

            action_parts = []
            for env_i, traj_js in enumerate(cmd_plan):
                if traj_js is None:
                    action_parts.append(noop_batch[env_i])
                else:
                    t_idx = min(traj_counters[env_i], traj_js.position.shape[0] - 1)
                    action_parts.append(traj_js.position[t_idx])
                traj_counters[env_i] += 1
            action_batch = torch.stack(action_parts)
            try:
                (obs_batch, rew_b, done_b, _, _), last_env_step_s = _step_env_with_collect_trace(
                    env,
                    action_batch,
                    phase="trajectory",
                    step_idx=step_idx,
                    action_idx=t + 1,
                    trace_steps=trace_steps,
                    smoke_step_timeout_s=smoke_step_timeout_s,
                )
            except TimeoutError:
                for episode_buffer in episode_buffers:
                    _clear_episode(dataset, episode_buffer)
                return 0

            step_rew = env.reward_manager._step_reward.squeeze(-1)
            _rew_val = step_rew.max().item()
            if _rew_val > _max_rew_this_skill:
                _max_rew_this_skill = _rew_val
            if _rew_val > 0.001:
                print(f"[Reward] step_idx={step_idx} max={_rew_val:.4f} threshold={reward_threshold}", flush=True)
            success_tracker |= step_rew > reward_threshold

            frame_write_start = time.perf_counter()
            if video_recorder is not None:
                imgs = [obs_batch["policy"]["wrist"][i].cpu().numpy() for i in range(env.num_envs)]
                video_recorder.add_frame(imgs)

            if not no_frame_write:
                state = torch.cat([
                    obs_batch["policy"]["ee_position"],
                    obs_batch["policy"]["ee_quat"],
                    obs_batch["policy"]["joint_pos"],
                ], dim=1)

                for i in range(env.num_envs):
                    frame = {
                        "observation.images.wrist":         obs_batch["policy"]["wrist"][i].cpu().numpy(),
                        "observation.images.right_shoulder": obs_batch["policy"]["right_shoulder"][i].cpu().numpy(),
                        "observation.images.guide":          obs_batch["policy"]["guide"][i].cpu().numpy(),
                        "observation.state":                 state[i].cpu().numpy(),
                        "action":                            action_batch[i, :8].cpu().numpy(),
                    }
                    _add_frame(dataset, episode_buffers[i], frame, task_name)
            last_frame_write_s = time.perf_counter() - frame_write_start

            if debug_collect and progress_interval > 0 and (t == 0 or (t + 1) % progress_interval == 0 or t + 1 == max_T):
                done_count = int(done_b.sum().item()) if isinstance(done_b, torch.Tensor) else int(sum(done_b))
                success_count = int(success_tracker.sum().item())
                print(
                    f"[ExecProgress] step_idx={step_idx} t={t + 1}/{max_T} "
                    f"elapsed={time.perf_counter() - exec_start:.1f}s "
                    f"env_step={last_env_step_s:.2f}s frame_write={last_frame_write_s:.2f}s "
                    f"max_rew={_rew_val:.4f} success={success_count}/{env.num_envs} done={done_count}/{env.num_envs} "
                    f"action_env0={_round_list(action_batch[0, :8])}",
                    flush=True,
                )

            executed_action_steps += 1
            if smoke_action_steps and executed_action_steps >= smoke_action_steps:
                return _finish_smoke_episode(
                    dataset,
                    episode_buffers,
                    no_frame_write=no_frame_write,
                    smoke_save_episode=smoke_save_episode,
                    executed_action_steps=executed_action_steps,
                )

        print(f"[Skill done] step_idx={step_idx} skill={skill_seq[step_idx]} max_rew={_max_rew_this_skill:.6f}", flush=True)

    num_success = 0
    if no_frame_write:
        if debug_collect:
            print("[CollectDebug] no_frame_write=True; skipping dataset.save_episode for dry-run episode", flush=True)
        for episode_buffer in episode_buffers:
            _clear_episode(dataset, episode_buffer)
        return 0

    for i in range(env.num_envs):
        if success_tracker[i].item():
            dataset.save_episode(episode_buffers[i])
            num_success += 1
        else:
            _clear_episode(dataset, episode_buffers[i])

    return num_success


# ─── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = args_cli

    try:
        collect_timing: CollectTiming = resolve_collect_timing(
            collect_decimation=args.collect_decimation,
            collect_render_interval=args.collect_render_interval,
            smoke_action_steps=args.smoke_action_steps,
            smoke_decimation=args.smoke_decimation,
            smoke_step_timeout_s=args.smoke_step_timeout_s,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke_save_episode and not collect_timing.smoke_enabled:
        parser.error("--smoke_save_episode requires --smoke_action_steps > 0")
    if args.smoke_save_episode and args.no_frame_write:
        parser.error("--smoke_save_episode requires frame writing; remove --no_frame_write")

    output_dir = f"./data/{args.dataset_name}"
    dataset = _create_or_load_dataset(args.dataset_name, args.fps, output_dir)

    env_cfg, scene_key, info = _build_collect_env(
        args.object,
        args.asset_id,
        args.scene_key,
        args.num_envs,
        args.no_guide,
        args.pos_rand,
        collect_timing.decimation,
        collect_timing.render_interval,
        args.asset_dir,
    )
    event_key = _collect_event_key(scene_key, args.pos_rand)
    env: ManagerBasedContinuousEnv = ManagerBasedContinuousEnv(cfg=env_cfg, scene_key=scene_key)

    if args.object == "bottle":
        if info.get("close_mode"):
            env.scene["bottle"].data.default_joint_pos[:, 0] = 0.005
            env.scene["bottle"].data.default_joint_pos[:, 1] = env.scene["bottle"].data.joint_pos_limits[:, 1, 1]
        else:
            env.scene["bottle"].data.default_joint_pos[:, 0] = 0.0
            env.scene["bottle"].data.default_joint_pos[:, 1] = 0.0

    video_recorder = None
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        vpath = os.path.join(args.video_dir, f"{scene_key}_{args.asset_id.replace('/', '_')}.mp4")
        video_recorder = _MosaicVideoRecorder(vpath, args.fps, args.num_envs)

    curobo_mg = TrajectoryGenerator("franka", env=env)
    task_name = scene_key

    collected, loops = 0, 0
    print(f"[Collect] Target: {args.target_episodes} episodes  Max loops: {args.max_loops}")
    if collect_timing.smoke_enabled:
        print(
            f"[CollectSmoke] action_steps={collect_timing.smoke_action_steps} "
            f"decimation={collect_timing.decimation} render_interval={collect_timing.render_interval} "
            f"step_timeout_s={collect_timing.smoke_step_timeout_s:g} "
            f"frame_write={not args.no_frame_write} save_episode={args.smoke_save_episode}",
            flush=True,
        )
    if args.debug_collect or args.no_frame_write or (args.skill_timeout_s and args.skill_timeout_s > 0):
        print(
            f"[Collect] debug_collect={args.debug_collect} progress_interval={args.progress_interval} "
            f"skill_timeout_s={args.skill_timeout_s:g} no_frame_write={args.no_frame_write}",
            flush=True,
        )

    while collected < args.target_episodes and loops < args.max_loops:
        n = _collect_one_episode(
            env,
            args.object,
            args.asset_id,
            scene_key,
            event_key,
            args.pos_rand,
            curobo_mg,
            info,
            dataset,
            task_name,
            video_recorder,
            debug_collect=args.debug_collect,
            progress_interval=args.progress_interval,
            skill_timeout_s=args.skill_timeout_s,
            no_frame_write=args.no_frame_write,
            smoke_action_steps=collect_timing.smoke_action_steps,
            smoke_step_timeout_s=collect_timing.smoke_step_timeout_s,
            smoke_save_episode=args.smoke_save_episode,
        )
        collected += n
        loops += 1
        print(f"  [{loops}/{args.max_loops}] Collected: {collected}/{args.target_episodes}")

    rate = (collected / loops * 100) if loops else 0.0
    print(f"[Summary] Loops: {loops}  Successes: {collected}  Rate: {rate:.1f}%")

    # CSV log (supports concurrent writes from multiple workers)
    os.makedirs("./outputs/collect_logs", exist_ok=True)
    csv_path = "./outputs/collect_logs/collection_results.csv"
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp", "object", "asset_id", "scene_key", "loops", "successes", "rate"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    args.object, args.asset_id, scene_key, loops, collected, f"{rate:.1f}"])
        fcntl.flock(f, fcntl.LOCK_UN)

    if video_recorder is not None:
        video_recorder.close()

    if args.push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
