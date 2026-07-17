#!/usr/bin/env python3
"""Persistent InsightBench evaluator that loads one policy per worker.

This is an opt-in alternative to ``scripts/evaluate.py``.  The original
single-job evaluator remains unchanged.  A persistent worker reads TSV jobs,
starts Isaac Sim once, loads the policy once after the first environment is
ready, and then creates/closes one environment per job.

Job file format::

    object<TAB>asset<TAB>task_idx

Use ``scripts/eval_batch_persistent.sh`` for normal multi-GPU operation.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import os
from pathlib import Path
import sys
import time
import traceback


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

from isaaclab.app import AppLauncher
from omegaconf import OmegaConf

from insightbench.utils.eval_artifacts import resolve_asset_split
from insightbench.utils.eval_config import load_eval_config, validate_required_eval_inputs


parser = argparse.ArgumentParser(
    description="InsightBench persistent evaluation worker (one policy load per process)."
)
parser.add_argument("--config", required=True, help="Path to eval YAML config.")
parser.add_argument(
    "--job-file",
    required=True,
    type=Path,
    help="TSV file containing object, asset, and task_idx columns.",
)
parser.add_argument(
    "--append-status",
    action="store_true",
    help="Preserve an existing status file when resuming a structured run.",
)
parser.add_argument("--log-dir", required=True, type=Path, help="Per-job log directory.")
parser.add_argument(
    "--status-file",
    required=True,
    type=Path,
    help="TSV status output used by the batch launcher.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Override parallel env count.")
parser.add_argument("--no_guide", action="store_true", help="Disable guide assets.")
parser.add_argument("--pos_rand", action="store_true", help="Enable reset position randomization.")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("overrides", nargs="*", help="OmegaConf key=value overrides.")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
cfg_cli = load_eval_config(args_cli.config, args_cli.overrides)
validate_required_eval_inputs(cfg_cli, args_cli.config)


def _read_jobs(path: Path) -> list[tuple[str, str, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Persistent eval job file not found: {path}")

    jobs: list[tuple[str, str, int]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 3:
            raise ValueError(
                f"{path}:{line_number}: expected object<TAB>asset<TAB>task_idx, got {raw_line!r}"
            )
        object_name, asset_name, task_token = fields
        if object_name not in {"cabinet", "door", "bottle"}:
            raise ValueError(f"{path}:{line_number}: unknown object {object_name!r}")
        task_idx = int(task_token)
        if task_idx < 0:
            raise ValueError(f"{path}:{line_number}: task_idx must be non-negative")
        jobs.append((object_name, asset_name, task_idx))
    return jobs


jobs_cli = _read_jobs(args_cli.job_file)
args_cli.log_dir.mkdir(parents=True, exist_ok=True)
args_cli.status_file.parent.mkdir(parents=True, exist_ok=True)
if args_cli.append_status:
    args_cli.status_file.touch()
else:
    args_cli.status_file.write_text("", encoding="utf-8")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Post-launch imports: Isaac Sim must be running first.
import torch
import isaacsim.core.utils.stage as stage_utils

from cfg.BaseTaskCfg import (
    REWARD_FOR_SCENE_KEY,
    SCENE_TASK_PROMPT,
    SCENE_TASK_PROMPT_GUIDE,
    SCENE_TASK_PROMPT_INSTRUCTION,
    SCENE_TASK_PROMPT_INSTRUCTION_REVERSE,
    SCENE_TASK_PROMPT_SEM,
)
from custom_lab.envs.manager_based_rl_step_env import ManagerBasedContinuousEnv
from isaaclab.sim import SimulationContext

from insightbench.envs.builder import (
    build_env,
    configure_eval_camera_pipeline,
    configure_eval_default_joint_state,
)
from insightbench.policies import load_policy
from insightbench.utils.obs import build_obs_images, build_obs_state
from insightbench.utils.results import save_result
from insightbench.utils.video import VideoRecorder


class PolicyLoadError(RuntimeError):
    """Stop a worker when its one-time policy initialization fails."""


class EnvironmentCleanupError(RuntimeError):
    """Stop a worker when the next job cannot safely reuse the Isaac process."""


_PROMPT_MAPS = {
    "guide": SCENE_TASK_PROMPT_GUIDE,
    "instruction": SCENE_TASK_PROMPT_INSTRUCTION,
    "sem": SCENE_TASK_PROMPT_SEM,
    "noguide": SCENE_TASK_PROMPT,
    "reverse": SCENE_TASK_PROMPT_INSTRUCTION_REVERSE,
}


def _get_task_prompt(infer_type: str, scene_key: str) -> str:
    prompt_map = _PROMPT_MAPS.get(infer_type)
    if prompt_map is None:
        raise ValueError(f"Unknown infer_type '{infer_type}'. Choose from: {list(_PROMPT_MAPS)}")
    return prompt_map[scene_key]


def _resolve_policy_inference_options(policy_cfg) -> tuple[str, bool]:
    """Read optional prompt/camera settings from an OmegaConf policy section."""
    infer_type = OmegaConf.select(policy_cfg, "infer_type", default="guide")
    guide_cam = OmegaConf.select(policy_cfg, "guide_cam", default=True)
    return infer_type, guide_cam


def _noop_warmup(env: ManagerBasedContinuousEnv, obs_batch: dict, warmup_steps: int):
    """Execute warmup hold steps to settle the simulation."""
    noop_batch = env.scene["robot"].data.joint_pos[:, :9].clone()
    for _ in range(warmup_steps):
        obs_batch, _, _, _, _ = env.step(noop_batch)
    return obs_batch


def run_episode(
    env: ManagerBasedContinuousEnv,
    policy,
    scene_key: str,
    task_prompts: list[str],
    policy_type: str,
    infer_type: str,
    guide_cam: bool,
    query_freq: int,
    eval_steps: int,
    warmup_steps: int,
    recorder: VideoRecorder | None,
    asset_name: str,
    task_name: str,
    task_idx: int,
    split: str,
    obj_name: str = "",
) -> torch.Tensor:
    """Run one evaluation episode. Returns bool tensor (num_envs,) — True = success."""
    obs_batch, _ = env.reset()
    if hasattr(policy, "reset_episode"):
        policy.reset_episode()
    else:
        policy.reset()

    obs_batch = _noop_warmup(env, obs_batch, warmup_steps)

    success_tracker = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if hasattr(policy, "generate_prompts"):
        task_prompts = policy.generate_prompts(obs_batch, obj_name, asset_name, scene_key)

    if recorder is not None:
        first_frame = obs_batch["policy"]["wrist"][0]
        recorder.open(
            first_frame.shape,
            object_name=obj_name,
            asset_name=asset_name,
            scene_key=task_name,
            split=split,
            task_idx=task_idx,
        )

    perf_totals = {
        "obs": 0.0,
        "reset": 0.0,
        "policy": 0.0,
        "env": 0.0,
        "video": 0.0,
        "total": 0.0,
    }
    reward_threshold = REWARD_FOR_SCENE_KEY[scene_key[0]]
    for step in range(eval_steps):
        step_started = time.perf_counter()

        phase_started = time.perf_counter()
        obs_state = build_obs_state(obs_batch, policy_type)
        obs_imgs = build_obs_images(obs_batch, infer_type, guide_cam=guide_cam)
        perf_totals["obs"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        if step % query_freq == 0:
            policy.reset()
        perf_totals["reset"] += time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        action = policy.select_action(obs_state, obs_imgs, task_prompts)
        perf_totals["policy"] += time.perf_counter() - phase_started
        expected_action_dim = env.action_manager.total_action_dim - 1
        if action.ndim != 2 or action.shape[1] != expected_action_dim:
            raise ValueError(
                f"Policy returned {tuple(action.shape)}; "
                f"expected [B, {expected_action_dim}] before gripper replication"
            )
        action = torch.cat([action, action[:, -1].unsqueeze(1)], dim=1)

        phase_started = time.perf_counter()
        obs_batch, _, _, _, _ = env.step(action)
        perf_totals["env"] += time.perf_counter() - phase_started

        step_rew = env.reward_manager._step_reward.squeeze(-1)
        success_tracker |= step_rew > reward_threshold

        phase_started = time.perf_counter()
        if recorder is not None:
            recorder.write_frame(obs_batch)
        perf_totals["video"] += time.perf_counter() - phase_started
        perf_totals["total"] += time.perf_counter() - step_started

        completed_steps = step + 1
        if completed_steps % 10 == 0 or completed_steps == eval_steps:
            averages = {
                name: total * 1000.0 / completed_steps for name, total in perf_totals.items()
            }
            _log(
                "[PERF] "
                f"step={completed_steps}/{eval_steps} "
                f"total_ms={averages['total']:.1f} "
                f"obs_ms={averages['obs']:.1f} "
                f"reset_ms={averages['reset']:.1f} "
                f"policy_ms={averages['policy']:.1f} "
                f"env_ms={averages['env']:.1f} "
                f"video_ms={averages['video']:.1f}"
            )

    if recorder is not None:
        recorder.close_and_rename(success_tracker.tolist())

    return success_tracker


def _log(msg: str) -> None:
    print(msg, flush=True)


def _release_open_video_writers(recorder: VideoRecorder | None) -> None:
    """Release temporary writers if a job fails before normal video finalization."""
    if recorder is None:
        return
    for writers in recorder._writers.values():
        for writer in writers:
            writer.release()
    recorder._writers.clear()
    recorder._temp_paths.clear()
    recorder._job_dir = None


def _close_environment(env: ManagerBasedContinuousEnv | None) -> None:
    """Fully tear down one job's simulation while keeping Kit and the policy alive."""
    cleanup_error: Exception | None = None
    simulation_context = env.sim if env is not None else SimulationContext.instance()

    if env is not None:
        try:
            env.close()
        except Exception as exc:  # Preserve cleanup progress before surfacing the error.
            cleanup_error = exc

    if simulation_context is not None:
        for operation in (simulation_context.stop, simulation_context.clear):
            try:
                operation()
            except Exception as exc:
                if cleanup_error is None:
                    cleanup_error = exc
        SimulationContext.clear_instance()

    # SimulationCfg.create_stage_in_memory is false in this benchmark.  Closing
    # ManagerBasedEnv clears the SimulationContext singleton but leaves its USD
    # stage attached, so replace it before constructing the next environment.
    try:
        stage_utils.create_new_stage()
        simulation_app.update()
    except Exception as exc:
        if cleanup_error is None:
            cleanup_error = exc

    if cleanup_error is not None:
        raise EnvironmentCleanupError(
            "Persistent evaluator could not fully reset the Isaac environment"
        ) from cleanup_error


def _append_status(object_name: str, asset_name: str, task_idx: int, status: str) -> None:
    with args_cli.status_file.open("a", encoding="utf-8") as stream:
        stream.write(f"{object_name}\t{asset_name}\t{task_idx}\t{status}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _job_log_path(object_name: str, asset_name: str, task_idx: int) -> Path:
    return args_cli.log_dir / f"{object_name}_{asset_name}_task{task_idx}.log"


def _run_job(
    *,
    object_name: str,
    asset_name: str,
    task_idx: int,
    policy_ref: list[object | None],
    device: torch.device,
) -> None:
    cfg = cfg_cli
    policy_cfg = cfg.policy
    eval_cfg = cfg.eval
    policy_device = torch.device(OmegaConf.select(policy_cfg, "device", default=str(device)))
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else eval_cfg.get("num_envs", 8)
    seed = args_cli.seed if args_cli.seed is not None else 42
    num_episodes = eval_cfg.get("num_episodes", 1)
    infer_type, guide_cam = _resolve_policy_inference_options(policy_cfg)

    env = None
    recorder = None
    try:
        policy = policy_ref[0]
        _log("")
        _log("=" * 60)
        _log(f"  Task   : {object_name} / {asset_name} / task{task_idx}")
        _log(
            f"  Policy : {policy_cfg.type}  "
            f"({getattr(policy_cfg, 'checkpoint_subfolder', policy_cfg.checkpoint)})"
        )
        _log(
            f"  Runs   : {num_episodes} episode(s) x {num_envs} envs = "
            f"{num_episodes * num_envs} attempts"
        )
        _log("=" * 60)

        _log("[1/3] Building environment...")
        env_cfg, scene_key, info = build_env(
            obj_name=object_name,
            task_idx=task_idx,
            asset_path=asset_name,
            no_guide=args_cli.no_guide,
            num_envs=num_envs,
            seed=seed,
            pos_rand=args_cli.pos_rand,
        )
        if eval_cfg.get("optimize_camera_pipeline", False):
            required_views = {"wrist", "right_shoulder"}
            if infer_type == "sem" or not guide_cam:
                required_views.add("left_shoulder")
            else:
                required_views.add("guide")
            if eval_cfg.get("save_video", False):
                required_views.update(eval_cfg.get("video_views", []))
            configure_eval_camera_pipeline(env_cfg, required_views)

        split = resolve_asset_split(object_name, asset_name)
        _log(f"[1/3] Artifact routing: scene={scene_key} split={split}")

        env = ManagerBasedContinuousEnv(cfg=env_cfg, scene_key=scene_key)
        env.eval_single_render = bool(eval_cfg.get("optimize_camera_pipeline", False))
        configure_eval_default_joint_state(env, object_name, info)
        _log(f"[1/3] Env ready  (scene_key={scene_key})")

        if policy is None:
            _log("[2/3] Loading policy once for this persistent worker...")
            try:
                policy = load_policy(policy_cfg, policy_device)
            except Exception as exc:
                raise PolicyLoadError("Persistent worker policy initialization failed") from exc
            policy_ref[0] = policy
            _log(f"[2/3] Policy loaded  ({policy_cfg.type})")
        else:
            _log(f"[2/3] Reusing loaded policy  ({policy_cfg.type})")

        task_prompt = _get_task_prompt(infer_type, scene_key)
        task_prompts = [task_prompt] * num_envs

        if eval_cfg.get("save_video", False):
            recorder = VideoRecorder(
                video_dir=eval_cfg.video_dir,
                views=list(eval_cfg.get("video_views", ["wrist", "right_shoulder"])),
                num_envs=num_envs,
                fps=eval_cfg.get("fps", 10),
            )

        _log("")
        _log(f"--- Running {num_episodes} episode(s) ---")
        total_success = 0
        for episode in range(num_episodes):
            success_mask = run_episode(
                env=env,
                policy=policy,
                scene_key=scene_key,
                task_prompts=task_prompts,
                policy_type=str(policy_cfg.type),
                infer_type=infer_type,
                guide_cam=guide_cam,
                query_freq=eval_cfg.get("query_freq", 10),
                eval_steps=eval_cfg.get("eval_steps", 100),
                warmup_steps=eval_cfg.get("warmup_steps", 10),
                recorder=recorder,
                asset_name=asset_name,
                task_name=scene_key,
                task_idx=task_idx,
                split=split,
                obj_name=object_name,
            )
            episode_successes = int(success_mask.sum().item())
            total_success += episode_successes
            _log(f"  ep{episode}: {episode_successes}/{num_envs} success")

        total_attempts = num_episodes * num_envs
        rate = total_success / total_attempts * 100
        _log("")
        _log("=" * 60)
        _log(f"  RESULT : {object_name}/{asset_name}/task{task_idx}")
        _log(f"  Score  : {total_success}/{total_attempts}  ({rate:.1f}%)")
        _log("=" * 60)

        save_result(
            results_dir=eval_cfg.results_dir,
            obj_name=object_name,
            asset_path=asset_name,
            task_idx=task_idx,
            scene_key=scene_key,
            successes=total_success,
            attempts=total_attempts,
            split=split,
        )
    finally:
        _release_open_video_writers(recorder)
        _close_environment(env)
        del env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_ref: list[object | None] = [None]
    failed_jobs = 0
    fatal_worker_error = False

    print(
        f"Persistent worker starting: jobs={len(jobs_cli)} device={device} "
        f"policy={cfg_cli.policy.type}",
        flush=True,
    )

    for job_index, (object_name, asset_name, task_idx) in enumerate(jobs_cli, 1):
        log_path = _job_log_path(object_name, asset_name, task_idx)
        print(
            f"[{job_index}/{len(jobs_cli)}] {object_name}/{asset_name}/task{task_idx} "
            f"-> {log_path}",
            flush=True,
        )
        status = "failed"
        worker_must_stop = False
        with log_path.open("w", encoding="utf-8", buffering=1) as log_stream:
            with redirect_stdout(log_stream), redirect_stderr(log_stream):
                try:
                    _run_job(
                        object_name=object_name,
                        asset_name=asset_name,
                        task_idx=task_idx,
                        policy_ref=policy_ref,
                        device=device,
                    )
                except PolicyLoadError:
                    worker_must_stop = True
                    traceback.print_exc()
                except EnvironmentCleanupError:
                    worker_must_stop = True
                    traceback.print_exc()
                except Exception:
                    traceback.print_exc()
                else:
                    status = "ok"

        _append_status(object_name, asset_name, task_idx, status)
        if status == "ok":
            print(f"[{job_index}/{len(jobs_cli)}] OK", flush=True)
        else:
            failed_jobs += 1
            print(f"[{job_index}/{len(jobs_cli)}] FAILED — {log_path}", flush=True)
        if worker_must_stop:
            fatal_worker_error = True
            print(
                "Persistent worker cannot safely continue; remaining jobs will be retried.",
                flush=True,
            )
            break

    if fatal_worker_error:
        return 2
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
