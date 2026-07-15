"""InsightBench unified evaluation entry point.

Usage:
    python scripts/evaluate.py \\
        --config configs/eval/pi0.yaml \\
        --object door --asset_path <ASSET_ID> --task_idx 0 \\
        --num_envs 8 --headless --enable_cameras

OmegaConf dot-notation overrides are supported after positional args:
    python scripts/evaluate.py --config configs/eval/pi0.yaml \\
        --object door --asset_path 14b --task_idx 1 \\
        policy.checkpoint=local/pretrained_model \\
        eval.save_video=false
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

from isaaclab.app import AppLauncher
from omegaconf import OmegaConf

from insightbench.utils.eval_config import load_eval_config, validate_required_eval_inputs


# ─── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="InsightBench evaluation.")
parser.add_argument("--config",      type=str, required=True, help="Path to eval YAML config.")
parser.add_argument("--object",      type=str, required=True, choices=["cabinet", "door", "bottle"])
parser.add_argument("--asset_path",  type=str, required=True, help="Asset identifier (e.g. '14b').")
parser.add_argument("--task_idx",    type=int, required=True, help="Task index within the asset.")
parser.add_argument("--num_envs",    type=int, default=None,  help="Override number of parallel envs.")
parser.add_argument("--no_guide",    action="store_true",     help="Disable guide asset in scene.")
parser.add_argument("--pos_rand",    action="store_true",     help="Enable object position randomization on each reset.")
parser.add_argument("--seed",        type=int, default=None)
# Remaining args are OmegaConf dot-notation overrides (e.g. policy.checkpoint=...)
parser.add_argument("overrides",     nargs="*", help="OmegaConf key=value overrides.")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
cfg_cli = load_eval_config(args_cli.config, args_cli.overrides)
validate_required_eval_inputs(cfg_cli, args_cli.config)
app_launcher  = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── Post-launch imports (Isaac Sim must be running) ──────────────────────────
import torch

from cfg.BaseTaskCfg import REWARD_FOR_SCENE_KEY, SCENE_TASK_PROMPT_GUIDE, SCENE_TASK_PROMPT_INSTRUCTION, SCENE_TASK_PROMPT_SEM, SCENE_TASK_PROMPT_INSTRUCTION_REVERSE, SCENE_TASK_PROMPT
from custom_lab.envs.manager_based_rl_step_env import ManagerBasedContinuousEnv

from insightbench.envs.builder import build_env
from insightbench.policies import load_policy
from insightbench.utils.obs import build_obs_state, build_obs_images
from insightbench.utils.video import VideoRecorder
from insightbench.utils.results import save_result


_PROMPT_MAPS = {
    "guide":       SCENE_TASK_PROMPT_GUIDE,
    "instruction": SCENE_TASK_PROMPT_INSTRUCTION,
    "sem":         SCENE_TASK_PROMPT_SEM,
    "noguide":     SCENE_TASK_PROMPT,
    "reverse":     SCENE_TASK_PROMPT_INSTRUCTION_REVERSE,
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
    obj_name: str = "",
) -> torch.Tensor:
    """Run one evaluation episode. Returns bool tensor (num_envs,) — True = success."""
    obs_batch, _ = env.reset()
    policy.reset()

    obs_batch = _noop_warmup(env, obs_batch, warmup_steps)

    success_tracker = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # InstructionGPT generates prompts from the guide image once per episode
    if hasattr(policy, "generate_prompts"):
        task_prompts = policy.generate_prompts(obs_batch, obj_name, asset_name, scene_key)

    if recorder is not None:
        first_frame = obs_batch["policy"]["wrist"][0]
        recorder.open(first_frame.shape, asset_name, task_name)

    for step in range(eval_steps):
        # Success check from reward manager
        step_rew = env.reward_manager._step_reward.squeeze(-1)
        reward_threshold = REWARD_FOR_SCENE_KEY[scene_key[0]]
        success_tracker |= step_rew > reward_threshold

        obs_state = build_obs_state(obs_batch, policy_type)
        obs_imgs  = build_obs_images(obs_batch, infer_type, guide_cam=guide_cam)

        if step % query_freq == 0:
            policy.reset()

        action = policy.select_action(obs_state, obs_imgs, task_prompts)
        expected_action_dim = env.action_manager.total_action_dim - 1
        if action.ndim != 2 or action.shape[1] != expected_action_dim:
            raise ValueError(
                f"Policy returned {tuple(action.shape)}; "
                f"expected [B, {expected_action_dim}] before gripper replication"
            )
        # The policy exposes one logical gripper command; replicate it for the two fingers.
        action = torch.cat([action, action[:, -1].unsqueeze(1)], dim=1)

        obs_batch, _, _, _, _ = env.step(action)

        if recorder is not None:
            recorder.write_frame(obs_batch)

    if recorder is not None:
        recorder.close_and_rename(success_tracker.tolist(), asset_name, task_name)

    return success_tracker


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    args = args_cli
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load and merge config ──────────────────────────────────────────────────
    cfg = cfg_cli

    policy_cfg = cfg.policy
    eval_cfg   = cfg.eval
    num_envs   = args.num_envs if args.num_envs is not None else eval_cfg.get("num_envs", 8)
    seed       = args.seed if args.seed is not None else 42
    num_episodes = eval_cfg.get("num_episodes", 1)

    _log(f"")
    _log(f"{'='*60}")
    _log(f"  Task   : {args.object} / {args.asset_path} / task{args.task_idx}")
    _log(f"  Policy : {policy_cfg.type}  ({getattr(policy_cfg, 'checkpoint_subfolder', policy_cfg.checkpoint)})")
    _log(f"  Runs   : {num_episodes} episode(s) x {num_envs} envs = {num_episodes * num_envs} attempts")
    _log(f"{'='*60}")

    # ── Build environment ──────────────────────────────────────────────────────
    _log(f"[1/3] Building environment...")
    env_cfg, scene_key, info = build_env(
        obj_name=args.object,
        task_idx=args.task_idx,
        asset_path=args.asset_path,
        no_guide=args.no_guide,
        num_envs=num_envs,
        seed=seed,
        pos_rand=args.pos_rand,
    )

    env: ManagerBasedContinuousEnv = ManagerBasedContinuousEnv(cfg=env_cfg, scene_key=scene_key)

    # Set gripper defaults
    env.scene["robot"].data.default_joint_pos[:, 7] = 0.04
    env.scene["robot"].data.default_joint_vel[:, 8] = 0.04
    if args.object == "bottle":
        if info.get("close_mode"):
            env.scene["bottle"].data.default_joint_pos[:, 0] = 0.005
            env.scene["bottle"].data.default_joint_pos[:, 1] = env.scene["bottle"].data.joint_pos_limits[:, 1, 1]
        else:
            env.scene["bottle"].data.default_joint_pos[:, 0] = 0.0
            env.scene["bottle"].data.default_joint_pos[:, 1] = 0.0

    _log(f"[1/3] Env ready  (scene_key={scene_key})")

    # ── Load policy ────────────────────────────────────────────────────────────
    _log(f"[2/3] Loading policy...")
    policy = load_policy(policy_cfg, device)
    _log(f"[2/3] Policy loaded  ({policy_cfg.type})")

    # ── Task prompts ───────────────────────────────────────────────────────────
    infer_type, guide_cam = _resolve_policy_inference_options(policy_cfg)
    task_prompt = _get_task_prompt(infer_type, scene_key)
    task_prompts = [task_prompt] * num_envs

    # ── Video recorder (optional) ──────────────────────────────────────────────
    recorder = None
    if eval_cfg.get("save_video", False):
        recorder = VideoRecorder(
            video_dir=eval_cfg.video_dir,
            views=["wrist", "right_shoulder"],
            num_envs=num_envs,
            fps=eval_cfg.get("fps", 10),
        )

    # ── Evaluation loop ────────────────────────────────────────────────────────
    _log(f"")
    _log(f"--- Running {num_episodes} episode(s) ---")
    total_success = 0

    for ep in range(num_episodes):
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
            asset_name=args.asset_path,
            task_name=scene_key,
            obj_name=args.object,
        )
        ep_successes = int(success_mask.sum().item())
        total_success += ep_successes
        _log(f"  ep{ep}: {ep_successes}/{num_envs} success")

    total_attempts = num_episodes * num_envs
    rate = total_success / total_attempts * 100
    _log(f"")
    _log(f"{'='*60}")
    _log(f"  RESULT : {args.object}/{args.asset_path}/task{args.task_idx}")
    _log(f"  Score  : {total_success}/{total_attempts}  ({rate:.1f}%)")
    _log(f"{'='*60}")

    save_result(
        results_dir=eval_cfg.results_dir,
        obj_name=args.object,
        asset_path=args.asset_path,
        task_idx=args.task_idx,
        scene_key=scene_key,
        successes=total_success,
        attempts=total_attempts,
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
