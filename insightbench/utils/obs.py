"""Observation builders — policy-specific state and image dicts."""

from __future__ import annotations

import torch


# Supported policy types and their required obs_state construction
_OBS_STATE_BUILDERS = {
    "pi0":              lambda p: p["joint_pos"][:, :8],
    "diffusion":        lambda p: p["joint_pos"][:, :8],
    "instruction_gpt":  lambda p: p["joint_pos"][:, :8],
    "smolvla":          lambda p: torch.cat([p["ee_position"], p["ee_quat"], p["joint_pos"]], dim=1),
    "guide_vla":        lambda p: torch.cat([p["ee_position"], p["ee_quat"], p["joint_pos"]], dim=1),
    "groot": lambda p: torch.cat(
        [p["ee_position"], p["ee_quat"], p["joint_pos"], p["joint_vel"], p["actions"]], dim=1
    )[:, :-2],
}


def build_obs_state(obs_batch: dict, policy_type: str) -> torch.Tensor:
    """Return the obs state tensor required by the given policy type.

    Args:
        obs_batch:   Raw observation dict from the Isaac Lab env.
        policy_type: One of pi0 | diffusion | groot | smolvla | guide_vla | instruction_gpt.
    """
    builder = _OBS_STATE_BUILDERS.get(policy_type)
    if builder is None:
        raise ValueError(
            f"Unknown policy type '{policy_type}'. Choose from: {list(_OBS_STATE_BUILDERS)}"
        )
    return builder(obs_batch["policy"])


def build_obs_images(
    obs_batch: dict,
    infer_type: str,
    guide_cam: bool = True,
    resize_fn=None,
) -> dict[str, torch.Tensor]:
    """Return the image observation dict for the policy.

    Args:
        obs_batch:  Raw observation dict from the Isaac Lab env.
        infer_type: guide | instruction | sem | noguide | reverse.
        guide_cam:  When True, use the guide camera instead of left_shoulder.
        resize_fn:  Optional callable applied to each image tensor (e.g. torchvision Resize).
    """
    p = obs_batch["policy"]

    def to_float(key: str) -> torch.Tensor:
        t = p[key].permute(0, 3, 1, 2).to(torch.float32) / 255.0
        return resize_fn(t) if resize_fn else t

    if infer_type == "sem":
        return {
            "observation.images.wrist":          to_float("wrist"),
            "observation.images.wrist_semantic":  to_float("left_shoulder"),
            "observation.images.right_shoulder":  to_float("right_shoulder"),
        }

    if not guide_cam:
        return {
            "observation.images.wrist":          to_float("wrist"),
            "observation.images.right_shoulder": to_float("right_shoulder"),
            "observation.images.left_shoulder":  to_float("left_shoulder"),
        }

    return {
        "observation.images.wrist":          to_float("wrist"),
        "observation.images.right_shoulder": to_float("right_shoulder"),
        "observation.images.guide":          to_float("guide"),
    }
