"""SmolVLA policy wrapper."""

from __future__ import annotations

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.constants import OBS_STATE

from insightbench.policies.base import PolicyBase
from insightbench.policies.pi0 import Pi0Wrapper  # reuse _trim_stats


class SmolVLAWrapper(PolicyBase):
    """Wraps SmolVLAPolicy.

    SmolVLA uses a 16-dim state (ee_pos + ee_quat + joint_pos) with
    state_start_idx=0, unlike Pi0 which uses joint_pos[:8] starting at idx 7.
    """

    def load(self) -> None:
        cfg = self.cfg
        root = getattr(cfg, "dataset_stats_root", None) or None
        dataset_meta = LeRobotDatasetMetadata(cfg.dataset_stats_repo, root=root)
        Pi0Wrapper._trim_stats(dataset_meta.stats, cfg)
        self.policy = SmolVLAPolicy.from_pretrained(
            cfg.checkpoint, dataset_stats=dataset_meta.stats
        )
        self.policy.to(self.device)
        self.policy.eval()

    def select_action(self, obs_state, obs_imgs, task_prompts):
        input_batch = {OBS_STATE: obs_state, "task": task_prompts, **obs_imgs}
        with torch.no_grad():
            self.policy.config.action_feature.shape = (self.cfg.action_dim,)
            action = self.policy.select_action(input_batch)
        return action
