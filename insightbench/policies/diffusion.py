"""Diffusion Policy wrapper."""

from __future__ import annotations

import torch
import torchvision.transforms as T
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.utils.constants import OBS_STATE

from insightbench.policies.base import PolicyBase
from insightbench.policies.pi0 import Pi0Wrapper  # reuse _trim_stats


class DiffusionWrapper(PolicyBase):
    """Wraps DiffusionPolicy for InsightBench evaluation.

    Applies a 224×224 resize to all camera inputs before inference,
    which differs from Pi0/SmolVLA that receive already-sized images.
    """

    def __init__(self, policy_cfg, device):
        super().__init__(policy_cfg, device)
        self._resize = T.Resize((224, 224), antialias=True)

    def load(self) -> None:
        cfg = self.cfg
        root = getattr(cfg, "dataset_stats_root", None) or None
        dataset_meta = LeRobotDatasetMetadata(cfg.dataset_stats_repo, root=root)
        Pi0Wrapper._trim_stats(dataset_meta.stats, cfg)
        self.policy = DiffusionPolicy.from_pretrained(
            cfg.checkpoint, dataset_stats=dataset_meta.stats
        )
        self.policy.to(self.device)
        self.policy.eval()

    def select_action(self, obs_state, obs_imgs, task_prompts):
        resized = {k: self._resize(v) for k, v in obs_imgs.items()}
        input_batch = {OBS_STATE: obs_state, "task": task_prompts, **resized}
        with torch.no_grad():
            self.policy.config.action_feature.shape = (self.cfg.action_dim,)
            action = self.policy.select_action(input_batch)
        return action
