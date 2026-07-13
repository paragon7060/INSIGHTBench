"""SmolVLA policy wrapper."""

from __future__ import annotations

from copy import deepcopy

import torch
from packaging.version import Version

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
from lerobot.utils.constants import OBS_STATE

from insightbench.policies.base import PolicyBase
from insightbench.policies.pi0 import Pi0Wrapper  # reuse _trim_stats


def _require_smolvla_processor_pipeline() -> None:
    """Fail fast outside the documented LeRobot 0.4.x SmolVLA API."""
    version = Version(lerobot.__version__)
    if not Version("0.4.0") <= version < Version("0.5.0"):
        raise RuntimeError(
            f"SmolVLA evaluation requires the pinned LeRobot 0.4.x processor API, not {version}."
        )


class SmolVLAWrapper(PolicyBase):
    """Wraps SmolVLAPolicy.

    SmolVLA uses a 16-dim state (ee_pos + ee_quat + joint_pos) with
    state_start_idx=0, unlike Pi0 which uses joint_pos[:8] starting at idx 7.
    """

    def __init__(self, policy_cfg, device: torch.device):
        super().__init__(policy_cfg, device)
        self._preprocessor = None
        self._postprocessor = None

    def load(self) -> None:
        _require_smolvla_processor_pipeline()
        cfg = self.cfg
        root = getattr(cfg, "dataset_stats_root", None) or None
        dataset_meta = LeRobotDatasetMetadata(cfg.dataset_stats_repo, root=root)
        Pi0Wrapper._trim_stats(dataset_meta.stats, cfg)
        # LeRobot 0.4.x stores normalization outside SmolVLAPolicy. Keep the
        # checkpoint loader free of stats and attach them to official processors.
        self.policy = SmolVLAPolicy.from_pretrained(cfg.checkpoint)
        self._preprocessor, self._postprocessor = make_smolvla_pre_post_processors(
            config=self.policy.config,
            dataset_stats=dataset_meta.stats,
        )
        self.policy.to(self.device)
        self.policy.eval()

    def select_action(self, obs_state, obs_imgs, task_prompts):
        input_batch = {OBS_STATE: obs_state, "task": task_prompts, **obs_imgs}
        if self._preprocessor is None or self._postprocessor is None:
            raise RuntimeError("SmolVLA processor pipeline was not initialized during policy loading.")
        batch = self._preprocessor(deepcopy(input_batch))
        with torch.no_grad():
            action = self.policy.select_action(batch)
        return self._postprocessor(action)
