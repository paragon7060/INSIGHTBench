"""GR00T policy wrapper."""

from __future__ import annotations

from copy import deepcopy

import torch
from lerobot.policies.groot.modeling_groot import GrootPolicy
from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from insightbench.policies.base import PolicyBase


class GrootWrapper(PolicyBase):
    """Wraps GrootPolicy with its pre/post-processor pipeline.

    GR00T does not require dataset normalisation stats — the model
    handles normalisation internally via its processor pipeline.
    """

    def __init__(self, policy_cfg, device):
        super().__init__(policy_cfg, device)
        self._pre = None
        self._post = None

    def load(self) -> None:
        ckpt = self.cfg.checkpoint
        self.policy = GrootPolicy.from_pretrained(ckpt)
        self.policy.to(self.device)
        self.policy.eval()
        self._pre  = PolicyProcessorPipeline.from_pretrained(ckpt, "policy_preprocessor.json")
        self._post = PolicyProcessorPipeline.from_pretrained(ckpt, "policy_postprocessor.json")

    def select_action(self, obs_state, obs_imgs, task_prompts):
        input_batch = {"observation.state": obs_state, "task": task_prompts, **obs_imgs}
        batch = self._pre(deepcopy(input_batch))
        with torch.no_grad():
            self.policy.config.action_feature.shape = (self.cfg.action_dim,)
            action_raw = self.policy.select_action(batch)
            processed  = self._post({"action": action_raw})
        return processed["action"]
