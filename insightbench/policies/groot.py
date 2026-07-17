"""GR00T policy wrapper."""

from __future__ import annotations

from copy import deepcopy

import torch
from lerobot.configs.policies import PreTrainedConfig
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
        previous_cuda_device = None
        if self.device.type == "cuda" and self.device.index is not None:
            previous_cuda_device = torch.cuda.current_device()
            torch.cuda.set_device(self.device)
        try:
            policy_config = PreTrainedConfig.from_pretrained(ckpt)
            policy_config.device = str(self.device)
            self.policy = GrootPolicy.from_pretrained(ckpt, config=policy_config)
            self.policy.to(self.device)
        finally:
            if previous_cuda_device is not None:
                torch.cuda.set_device(previous_cuda_device)
        self.policy.eval()
        self._pre = PolicyProcessorPipeline.from_pretrained(
            ckpt,
            "policy_preprocessor.json",
            overrides={"device_processor": {"device": str(self.device)}},
        )
        self._post = PolicyProcessorPipeline.from_pretrained(ckpt, "policy_postprocessor.json")

    def select_action(self, obs_state, obs_imgs, task_prompts):
        # Stage cross-GPU observations through CPU so the checkpoint's device
        # processor moves them to the explicitly selected policy device.
        if obs_state.device != self.device:
            obs_state = obs_state.detach().cpu()
            obs_imgs = {key: value.detach().cpu() for key, value in obs_imgs.items()}
        input_batch = {"observation.state": obs_state, "task": task_prompts, **obs_imgs}
        batch = self._pre(deepcopy(input_batch))
        with torch.no_grad():
            self.policy.config.action_feature.shape = (self.cfg.action_dim,)
            action_raw = self.policy.select_action(batch)
            processed  = self._post({"action": action_raw})
        return processed["action"]
