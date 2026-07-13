"""GPT-Vision instruction generation + Pi0 policy wrapper."""

from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import PIL.Image
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.constants import OBS_STATE

from insightbench.policies.base import PolicyBase
from insightbench.policies.pi0 import Pi0Wrapper


_CLOSE_BOTTLE_SCENES = {"5d", "5e", "5f", "5h"}

_SYSTEM_PROMPT_OPEN = """
You are an expert AI assistant specialising in robotics. Look at the image and generate
a single imperative sentence instructing a robot to perform the visible manipulation task.
Rules:
  - cabinet → "Find the arrow guide and open the indicated drawer. The indicated drawer is ..."
  - door    → "Open the door, rotate (clockwise|counter-clockwise) and (push|pull)."
  - bottle (open, no squeeze) → "Open the bottle in (counter-clockwise|clockwise) direction."
  - bottle (open, squeeze)    → "Grip the cap on the sides indicated by the 'squeeze' arrow and open the bottle in clockwise direction. The arrows are at N-degrees counter-clockwise from horizontal axis."
Angles: 0° = horizontal left→right; 90° = vertical bottom→top; measured counter-clockwise.
"""

_SYSTEM_PROMPT_CLOSE = """
You are an expert AI assistant specialising in robotics. Look at the image and generate
a single imperative sentence instructing a robot to close the visible bottle.
Rules:
  - bottle (close) → "Close the bottle in (counter-clockwise|clockwise) direction."
"""

_USER_PROMPT = "Now, using this new image, generate an instruction.\n\n**Instruction:**"


class InstructionGPTWrapper(PolicyBase):
    """Generates language instructions from the guide-camera image via GPT-4o,
    then feeds them to Pi0 for action inference.

    The OpenAI API key must be set in the ``OPENAI_API_KEY`` environment variable
    (never hard-code it in source files).
    """

    def __init__(self, policy_cfg, device):
        super().__init__(policy_cfg, device)
        self._cached_prompts: list[str] | None = None
        self._gpt_model = getattr(policy_cfg, "gpt_model", "gpt-4o")
        self._debug_dir = getattr(policy_cfg, "debug_dir", "./outputs/gpt_debug")

    def load(self) -> None:
        cfg = self.cfg
        root = getattr(cfg, "dataset_stats_root", None) or None
        dataset_meta = LeRobotDatasetMetadata(cfg.dataset_stats_repo, root=root)
        Pi0Wrapper._trim_stats(dataset_meta.stats, cfg)
        self.policy = PI0Policy.from_pretrained(
            cfg.checkpoint, dataset_stats=dataset_meta.stats
        )
        self.policy.to(self.device)
        self.policy.eval()

    def generate_prompts(
        self, obs_batch: dict, obj_name: str, asset_path: str, scene_key: str
    ) -> list[str]:
        """Call GPT-4o once at episode start to get per-env language instructions."""
        prompts = []
        for env_id in range(obs_batch["policy"]["wrist"].shape[0]):
            if obj_name in ("door", "bottle"):
                img_np = obs_batch["policy"]["guide"][env_id].cpu().numpy().astype(np.uint8)
            else:
                img_np = obs_batch["policy"]["right_shoulder"][env_id].cpu().numpy().astype(np.uint8)
            prompts.append(
                self._call_gpt(img_np, obj_name, asset_path, scene_key, env_id)
            )
        self._cached_prompts = prompts
        return prompts

    def _call_gpt(
        self, image_np: np.ndarray, obj_name: str, asset_path: str, scene_key: str, env_id: int
    ) -> str:
        from openai import OpenAI

        os.makedirs(self._debug_dir, exist_ok=True)
        prefix = f"{obj_name}_{asset_path}_task{scene_key}_env{env_id}"
        PIL.Image.fromarray(image_np, "RGB").save(
            os.path.join(self._debug_dir, f"{prefix}_input.png")
        )

        is_close = scene_key in _CLOSE_BOTTLE_SCENES
        system = _SYSTEM_PROMPT_CLOSE if is_close else _SYSTEM_PROMPT_OPEN

        _, buf = cv2.imencode(".jpg", cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        b64 = base64.b64encode(buf).decode("utf-8")

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self._gpt_model,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _USER_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
            max_tokens=128,
        )
        prompt = resp.choices[0].message.content.strip()

        with open(os.path.join(self._debug_dir, f"{prefix}_output.txt"), "w") as f:
            f.write(prompt)
        print(f"[GPT] env{env_id}: {prompt}")
        return prompt

    def select_action(self, obs_state, obs_imgs, task_prompts):
        if self._cached_prompts is not None:
            task_prompts = self._cached_prompts
        input_batch = {OBS_STATE: obs_state, "task": task_prompts, **obs_imgs}
        with torch.no_grad():
            self.policy.config.action_feature.shape = (self.cfg.action_dim,)
            action = self.policy.select_action(input_batch)
        return action
