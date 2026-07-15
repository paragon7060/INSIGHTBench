"""Python 3.11 client wrapper for the isolated LeRobot 0.6 GR00T server."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
import torch

from insightbench.policies.base import PolicyBase


_PANDA_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
_PANDA_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)


def _encode_array(value: np.ndarray) -> dict:
    buffer = BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return {
        "format": "npy_base64",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _decode_array(value: dict) -> np.ndarray:
    if value.get("format") != "npy_base64":
        return np.asarray(value, dtype=np.float32)
    return np.load(BytesIO(base64.b64decode(value["data"])), allow_pickle=False)


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600.0) as response:  # noqa: S310 - configured localhost bridge
        return json.loads(response.read().decode("utf-8"))


class Groot060ClientWrapper(PolicyBase):
    """Map INSIGHTBench observations to the local Python 3.12 GR00T server."""

    def __init__(self, policy_cfg, device: torch.device):
        super().__init__(policy_cfg, device)
        self.endpoint = str(policy_cfg.endpoint).rstrip("/")
        self.reset_endpoint = str(policy_cfg.reset_endpoint)
        self.health_endpoint = str(policy_cfg.health_endpoint)
        self.external_image_key = str(policy_cfg.get("external_image_key", "observation.images.right_shoulder"))
        self.wrist_image_key = str(policy_cfg.get("wrist_image_key", "observation.images.wrist"))

    def load(self) -> None:
        health = _request_json("GET", self.health_endpoint)
        if not health.get("ok"):
            raise RuntimeError(f"GR00T 0.6 server health check failed: {health}")
        if int(health["state_dim"]) != int(self.cfg.state_dim):
            raise ValueError(f"Server state_dim={health['state_dim']} but config has {self.cfg.state_dim}")
        if int(health["action_dim"]) != int(self.cfg.action_dim):
            raise ValueError(f"Server action_dim={health['action_dim']} but config has {self.cfg.action_dim}")
        self.policy = True

    def reset(self) -> None:
        _request_json("POST", self.reset_endpoint, {})

    def select_action(self, obs_state, obs_imgs, task_prompts):
        if obs_state.ndim != 2 or obs_state.shape[1] != int(self.cfg.state_dim):
            raise ValueError(f"Expected state [B, {self.cfg.state_dim}], got {tuple(obs_state.shape)}")
        if self.external_image_key not in obs_imgs or self.wrist_image_key not in obs_imgs:
            raise KeyError(f"Need {self.external_image_key} and {self.wrist_image_key}; got {list(obs_imgs)}")

        def to_uint8(tensor: torch.Tensor) -> np.ndarray:
            image = tensor.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
            return np.ascontiguousarray(image)

        payload = {
            "state": _encode_array(obs_state.detach().to(torch.float32).cpu().numpy()),
            "images": {
                "exterior_image_1_left": _encode_array(to_uint8(obs_imgs[self.external_image_key])),
                "wrist_image_left": _encode_array(to_uint8(obs_imgs[self.wrist_image_key])),
            },
            "task": list(task_prompts),
        }
        response = _request_json("POST", self.endpoint, payload)
        if not response.get("ok"):
            raise RuntimeError(f"GR00T 0.6 inference failed: {response}")
        action = _decode_array(response["action"]).astype(np.float32, copy=False)
        if action.ndim != 2 or action.shape[1] != int(self.cfg.action_dim):
            raise ValueError(f"Expected action [B, {self.cfg.action_dim}], got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("GR00T 0.6 returned non-finite action values")

        # oxe_droid_relative_eef_relative_joint: eef_9d (0:9), gripper (9), joints (10:17).
        joints = np.clip(action[:, 10:17], _PANDA_LOWER, _PANDA_UPPER)
        gripper = np.clip(action[:, 9:10], 0.0, 1.0) * 0.04
        env_action = np.concatenate([joints, gripper], axis=1).astype(np.float32, copy=False)
        return torch.from_numpy(env_action).to(self.device)
