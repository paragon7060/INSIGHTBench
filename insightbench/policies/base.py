"""Abstract base class for all policy wrappers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class PolicyBase(ABC):
    """Wraps a LeRobot-style policy with a unified interface.

    Subclasses handle loading, obs formatting, and action selection
    for each specific policy type.
    """

    def __init__(self, policy_cfg, device: torch.device):
        self.cfg = policy_cfg
        self.device = device
        self.policy = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights from cfg.checkpoint."""
        ...

    @abstractmethod
    def select_action(
        self,
        obs_state: torch.Tensor,
        obs_imgs: dict[str, torch.Tensor],
        task_prompts: list[str],
    ) -> torch.Tensor:
        """Return action tensor of shape (B, action_dim)."""
        ...

    def reset(self) -> None:
        """Reset internal policy state (e.g. action chunking buffer)."""
        if self.policy is not None:
            self.policy.reset()
