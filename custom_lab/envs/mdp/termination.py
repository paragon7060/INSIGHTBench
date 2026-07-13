from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

def success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the task is successfully completed."""
    env.reward_buf
    return env.task_success_buf