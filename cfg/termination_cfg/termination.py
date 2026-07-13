from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm


def skill_finish(env: ManagerBasedRLEnv, command_name: str, skill_length: int) -> torch.Tensor:
    """Terminate the episode based on the total number of times commands have been re-sampled.

    This makes the maximum episode length fluid in nature as it depends on how the commands are
    sampled. It is useful in situations where delayed rewards are used :cite:`rudin2022advanced`.
    """
    command: CommandTerm = env.command_manager.get_term(command_name)
    # TODO : <= or < check!
    return torch.logical_or((command.time_left <= env.step_dt), (command.skill_step >= skill_length))

def mg_failed(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.full((1,), env.action_manager._terms['arm_action'].mg_failed, device=env.device, dtype=torch.bool)

