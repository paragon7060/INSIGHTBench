from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationData
from isaaclab.sensors import FrameTransformerData
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.assets import Articulation, RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

def root_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """
    The root states of the asset.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_state_w[:, :7]

def pos_guide(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    The relative position of the guide in the perspective of env origins.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    
    return asset.data.body_pos_w[:,0,:3] - env.scene.env_origins