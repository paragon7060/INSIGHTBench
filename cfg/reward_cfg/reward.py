from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat
from isaaclab.assets import Articulation
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def joint_pos_success(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, success_js_value: float) -> torch.Tensor:
    """Bonus for joint ids > success_js_value.
    """
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids[0]]
    open_easy = (joint_pos > success_js_value/2) * 0.4
    open_medium = (joint_pos > success_js_value) * 0.6

    return open_easy + open_medium
    
def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), offset: float = 0.0, reverse: bool = False) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    reverse_weight = 1
    if reverse:
        reverse_weight = -1
    return reverse_weight * asset.data.joint_pos[:, asset_cfg.joint_ids].flatten() + offset

def joint_pos_for_joint_id(env: ManagerBasedRLEnv, joint_id: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("cabinet")) -> torch.Tensor:
    """The joint positions of the asset.
    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    joint_id_idx = joint_names.index(joint_id)

    return asset.data.joint_pos[:,joint_id_idx].flatten()