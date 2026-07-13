from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
from pxr import UsdLux, Gf, Sdf, UsdGeom
import omni.usd

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaacsim.core.prims import XFormPrim 

def randomize_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]] | None = None,
    intensity_range: tuple[float, float] | None = None,
    color_rgb_range: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """
    Randomize Light's Pose, Intensity, and Color.
    
    Args:
        pose_range: dict with keys 'x', 'y', 'z', 'roll', 'pitch', 'yaw' and values (min, max).
        intensity_range: tuple of (min, max) for light intensity.
        color_rgb_range: tuple of ((min_r, min_g, min_b), (max_r, max_g, max_b)). 
                         Example: ((0.8, 0.8, 0.8), (1.0, 1.0, 1.0)) for bright whiteish variations.
    """
    # 1. 객체 및 USD Prim 가져오기
    light: XFormPrim = env.scene[asset_cfg.name]
    prim = light._prims[0]
    light_api = UsdLux.LightAPI(prim)
    
    device = env.device

    # --- 2. Intensity (밝기) ---
    if intensity_range:
        # 범위 내 랜덤 float 샘플링
        rand_intensity = float(np.random.uniform(intensity_range[0], intensity_range[1]))
        # USD 속성 적용
        light_api.GetIntensityAttr().Set(rand_intensity)

    # --- 3. Color (색상) ---
    if color_rgb_range:
        min_rgb, max_rgb = color_rgb_range
        # R, G, B 각각 랜덤 샘플링
        r = float(np.random.uniform(min_rgb[0], max_rgb[0]))
        g = float(np.random.uniform(min_rgb[1], max_rgb[1]))
        b = float(np.random.uniform(min_rgb[2], max_rgb[2]))
        
        # USD Color 속성은 Gf.Vec3f 타입 사용
        light_api.GetColorAttr().Set(Gf.Vec3f(r, g, b))