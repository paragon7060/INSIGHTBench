
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
from isaaclab.sensors.camera import Camera
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
import omni.physx.scripts.utils as physxUtils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLStepEnv

def reset_guide_position_side_change(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cabinet_asset_name: str = "cabinet",
):
    """
    Reset and randomize the position of all guide assets in the scene.
    This function uses the current cabinet position as reference to avoid collisions.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to reset. If None, reset all environments
        pose_range: Range for position and orientation randomization
        cabinet_asset_name: Name of the cabinet asset to use as reference
    """
    # Set env_ids if None
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.scene.device)
    
    # Get current cabinet position as reference
    try:
        cabinet = env.scene[cabinet_asset_name]
        # Get current cabinet position (not default)
        cabinet_current_pos = cabinet.data.root_pos_w[env_ids].clone()
        cabinet_default_pos = cabinet.data.default_root_state[env_ids, 0:3].clone()
    except KeyError:
        print(f"[WARNING] Cabinet asset '{cabinet_asset_name}' not found, using default positions")
        cabinet_current_pos = None
        cabinet_default_pos = None
    
    # Find all guide assets in the scene
    guide_assets = []
    
    # Check rigid objects
    for asset_name, asset in env.scene.rigid_objects.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    
    # Check articulations (in case guide assets are articulations)
    for asset_name, asset in env.scene.articulations.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    
    # Check extras (in case guide assets are in extras)
    for asset_name, asset in env.scene.extras.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    
    if not guide_assets:
        return

    # Generate random position offset for all guide assets (same for each env)
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=env.scene.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], 
        (len(env_ids), 6), device=env.scene.device
    )
        
    # Process each guide asset
    for asset_name, asset in guide_assets:
        # Get default root state
        root_states = asset.data.default_root_state[env_ids].clone()
        
        # Calculate base position: use cabinet current position if available, otherwise use default
        if cabinet_current_pos is not None and cabinet_default_pos is not None:
            # Calculate relative position from cabinet's current position
            # This assumes guide assets are positioned relative to cabinet
            cabinet_noise = cabinet_current_pos - env.scene.env_origins[env_ids] - cabinet_default_pos
        
        # Apply position offset on top of the base position
        positions = root_states[:, 0:3] + cabinet_noise + pose_samples[:, 0:3] + env.scene.env_origins[env_ids]
        orientations = root_states[:, 3:7]



        # Set into the physics simulation
        asset.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), 
            env_ids=env_ids
        )

def make_fixed_joints(
    env,
    env_ids: torch.Tensor | None,
    src_cfg: SceneEntityCfg = SceneEntityCfg("guide_open"),
    tgt_cfg: SceneEntityCfg = SceneEntityCfg("cabinet"),
    joint_idx: int = 0,
):
    """
    guide_open 의 각 link와 cabinet 의 각 link 를 1:1 매칭해서 Fixed joint 를 만듭니다.
    """
    # 1) env_ids 기본값 설정
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.scene.device)

    # 2) 에셋 가져오기
    src = env.scene[src_cfg.name]   # guide_open
    tgt = env.scene[tgt_cfg.name]   # cabinet

    # 3) 각 에셋의 prim_paths 리스트
    src_paths = src.root_physx_view.prim_paths   # List[List[str]] 혹은 List[str]
    tgt_paths = tgt.root_physx_view.link_paths

    # 4) cabinet 쪽은 원하는 joint_idx 에 해당하는 prim 만 뽑아서 1차원 리스트로 만들기
    #    (예: art_asset_link_prim_list[i] 가 각 env의 path 리스트라면, paths[joint_idx] 로 특정 link 선택)
    tgt_selected = [paths[joint_idx] for paths in tgt_paths]

    # 5) zip 으로 (guide_link, cabinet_link) 쌍 순회하며 joint 생성
    for tgt_prim, src_prim in zip(src_paths, tgt_selected):
        physxUtils.createJoint(
            env.scene.stage,
            "Fixed",
            env.scene.stage.GetPrimAtPath(src_prim),
            env.scene.stage.GetPrimAtPath(tgt_prim)
        )


def reset_guide_and_make_fixed_joints(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cabinet_asset_name: str = "cabinet",
):
    """
    Reset and randomize the position of all guide assets in the scene.
    This function uses the current cabinet position as reference to avoid collisions.
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to reset. If None, reset all environments
        pose_range: Range for position and orientation randomization
        cabinet_asset_name: Name of the cabinet asset to use as reference
    """
    # Set env_ids if None
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.scene.device)
    
    # Get current cabinet position as reference
    try:
        cabinet = env.scene[cabinet_asset_name]
        # Get current cabinet position (not default)
        cabinet_current_pos = cabinet.data.root_pos_w[env_ids].clone()
        cabinet_default_pos = cabinet.data.default_root_state[env_ids, 0:3].clone()
    except KeyError:
        print(f"[WARNING] Cabinet asset '{cabinet_asset_name}' not found, using default positions")
        cabinet_current_pos = None
        cabinet_default_pos = None
    
    # Find all guide assets in the scene
    guide_assets = []
    
    # Check rigid objects
    for asset_name, asset in env.scene.rigid_objects.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    
    # Check articulations (in case guide assets are articulations)
    for asset_name, asset in env.scene.articulations.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    
    # Check extras (in case guide assets are in extras)
    for asset_name, asset in env.scene.extras.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    
    if not guide_assets:
        return

    # Generate random position offset for all guide assets (same for each env)
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=env.scene.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], 
        (len(env_ids), 6), device=env.scene.device
    )
        
    # Process each guide asset
    for asset_name, asset in guide_assets:
        print(f"[INFO] Processing guide asset: {asset_name}")
        
        # Get default root state
        root_states = asset.data.default_root_state[env_ids].clone()
        
        # Calculate base position: use cabinet current position if available, otherwise use default
        if cabinet_current_pos is not None and cabinet_default_pos is not None:
            # Calculate relative position from cabinet's current position
            # This assumes guide assets are positioned relative to cabinet
            cabinet_noise = cabinet_current_pos - env.scene.env_origins[env_ids] - cabinet_default_pos
        
        # Apply position offset on top of the base position
        positions = root_states[:, 0:3] + cabinet_noise + pose_samples[:, 0:3]
        
        # Handle orientation based on asset type
        if 'arrow' in asset_name.lower():
            # For guide_arrow, maintain pointing direction by adjusting orientation
            # Calculate the direction vector from base position to new position
            direction_vector = positions - root_states[:, 0:3]
            
            # If there's significant movement, adjust orientation to maintain pointing direction
            if torch.norm(direction_vector).item() > 1e-6:
                # Get original orientation
                original_quat = root_states[:, 3:7]
                
                # For arrow, we want to maintain the relative pointing direction
                # Calculate the movement magnitude and direction
                movement_magnitude = torch.norm(direction_vector, dim=1, keepdim=True)
                movement_direction = direction_vector / (movement_magnitude + 1e-8)
                
                # Create a rotation that aligns the arrow's forward direction with the movement direction
                # Assuming arrow's forward direction is along its local Z-axis
                # We'll create a quaternion that rotates from (0,0,1) to the movement direction
                forward_vector = torch.tensor([0.0, 0.0, 1.0], device=env.scene.device).unsqueeze(0).repeat(len(env_ids), 1)
                
                # Calculate rotation between forward vector and movement direction
                # Use cross product to find rotation axis
                rotation_axis = torch.cross(forward_vector, movement_direction, dim=1)
                rotation_axis_norm = torch.norm(rotation_axis, dim=1, keepdim=True)
                
                # Normalize rotation axis
                rotation_axis = rotation_axis / (rotation_axis_norm + 1e-8)
                
                # Calculate rotation angle (dot product gives cosine of angle)
                cos_angle = torch.sum(forward_vector * movement_direction, dim=1, keepdim=True)
                cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
                angle = torch.acos(cos_angle)
                
                # Create quaternion from axis-angle representation
                sin_half_angle = torch.sin(angle / 2.0)
                cos_half_angle = torch.cos(angle / 2.0)
                
                # Quaternion: (x*sin(θ/2), y*sin(θ/2), z*sin(θ/2), cos(θ/2))
                rotation_quat = torch.cat([
                    rotation_axis * sin_half_angle,
                    cos_half_angle
                ], dim=1)
                
                # Apply the rotation to original orientation
                orientations = math_utils.quat_mul(original_quat, rotation_quat)
            else:
                orientations = root_states[:, 3:7]
        else:
            # For other guide assets, apply random orientation offset
            orientations_delta = math_utils.quat_from_euler_xyz(
                pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
            )
            orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        
        # Set into the physics simulation
        asset.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1), 
            env_ids=env_ids
        )
        asset.data.default_root_state[env_ids, 0:3] = positions
        
        # Update the default root state to reflect the new base position
        new_root_state = asset.data.default_root_state.clone()
        if cabinet_current_pos is not None and cabinet_default_pos is not None:
            # Update to use cabinet-relative positioning
            cabinet_offset = cabinet_current_pos - cabinet_default_pos
            new_root_state[env_ids, 0:3] = root_states[:, 0:3] + cabinet_offset
        else:
            # Add the pose offset to default
            new_root_state[env_ids, 0:3] += pose_samples[:, 0:3]
        asset._data.default_root_state = new_root_state

    # if env.scene.stage.GetPrimAtPath("/World/Cabinet/drawer_top/guide_arrow"):
    #     env.scene.stage.RemovePrim(env.scene.stage.GetPrimAtPath("/World/Cabinet/drawer_top/guide_arrow"))
    # if env.scene.stage.GetPrimAtPath("/World/Cabinet/drawer_top/guide_open"):
    #     env.scene.stage.RemovePrim(env.scene.stage.GetPrimAtPath("/World/Cabinet/drawer_top/guide_open"))

    # src_cfg = [SceneEntityCfg("guide_open"), SceneEntityCfg("guide_arrow")]
    # tgt_cfg = SceneEntityCfg("cabinet")
    # joint_idx = [4, 4]

    # for src_cfg, joint_idx in zip(src_cfg, joint_idx):
    #     # 2) 에셋 가져오기
    #     src = env.scene[src_cfg.name]   # guide_open
    #     tgt = env.scene[tgt_cfg.name]   # cabinet

    #     # 3) 각 에셋의 prim_paths 리스트
    #     src_paths = src.root_physx_view.prim_paths   # List[List[str]] 혹은 List[str]
    #     tgt_paths = tgt.root_physx_view.link_paths

    #     # 4) cabinet 쪽은 원하는 joint_idx 에 해당하는 prim 만 뽑아서 1차원 리스트로 만들기
    #     #    (예: art_asset_link_prim_list[i] 가 각 env의 path 리스트라면, paths[joint_idx] 로 특정 link 선택)
    #     tgt_selected = [paths[joint_idx] for paths in tgt_paths]

    #     # 5) zip 으로 (guide_link, cabinet_link) 쌍 순회하며 joint 생성
    #     for tgt_prim, src_prim in zip(src_paths, tgt_selected):
    #         physxUtils.createJoint(
    #             env.scene.stage,
    #             "Fixed",
    #             env.scene.stage.GetPrimAtPath(src_prim),
    #             env.scene.stage.GetPrimAtPath(tgt_prim)
    #         )

def change_guide_side(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
):
    """
    env_ids 중에서 무작위로 하나만 골라 해당 env의 guide asset만 y축(각 env origin 기준) 반전.
    arrow면 orientation도 180도(yaw) 반전.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.scene.device)
    # 무작위로 하나 선택
    if len(env_ids) == 1:
        chosen = env_ids[0]
    else:
        idx = torch.randint(0, len(env_ids), (1,), device=env_ids.device)
        chosen = env_ids[idx]
    chosen = chosen.unsqueeze(0) if chosen.ndim == 0 else chosen  # shape (1,)

    # origin
    origin = env.scene["cabinet"].data.root_pos_w[chosen]

    # guide asset 처리
    guide_assets = []
    for asset_name, asset in env.scene.rigid_objects.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    for asset_name, asset in env.scene.articulations.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    for asset_name, asset in env.scene.extras.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    if not guide_assets:
        return

    for asset_name, asset in guide_assets:
        root_states = asset.data.default_root_state[chosen].clone()
        # y좌표만 origin 기준 반전
        root_states[:, 1] = - root_states[:, 1]
        root_states[:, 0:3] = root_states[:, 0:3] + origin
        # arrow면 orientation도 반전
        if 'arrow' in asset_name.lower():
            roll, pitch, yaw = math_utils.euler_xyz_from_quat(root_states[:, 3:7])
            yaw += np.pi
            new_quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
            root_states[:, 3:7] = new_quat
        asset.write_root_pose_to_sim(root_states[:, 0:7], env_ids=chosen)

def reset_guide_position_with_random_flip(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    cabinet_asset_name: str = "cabinet",
    do_random_flip: bool = False,
):
    """
    guide object 위치 랜덤화 + (옵션) 무작위로 한 env만 side flip!
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.scene.device)
    # cabinet 기준 위치 계산 (생략, 기존 코드와 동일)
    try:
        cabinet = env.scene[cabinet_asset_name]
        cabinet_current_pos = cabinet.data.root_pos_w[env_ids].clone()
        cabinet_default_pos = cabinet.data.default_root_state[env_ids, 0:3].clone()
    except KeyError:
        cabinet_current_pos = None
        cabinet_default_pos = None

    guide_assets = []
    for asset_name, asset in env.scene.rigid_objects.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    for asset_name, asset in env.scene.articulations.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    for asset_name, asset in env.scene.extras.items():
        if 'guide' in asset_name.lower():
            guide_assets.append((asset_name, asset))
    if not guide_assets:
        return

    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=env.scene.device)
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], 
        (len(env_ids), 6), device=env.scene.device
    )

    # 무작위로 하나 선택 (do_random_flip이 True일 때만)
    flip_idx = None
    if do_random_flip and len(env_ids) > 0:
        rand_idx = torch.randint(0, len(env_ids), (1,), device=env_ids.device)
        flip_idx = rand_idx.item()

    for asset_name, asset in guide_assets:
        root_states = asset.data.default_root_state[env_ids].clone()
        if cabinet_current_pos is not None and cabinet_default_pos is not None:
            cabinet_noise = cabinet_current_pos - env.scene.env_origins[env_ids] - cabinet_default_pos
        guide_positions = root_states[:, 0:3] + pose_samples[:, 0:3]
        orientations = root_states[:, 3:7]
        # 무작위로 고른 env만 flip
        if flip_idx is not None:
            origin = env.scene["cabinet"].data.root_pos_w[flip_idx]
            # y좌표 반전 (origin 기준)
            guide_positions[flip_idx, 1] = - guide_positions[flip_idx, 1]
            # arrow면 orientation 반전
            if 'arrow' in asset_name.lower():
                roll, pitch, yaw = math_utils.euler_xyz_from_quat(orientations[flip_idx].unsqueeze(0))
                yaw += np.pi
                orientations[flip_idx] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)[0]
        positions = guide_positions[:, 0:3] + cabinet_noise + env.scene.env_origins[env_ids]

        asset.write_root_pose_to_sim(
            torch.cat([positions, orientations], dim=-1),
            env_ids=env_ids
        )

def reset_root_state_uniform_ori(
    env: ManagerBasedRLStepEnv,
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
    env.asset_ori_delta = rand_samples[:,3:6]
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def reset_camera_followup(
    env: ManagerBasedRLStepEnv,
    env_ids: torch.Tensor,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_top"),
    tgt_asset_cfg: SceneEntityCfg = SceneEntityCfg("bottle")
):
    """
    Set the camera position to follow the location of the tgt_asset.
    """
    # extract the used quantities (to enable type-hinting)
    # asset: 
    camera: Camera = env.scene[camera_cfg.name]
    target: RigidObject | Articulation = env.scene[tgt_asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=camera.device)

    # 2. 타겟의 현재 위치(월드) 얻기
    cam_pos = target.data.root_state_w[env_ids, 0:3].clone()  # (num_envs, 3)

    # 3. 카메라 위치 계산 (타겟 위치 + 오프셋)
    cam_pos[:,2] += 0.5

    # 6. 시뮬레이터에 적용
    camera.set_world_poses(cam_pos, env_ids=env_ids, convention="world")


def find_handle_spawn_guide(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    cabinet_asset_name: str = "cabinet",
):
    """find handle with prim string -> sampling from candidate plane -> fix guide positions"""
    pass


# Replicator 확장 on/off 유틸
try:
    from isaacsim.core.utils.extensions import enable_extension
except ImportError:
    # 구버전(2023/2024)에서 흔히 쓰는 경로
    from omni.isaac.core.utils.extensions import enable_extension

# 이벤트/매니저 베이스 클래스
from isaaclab.managers import ManagerTermBase, EventTermCfg 

def randomize_visual_color(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
    mesh_name: str = "",
    event_name: str = "randomize_color",
):
    """Randomize visual color using Replicator. Designed for isaaclab (function-style events.py)."""

    # ---- setup replicator ----
    enable_extension("omni.replicator.core")
    import omni.replicator.core as rep

    if env.cfg.scene.replicate_physics:
        raise RuntimeError(
            "Unable to randomize visual color with scene replication enabled. "
            "Set 'replicate_physics=False' in InteractiveSceneCfg."
        )

    asset = env.scene[asset_cfg.name]
    if not mesh_name.startswith("/"):
        mesh_name = "/" + mesh_name
    mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"

    # 색상 파싱
    if isinstance(colors, dict):
        low = [colors[k][0] for k in ("r", "g", "b")]
        high = [colors[k][1] for k in ("r", "g", "b")]
        colors_rep = rep.distribution.uniform(low, high)
    else:
        colors_rep = list(colors)

    # 그래프가 중복 생성되는 것을 막고 싶다면 전역 캐시 사용
    global _COLOR_RAND_CACHE
    try:
        _COLOR_RAND_CACHE
    except NameError:
        _COLOR_RAND_CACHE = {}

    key = (mesh_prim_path, event_name)
    if key not in _COLOR_RAND_CACHE:
        def _build_graph():
            prims = rep.get.prims(path_pattern=mesh_prim_path)
            with prims:
                rep.randomizer.color(colors=colors_rep)
            return prims.node

        with rep.trigger.on_custom_event(event_name=event_name):
            node = _build_graph()
        _COLOR_RAND_CACHE[key] = node

    # 트리거 실행
    rep.utils.send_og_event(event_name)