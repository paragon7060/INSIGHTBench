import torch
import matplotlib.pyplot as plt
import os
import glob
import re
import json
import h5py
import numpy as np
import math
import cv2
import shutil

from isaaclab.utils.math import quat_from_euler_xyz, euler_xyz_from_quat

from custom_lab.envs.manager_based_rl_step_env import ManagerBasedRLStepEnv
"""Rest everything follows."""
from cfg.BaseCfg import CommandsCfg, TerminationsCfg, ActionsCfg, ObsCfg
from cfg.BaseEnvCfg import DynamicEnvCfg, extract_batched_guide_info
# from cfg.scene2Cfg
from cfg.BaseTaskCfg import(
    SCENE_CLASSES,
    OBS_CLASSES,
    REW_CLASSES,
    EVENT_CLASSES,
    TARGET_OBJECT_NAME,
)
from cfg.ActionRangeCfg import SCENE_ACTION_CONFIG

def sample_actions_from_body_pose(
    env: ManagerBasedRLStepEnv,
    scene_key: str,
    step_idx: int,
    num_envs: int,
    device: torch.device
) -> torch.Tensor:
    """
    bbox_json 의 cover 정보로부터 endeffector가 가야할 적절한 posiion을 추출
    rotation 의 경우 guide_mode 에 따라서 scene_key 에 따라서 범위 선택
    """
    cfg = SCENE_ACTION_CONFIG[scene_key]
    seq = cfg["skill_sequence"]
    n_skills = len(seq)
    skill_id = seq[step_idx % n_skills]
    skill_id_col = torch.full((num_envs, 1), float(skill_id), device=device)


    if skill_id in {0,1}:
        # ───────────────────────────── 스킬 1 ──────────────────────────────
        handle_pos = env.scene["door"].data.body_pos_w[:,-1,:] - env.scene.env_origins

        sampling_half_width = torch.tensor([0.00, 0.000, 0.00], device=device)
        random_offsets = (2 * torch.rand_like(handle_pos) - 1) * sampling_half_width
        pos_stack = handle_pos + random_offsets
        pos_stack[:,0] -= 0.04

        # 2) Euler 각(roll, pitch, yaw) 샘플 (단위: 도→라디안)
        euler_ranges = cfg["param_ranges"][1][3:]
        # euler_ranges = [[roll_lo, roll_hi], [pitch_lo, pitch_hi], [yaw_lo, yaw_hi]] # (단위: 도)
        roll_deg  = (euler_ranges[0][1]  - euler_ranges[0][0])  * torch.rand(num_envs, device=device) + euler_ranges[0][0]
        pitch_deg = (euler_ranges[1][1]  - euler_ranges[1][0])  * torch.rand(num_envs, device=device) + euler_ranges[1][0]
        yaw_deg   = (euler_ranges[2][1]  - euler_ranges[2][0])  * torch.rand(num_envs, device=device) + euler_ranges[2][0]
        # 라디안 변환
        roll_rad  = roll_deg  * math.pi / 180.0
        pitch_rad = pitch_deg * math.pi / 180.0
        yaw_rad   = yaw_deg   * math.pi / 180.0

        # 3) Isaac Lab 내장 함수로 쿼터니언 생성 (XYZ 순서의 Euler → (w,x,y,z) 반환)
        quat_batch = quat_from_euler_xyz(roll_rad, pitch_rad, yaw_rad)  # shape=(num_envs, 4)

        # 4) action 텐서 결합: [skill, x, y, z, qw, qx, qy, qz]
        action_body = torch.cat([pos_stack, quat_batch], dim=1)  # shape=(num_envs,7)
        return torch.cat([skill_id_col, action_body], dim=1)     # 최종 shape=(num_envs,8)

    elif skill_id in {3,4}:
        # ───────────────────────────── 스킬 4 ──────────────────────────────
        # 1) distance만 샘플 (나머지 파라미터는 모두 0)
        dist_range = cfg["param_ranges"][skill_id][0]  # 예: [-1.1, -0.3]
        dists = (dist_range[1] - dist_range[0]) * torch.rand(num_envs, device=device) + dist_range[0]

        # 2) 나머지 6개 파라미터를 0으로 채움
        zeros_pos  = torch.zeros((num_envs, 2), device=device)  # y,z 는 0
        zeros_quat = torch.zeros((num_envs, 4), device=device)  # qw,qx,qy,qz 는 모두 0

        # skill=4의 최종 action: [4, distance, 0, 0, 0, 0, 0, 0]
        action_body = torch.cat([dists.unsqueeze(1), zeros_pos, zeros_quat], dim=1)  # shape=(num_envs,7)
        return torch.cat([skill_id_col, action_body], dim=1)  # shape=(num_envs,8)

    else:
        raise ValueError(f"지원하지 않는 skill_id: {skill_id}")
    

def sample_actions_from_bbox(
    env: ManagerBasedRLStepEnv,
    scene_key: str,
    bbox_info: dict,
    step_idx: int,
    num_envs: int,
    device: torch.device
) -> torch.Tensor:
    """
    bbox_json 의 cover 정보로부터 endeffector가 가야할 적절한 posiion을 추출
    rotation 의 경우 guide_mode 에 따라서 scene_key 에 따라서 범위 선택
    """
    cfg = SCENE_ACTION_CONFIG[scene_key]
    seq = cfg["skill_sequence"]
    n_skills = len(seq)
    skill_id = seq[step_idx % n_skills]
    skill_id_col = torch.full((num_envs, 1), float(skill_id), device=device)
    bbox_center = bbox_info["center"]
    z_max = bbox_info["max"][-1]


    if skill_id in {0,1}:
        # ───────────────────────────── 스킬 1 ──────────────────────────────
        # 1) 위치(x,y,z)부터 샘플
        pos_ranges = [[-0.005, +0.005],[-0.005, +0.005],[bbox_center[2]-0.01, z_max+0.0]]
        xs = (pos_ranges[0][1] - pos_ranges[0][0]) * torch.rand(num_envs, device=device) + pos_ranges[0][0]
        ys = (pos_ranges[1][1] - pos_ranges[1][0]) * torch.rand(num_envs, device=device) + pos_ranges[1][0]
        zs = (pos_ranges[2][1] - pos_ranges[2][0]) * torch.rand(num_envs, device=device) + pos_ranges[2][0]

        # 2) Euler 각(roll, pitch, yaw) 샘플 (단위: 도→라디안)
        euler_ranges = cfg["param_ranges"][1][3:]
        # euler_ranges = [[roll_lo, roll_hi], [pitch_lo, pitch_hi], [yaw_lo, yaw_hi]] # (단위: 도)
        roll_deg  = (euler_ranges[0][1]  - euler_ranges[0][0])  * torch.rand(num_envs, device=device) + euler_ranges[0][0]
        pitch_deg = (euler_ranges[1][1]  - euler_ranges[1][0])  * torch.rand(num_envs, device=device) + euler_ranges[1][0]
        yaw_deg   = (euler_ranges[2][1]  - euler_ranges[2][0])  * torch.rand(num_envs, device=device) + euler_ranges[2][0]
        # 라디안 변환
        roll_rad  = roll_deg  * math.pi / 180.0
        pitch_rad = pitch_deg * math.pi / 180.0
        yaw_rad   = yaw_deg   * math.pi / 180.0

        if scene_key in {"5a", "5g"}:
            # asset_root_state = env.scene["bottle"].data.root_state_w[:,3:7].clone()
            # ori_delta = quat_mul(quat_conjugate(env.scene["bottle"].data.default_root_state[:,3:7]), asset_root_state)
            # q_delta = euler_xyz_from_quat(ori_delta)
            # yaw_rad -= q_delta[0]
            ori_delta = env.asset_ori_delta
            yaw_rad -= ori_delta[:,0]

        # 3) Isaac Lab 내장 함수로 쿼터니언 생성 (XYZ 순서의 Euler → (w,x,y,z) 반환)
        quat_batch = quat_from_euler_xyz(roll_rad, pitch_rad, yaw_rad)  # shape=(num_envs, 4)

        # 4) action 텐서 결합: [skill, x, y, z, qw, qx, qy, qz]
        pos_stack  = torch.stack([xs, ys, zs], dim=1)      # shape=(num_envs,3)
        action_body = torch.cat([pos_stack, quat_batch], dim=1)  # shape=(num_envs,7)
        return torch.cat([skill_id_col, action_body], dim=1)     # 최종 shape=(num_envs,8)

    elif skill_id in {3,4}:
        # ───────────────────────────── 스킬 4 ──────────────────────────────
        # 1) distance만 샘플 (나머지 파라미터는 모두 0)
        dist_range = cfg["param_ranges"][skill_id][0]  # 예: [-1.1, -0.3]
        dists = (dist_range[1] - dist_range[0]) * torch.rand(num_envs, device=device) + dist_range[0]

        # 2) 나머지 6개 파라미터를 0으로 채움
        zeros_pos  = torch.zeros((num_envs, 2), device=device)  # y,z 는 0
        zeros_quat = torch.zeros((num_envs, 4), device=device)  # qw,qx,qy,qz 는 모두 0

        # skill=4의 최종 action: [4, distance, 0, 0, 0, 0, 0, 0]
        action_body = torch.cat([dists.unsqueeze(1), zeros_pos, zeros_quat], dim=1)  # shape=(num_envs,7)
        return torch.cat([skill_id_col, action_body], dim=1)  # shape=(num_envs,8)

    else:
        raise ValueError(f"지원하지 않는 skill_id: {skill_id}")
    

def sample_actions_for_cabinet(
    env: ManagerBasedRLStepEnv,
    scene_key: str,
    step_idx: int,
    num_envs: int,
    info_handle_point: list,
    device: torch.device,
    handle_type: str|None = None,
) -> torch.Tensor:
    """
    각 스텝에서 스킬에 맞게 파라미터를 샘플링하여
    torch.Tensor(shape=(num_envs, 8))으로 반환합니다.
      - [:,0]   = skill_id (float)
      - [:,1:4] = 스킬별로 달라지는 연속 파라미터 (예: 스킬1은 x,y,z; 스킬4는 distance, 그 외 0)
      - [:,4:8] = 스킬1인 경우 [qw,qx,qy,qz], 스킬4인 경우 모두 0
    """
    cfg = SCENE_ACTION_CONFIG[scene_key]
    seq = cfg["skill_sequence"]
    n_skills = len(seq)

    # 현재 step에서 수행할 skill_id를 순환 방식으로 결정
    skill_id = seq[step_idx % n_skills]
    skill_id_col = torch.full((num_envs, 1), float(skill_id), device=device)

    if skill_id in {0,1}:
        # ───────────────────────────── 스킬 1 ──────────────────────────────
        # 1) 위치(x,y,z)부터 샘플
        low = info_handle_point[0][0] + 0.1 * (info_handle_point[0][1] - info_handle_point[0][0])
        high = info_handle_point[0][1] - 0.1 * (info_handle_point[0][1] - info_handle_point[0][0]) 
        action_list = []
        for _ in range(num_envs):
            act = np.random.uniform(low, high)
            act[0] += 0.005
            action_list.append(act)
        pos_stack = torch.tensor(np.array(action_list),device=device)
        
        # 2) Euler 각(roll, pitch, yaw) 샘플 (단위: 도→라디안)
        if handle_type == "vertical":
            euler_ranges = [[80,100],[70,110],[80,100]]
        elif handle_type == "horizontal":
            euler_ranges = [[80,100],[160, 200],[80,100]]
        else:
            euler_ranges = cfg["param_ranges"][1][3:]

        # euler_ranges = [[roll_lo, roll_hi], [pitch_lo, pitch_hi], [yaw_lo, yaw_hi]] (단위: 도)
        roll_deg  = (euler_ranges[0][1]  - euler_ranges[0][0])  * torch.rand(num_envs, device=device) + euler_ranges[0][0]
        pitch_deg = (euler_ranges[1][1]  - euler_ranges[1][0])  * torch.rand(num_envs, device=device) + euler_ranges[1][0]
        yaw_deg   = (euler_ranges[2][1]  - euler_ranges[2][0])  * torch.rand(num_envs, device=device) + euler_ranges[2][0]
        # 라디안 변환
        roll_rad  = roll_deg  * math.pi / 180.0
        pitch_rad = pitch_deg * math.pi / 180.0
        yaw_rad   = yaw_deg   * math.pi / 180.0

        # 3) Isaac Lab 내장 함수로 쿼터니언 생성 (XYZ 순서의 Euler → (w,x,y,z) 반환)
        quat_batch = quat_from_euler_xyz(roll_rad, pitch_rad, yaw_rad)  # shape=(num_envs, 4)

        # 4) action 텐서 결합: [skill, x, y, z, qw, qx, qy, qz]
        action_body = torch.cat([pos_stack, quat_batch], dim=1)  # shape=(num_envs,7)
        return torch.cat([skill_id_col, action_body], dim=1)     # 최종 shape=(num_envs,8)

    elif skill_id in {3,4}:
        # ───────────────────────────── 스킬 4 ──────────────────────────────
        # 1) distance만 샘플 (나머지 파라미터는 모두 0)
        dist_range = cfg["param_ranges"][skill_id][0]  # 예: [-1.1, -0.3]
        dists = (dist_range[1] - dist_range[0]) * torch.rand(num_envs, device=device) + dist_range[0]

        # 2) 나머지 6개 파라미터를 0으로 채움
        zeros_pos  = torch.zeros((num_envs, 2), device=device)  # y,z 는 0
        zeros_quat = torch.zeros((num_envs, 4), device=device)  # qw,qx,qy,qz 는 모두 0

        # skill=4의 최종 action: [4, distance, 0, 0, 0, 0, 0, 0]
        action_body = torch.cat([dists.unsqueeze(1), zeros_pos, zeros_quat], dim=1)  # shape=(num_envs,7)
        return torch.cat([skill_id_col, action_body], dim=1)  # shape=(num_envs,8)

    else:
        raise ValueError(f"지원하지 않는 skill_id: {skill_id}")