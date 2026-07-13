# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LeRobot 형식으로 door manipulation 데이터를 기록하는 스크립트.

LeRobot의 표준화된 데이터셋 형식을 사용하여 door manipulation 시퀀스를 기록합니다.

사용법:
    ./isaaclab.sh -p source/project/asset_lab/lerobot_door_recorder.py --dataset.repo_id=your_username/door_dataset --dataset.num_episodes=10
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Any, List

from isaaclab.app import AppLauncher

# CLI 설정
parser = argparse.ArgumentParser(description="LeRobot 형식으로 door manipulation 데이터 기록")
parser.add_argument("--num_envs", type=int, default=4, help="생성할 환경 수")
parser.add_argument("--scene", type=str, default="3ext", help="사용할 scene 설정")
parser.add_argument("--object", type=str, default="door", help="조작할 객체")

# LeRobot 스타일의 데이터셋 설정
parser.add_argument("--dataset.repo_id", type=str, default="local/door_dataset", help="데이터셋 저장소 ID (예: username/dataset_name)")
parser.add_argument("--dataset.num_episodes", type=int, default=50, help="기록할 에피소드 수")
parser.add_argument("--dataset.episode_time_s", type=float, default=20.0, help="에피소드당 기록 시간(초)")
parser.add_argument("--dataset.fps", type=int, default=30, help="기록 FPS")
parser.add_argument("--dataset.single_task", type=str, default="Open the door", help="수행할 작업 설명")
parser.add_argument("--dataset.root", type=str, default=None, help="로컬 저장 경로")
parser.add_argument("--dataset.push_to_hub", action="store_true", help="HuggingFace Hub에 업로드")
parser.add_argument("--dataset.private", action="store_true", help="비공개 저장소로 업로드")

# Isaac Lab 설정
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""시뮬레이터 시작 후 import"""
import torch
import numpy as np
import json
from dataclasses import dataclass
from typing import Optional

# LeRobot 데이터셋 관련 import
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
    LEROBOT_AVAILABLE = True
except ImportError:
    print("[WARN] LeRobot not available, using fallback dataset format")
    LEROBOT_AVAILABLE = False

# Isaac Lab imports
from cfg.BaseCfg import CommandsCfg, TerminationsCfg, ActionsCfg, TARGET_OBJECT_NAME
from cfg.BaseEnvCfg import DynamicEnvCfg
from cfg.scene1Cfg import Scene1ObsCfg
from cfg.BaseCfg import SCENE_CLASSES, OBS_CLASSES, REW_CLASSES, EVENT_CLASSES
from cfg.helper import sample_urdf_get_info
from cfg.scene3ExtCfg import make_door_scene_cfg
from custom_lab.envs.manager_based_rl_step_env import ManagerBasedContinuousEnv
from isaaclab.utils.math import quat_from_euler_xyz
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext


@dataclass
class LeRobotDoorConfig:
    """LeRobot 스타일의 door recording 설정"""
    repo_id: str
    num_episodes: int = 50
    episode_time_s: float = 20.0
    fps: int = 30
    single_task: str = "Open the door"
    root: Optional[str] = None
    push_to_hub: bool = True
    private: bool = False


class DoorDataRecorder:
    """LeRobot 형식으로 door manipulation 데이터를 기록하는 클래스"""
    
    def __init__(self, config: LeRobotDoorConfig):
        self.config = config
        self.dataset = None
        self.episode_buffer = None
        self.current_episode = 0
        
        if LEROBOT_AVAILABLE:
            self._setup_lerobot_dataset()
        else:
            self._setup_fallback_dataset()
    
    def _setup_lerobot_dataset(self):
        """LeRobot 데이터셋 설정"""
        # 로봇 액션/관찰 특성 정의 (실제 로봇 대신 시뮬레이션 환경에 맞게 조정)
        action_features = {
            "skill_id": {"dtype": "float32", "shape": [1]},
            "ee_position": {"dtype": "float32", "shape": [3]},
            "ee_orientation": {"dtype": "float32", "shape": [4]},
        }
        
        observation_features = {
            "ee_position": {"dtype": "float32", "shape": [3]},
            "ee_orientation": {"dtype": "float32", "shape": [4]},
            "joint_positions": {"dtype": "float32", "shape": [7]},
            "door_joint_position": {"dtype": "float32", "shape": [1]},
            "door_joint_velocity": {"dtype": "float32", "shape": [1]},
        }
        
        # 비디오 카메라가 있다면 추가
        if hasattr(self, 'camera_features'):
            observation_features.update(self.camera_features)
        
        # LeRobot 데이터셋 특성으로 변환
        action_features_hw = hw_to_dataset_features(action_features, "action", use_videos=False)
        obs_features_hw = hw_to_dataset_features(observation_features, "observation", use_videos=False)
        dataset_features = {**action_features_hw, **obs_features_hw}
        
        # 데이터셋 생성
        self.dataset = LeRobotDataset.create(
            repo_id=self.config.repo_id,
            fps=self.config.fps,
            features=dataset_features,
            robot_type="isaac_sim_door",
            root=self.config.root,
            use_videos=False,  # 이미지로 저장
        )
        
        # 이미지 writer 시작
        self.dataset.start_image_writer(num_processes=0, num_threads=4)
    
    def _setup_fallback_dataset(self):
        """LeRobot이 없을 때 사용할 fallback 데이터셋"""
        print("[INFO] Using fallback dataset format (HDF5)")
        self.dataset = None
        self.episode_buffer = []
    
    def add_frame(self, observation: Dict[str, Any], action: Dict[str, Any], task: str):
        """프레임 추가"""
        if LEROBOT_AVAILABLE and self.dataset is not None:
            # LeRobot 형식으로 프레임 구성
            obs_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")
            action_frame = build_dataset_frame(self.dataset.features, action, prefix="action")
            frame = {**obs_frame, **action_frame}
            
            self.dataset.add_frame(frame, task=task)
        else:
            # Fallback: 간단한 딕셔너리로 저장
            frame = {
                "observation": observation,
                "action": action,
                "task": task,
                "timestamp": time.time()
            }
            self.episode_buffer.append(frame)
    
    def save_episode(self):
        """에피소드 저장"""
        if LEROBOT_AVAILABLE and self.dataset is not None:
            self.dataset.save_episode()
            print(f"[INFO] Saved episode {self.current_episode}")
        else:
            # Fallback: JSON으로 저장
            episode_dir = Path(self.config.root or "./demos/lerobot_fallback") / f"episode_{self.current_episode:06d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            
            episode_data = {
                "episode_id": self.current_episode,
                "task": self.config.single_task,
                "fps": self.config.fps,
                "frames": self.episode_buffer
            }
            
            with open(episode_dir / "episode.json", "w") as f:
                json.dump(episode_data, f, indent=2)
            
            print(f"[INFO] Saved fallback episode {self.current_episode}")
            self.episode_buffer = []
        
        self.current_episode += 1
    
    def close(self):
        """데이터셋 정리"""
        if LEROBOT_AVAILABLE and self.dataset is not None:
            self.dataset.stop_image_writer()
            if self.config.push_to_hub:
                try:
                    self.dataset.push_to_hub(private=self.config.private)
                    print(f"[INFO] Dataset pushed to Hub: {self.config.repo_id}")
                except Exception as e:
                    print(f"[WARN] Failed to push to Hub: {e}")
                    print("[INFO] Dataset saved locally only")
        else:
            print(f"[INFO] Dataset saved locally to: {self.config.root or './demos/lerobot_fallback'}")


def sample_actions_from_door_pose(env: ManagerBasedRLStepEnv, scene_key: str, step_idx: int, 
                                 num_envs: int, device: torch.device) -> torch.Tensor:
    """door pose에서 액션 샘플링 (기존 코드와 유사하지만 LeRobot 형식에 맞게 조정)"""
    # 기존 sample_actions_from_body_pose 로직을 여기에 구현
    # LeRobot의 action_features 형식에 맞게 반환
    
    # 예시 구현 (실제로는 기존 로직을 사용)
    skill_id = torch.full((num_envs, 1), 0.0, device=device)
    ee_pos = torch.randn((num_envs, 3), device=device) * 0.1
    ee_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).expand(num_envs, 4)
    
    return torch.cat([skill_id, ee_pos, ee_quat], dim=1)


def record_door_episode(env: ManagerBasedRLStepEnv, recorder: DoorDataRecorder, 
                       scene_key: str, episode_time_s: float, fps: int):
    """단일 door manipulation 에피소드 기록"""
    # 환경 리셋
    obs_batch = env.reset()[0]
    
    # 초기 관찰 기록
    initial_obs = {
        "ee_position": obs_batch["policy"][:, 0:3].cpu().numpy(),
        "ee_orientation": obs_batch["policy"][:, 3:7].cpu().numpy(),
        "joint_positions": obs_batch["policy"][:, 7:14].cpu().numpy(),
        "door_joint_position": env.scene["door"].data.joint_pos[:, 0].cpu().numpy(),
        "door_joint_velocity": env.scene["door"].data.joint_vel[:, 0].cpu().numpy(),
    }
    
    # 초기 액션 (no-op)
    noop_action = torch.tensor([[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * env.num_envs, device=env.device)
    recorder.add_frame(initial_obs, {"skill_id": [4.0], "ee_position": [0,0,0], "ee_orientation": [1,0,0,0]}, 
                      recorder.config.single_task)
    
    # 시뮬레이션 루프
    max_steps = int(episode_time_s * fps / env.cfg.decimation)
    frame_count = 0
    
    for step_idx in range(max_steps):
        # 액션 샘플링
        action_batch = sample_actions_from_door_pose(env, scene_key, step_idx, env.num_envs, env.device)
        
        # 환경 스텝
        obs_batch, rew_b, done_b, _, _ = env.step(action_batch)
        
        # 관찰 및 액션을 LeRobot 형식으로 변환
        observation = {
            "ee_position": obs_batch["policy"][:, 0:3].cpu().numpy(),
            "ee_orientation": obs_batch["policy"][:, 3:7].cpu().numpy(),
            "joint_positions": obs_batch["policy"][:, 7:14].cpu().numpy(),
            "door_joint_position": env.scene["door"].data.joint_pos[:, 0].cpu().numpy(),
            "door_joint_velocity": env.scene["door"].data.joint_vel[:, 0].cpu().numpy(),
        }
        
        action = {
            "skill_id": action_batch[:, 0].cpu().numpy(),
            "ee_position": action_batch[:, 1:4].cpu().numpy(),
            "ee_orientation": action_batch[:, 4:8].cpu().numpy(),
        }
        
        # 프레임 추가
        recorder.add_frame(observation, action, recorder.config.single_task)
        frame_count += 1
        
        # FPS 제어
        time.sleep(1.0 / fps)
        
        # 종료 조건
        if done_b.all():
            break
    
    print(f"[INFO] Recorded {frame_count} frames for episode {recorder.current_episode}")


def main():
    """메인 함수"""
    # 설정 파싱
    config = LeRobotDoorConfig(
        repo_id=args_cli.dataset.repo_id,
        num_episodes=args_cli.dataset.num_episodes,
        episode_time_s=args_cli.dataset.episode_time_s,
        fps=args_cli.dataset.fps,
        single_task=args_cli.dataset.single_task,
        root=args_cli.dataset.root,
        push_to_hub=args_cli.dataset.push_to_hub,
        private=args_cli.dataset.private,
    )
    
    # 데이터 레코더 초기화
    recorder = DoorDataRecorder(config)
    
    print(f"[INFO] Starting LeRobot door recording: {config.repo_id}")
    print(f"[INFO] Target episodes: {config.num_episodes}")
    print(f"[INFO] Episode duration: {config.episode_time_s}s")
    print(f"[INFO] FPS: {config.fps}")
    
    # 에피소드별 기록
    for episode in range(config.num_episodes):
        print(f"\n[INFO] Recording episode {episode + 1}/{config.num_episodes}")
        
        # Scene 설정 (기존 코드에서 가져옴)
        if args_cli.object == "door":
            dir_path = "Assets/AdaManip/assets/door"
        
        # Scene 샘플링
        usd_path, guide_path, asset_init_pos, guide_init_pos, asset_init_pos, door_scale, pull_mode, ccw_mode, scene_key = \
            sample_urdf_get_info(dir_path, TARGET_OBJECT_NAME[args_cli.scene[0]])
        
        # Scene 설정 생성
        scene_cfg, reward_cfg = make_door_scene_cfg(
            usd_path=usd_path,
            guide_path=guide_path,
            asset_init_pos=asset_init_pos,
            guide_init_pos=guide_init_pos,
            door_scale=door_scale,
        )
        
        scene_cfg.num_envs = args_cli.num_envs
        scene_cfg.env_spacing = 2.0
        
        # 환경 설정
        env_cfg = DynamicEnvCfg(
            scene=scene_cfg,
            observations=Scene1ObsCfg(),
            commands=CommandsCfg(),
            actions=ActionsCfg(),
            rewards=reward_cfg,
            terminations=TerminationsCfg(),
            events=EVENT_CLASSES[args_cli.scene](),
        )
        env_cfg.episode_length_s = config.episode_time_s
        
        # 환경 생성
        env = ManagerBasedRLStepEnv(cfg=env_cfg)
        
        try:
            # 에피소드 기록
            record_door_episode(env, recorder, scene_key, config.episode_time_s, config.fps)
            
            # 에피소드 저장
            recorder.save_episode()
            
        finally:
            env.close()
    
    # 데이터셋 정리
    recorder.close()
    
    print(f"\n[INFO] Recording completed!")
    if config.push_to_hub:
        print(f"[INFO] Dataset pushed to HuggingFace Hub: https://huggingface.co/datasets/{config.repo_id}")
    else:
        print(f"[INFO] Dataset saved locally to: {config.root or './demos/lerobot_fallback'}")


if __name__ == "__main__":
    main()
    simulation_app.close() 