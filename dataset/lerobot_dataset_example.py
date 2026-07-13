#!/usr/bin/env python3
"""
LeRobot 데이터셋 사용 예제

이 스크립트는 IsaacLab에서 수집한 데이터를 LeRobot 형식으로 변환하고 사용하는 방법을 보여줍니다.
환경별로 따로 에피소드에 저장하는 방법을 포함합니다.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

def create_simple_lerobot_dataset():
    """간단한 LeRobot 데이터셋 생성 예제"""
    
    # 데이터셋 특성 정의
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (12,),  # end-effector position(3) + quaternion(4) + joint positions(5)
            "names": ["ee_x", "ee_y", "ee_z", "ee_qw", "ee_qx", "ee_qy", "ee_qz", "j1", "j2", "j3", "j4", "j5"]
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": (3, 224, 224),
            "names": ["height", "width", "channels"]
        },
        "action": {
            "dtype": "float32", 
            "shape": (8,),  # 7 joint positions + 1 gripper
            "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["reward"]
        }
    }
    
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # LeRobot 데이터셋 생성
        dataset = LeRobotDataset.create(
            repo_id="isaaclab_demo_dataset",
            fps=60,
            features=features,
            root="./data/lerobot_demo",
            robot_type="franka_panda",
            use_videos=False,  # 이미지로 저장
            batch_encoding_size=1
        )
        
        print("LeRobot 데이터셋이 성공적으로 생성되었습니다!")
        return dataset
        
    except ImportError:
        print("LeRobot 라이브러리가 설치되지 않았습니다.")
        print("다음 명령어로 설치하세요:")
        print("pip install lerobot")
        return None

def add_sample_episode_multi_env(dataset, num_envs=2):
    """여러 환경에서 샘플 에피소드 데이터 추가 (환경별로 따로 저장)"""
    
    if dataset is None:
        return
    
    # 각 환경별로 에피소드 버퍼 생성
    episode_buffers = []
    for env_idx in range(num_envs):
        episode_buffer = dataset.create_episode_buffer()
        episode_buffers.append(episode_buffer)
    
    # 샘플 데이터 생성 (실제로는 IsaacLab에서 수집된 데이터를 사용)
    num_frames = 100
    
    print(f"{num_envs}개 환경에서 {num_frames} 프레임씩 데이터 수집 중...")
    
    for frame_idx in range(num_frames):
        # 각 환경별로 데이터 생성
        for env_idx in range(num_envs):
            # 환경별로 다른 시드로 랜덤 데이터 생성 (실제로는 각 환경의 실제 데이터)
            np.random.seed(env_idx * 1000 + frame_idx)
            
            # 샘플 관찰 데이터
            obs_state = np.random.randn(12).astype(np.float32)
            
            # 샘플 이미지 데이터 (224x224 RGB)
            wrist_image = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
            
            # 샘플 액션 데이터
            action = np.random.randn(8).astype(np.float32)
            
            # 샘플 리워드 데이터
            reward = np.array([np.random.random()], dtype=np.float32)
            
            # 프레임 데이터 구성
            frame = {
                "observation.state": obs_state,
                "observation.images.wrist": wrist_image,
                "action": action,
                "next.reward": reward
            }
            
            # 각 환경의 에피소드 버퍼에 프레임 추가
            episode_buffers[env_idx].update(frame)
    
    # 각 환경별로 에피소드 저장
    for env_idx in range(num_envs):
        dataset.save_episode(episode_buffers[env_idx])
        print(f"환경 {env_idx}의 에피소드가 저장되었습니다.")

def add_sample_episode_single_env(dataset):
    """단일 환경에서 샘플 에피소드 데이터 추가 (기존 방식)"""
    
    if dataset is None:
        return
    
    # 에피소드 버퍼 생성
    episode_buffer = dataset.create_episode_buffer()
    
    # 샘플 데이터 생성 (실제로는 IsaacLab에서 수집된 데이터를 사용)
    num_frames = 100
    
    for frame_idx in range(num_frames):
        # 샘플 관찰 데이터
        obs_state = np.random.randn(12).astype(np.float32)
        
        # 샘플 이미지 데이터 (224x224 RGB)
        wrist_image = np.random.randint(0, 255, (3, 224, 224), dtype=np.uint8)
        
        # 샘플 액션 데이터
        action = np.random.randn(8).astype(np.float32)
        
        # 샘플 리워드 데이터
        reward = np.array([np.random.random()], dtype=np.float32)
        
        # 프레임 데이터 구성
        frame = {
            "observation.state": obs_state,
            "observation.images.wrist": wrist_image,
            "action": action,
            "next.reward": reward
        }
        
        # 프레임을 데이터셋에 추가
        dataset.add_frame(frame, "sample_task")
    
    # 에피소드 저장
    dataset.save_episode()
    print(f"샘플 에피소드 ({num_frames} 프레임)가 추가되었습니다.")

def load_and_use_dataset():
    """저장된 데이터셋 로드 및 사용"""
    
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        
        # 저장된 데이터셋 로드
        dataset = LeRobotDataset(
            repo_id="isaaclab_demo_dataset",
            root="./data/lerobot_demo"
        )
        
        print(f"데이터셋 로드 완료:")
        print(f"- 총 에피소드 수: {dataset.num_episodes}")
        print(f"- 총 프레임 수: {dataset.num_frames}")
        print(f"- FPS: {dataset.fps}")
        print(f"- 특성: {list(dataset.features.keys())}")
        
        # 데이터셋에서 샘플 가져오기
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n샘플 데이터:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  {key}: {type(value)}")
        
        return dataset
        
    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생: {e}")
        return None

def demonstrate_environment_separation():
    """환경별 분리 저장 방법을 보여주는 예제"""
    
    print("\n=== 환경별 에피소드 분리 저장 예제 ===")
    
    # 데이터셋 생성
    dataset = create_simple_lerobot_dataset()
    if dataset is None:
        return
    
    # 방법 1: 환경별로 따로 에피소드 저장 (권장)
    print("\n1. 환경별로 따로 에피소드 저장:")
    add_sample_episode_multi_env(dataset, num_envs=3)
    
    # 방법 2: 기존 방식 (모든 환경 데이터를 하나의 에피소드에 저장)
    print("\n2. 기존 방식 (단일 에피소드):")
    add_sample_episode_single_env(dataset)
    
    # 결과 확인
    print("\n3. 저장된 에피소드 확인:")
    load_and_use_dataset()

def main():
    """메인 함수"""
    print("=== LeRobot 데이터셋 예제 ===\n")
    
    # 환경별 분리 저장 방법 시연
    demonstrate_environment_separation()
    
    print("\n=== 완료 ===")
    print("데이터셋이 './data/lerobot_demo'에 저장되었습니다.")

if __name__ == "__main__":
    main() 