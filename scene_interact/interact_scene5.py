import sys
import os

this_file = os.path.abspath(__file__)
current_dir = os.path.dirname(this_file)
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import matplotlib.pyplot as plt
import os
import imageio
import gymnasium as gym

from custom_lab.envs.manager_based_rl_step_env import ManagerBasedRLStepEnv
from cfg.scene5Cfg import BottleSkillSqueezeCfg, BottleSkillOpenCfg, BottleSkillOpenReversedCfg, BottleSkillCloseCfg, BottleSkillCloseReversedCfg
# 이미지 저장 경로 설정
save_dir = "./output"
os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성

def main():
    env_cfg = BottleSkillSqueezeCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLStepEnv(cfg=env_cfg)
    count = 0

    while simulation_app.is_running():
        # 사용자가 직접 8개의 파라미터 값을 입력하도록 요청
        print("\n[INPUT] 8개의 action parameter 값을 입력하세요 (쉼표로 구분):")
        user_input = input("예시: 0,0.45,0,1.0,-0.5,-0.5,-0.5,-0.5\n입력: ").strip()
        if user_input == "reset":
            env.reset()

        try:
            # 입력된 문자열을 쉼표 단위로 분리 후 float 리스트로 변환
            values = [float(val.strip()) for val in user_input.split(',')]
            if len(values) != 8:
                raise ValueError("8개의 파라미터가 필요합니다.")
        except Exception as e:
            print(f"[ERROR] 올바른 입력이 아닙니다: {e}")
            # 잘못된 입력이면 기본 파라미터를 사용합니다.
            values = [0, 0.5, 0.0, 0.7, -0.5, -0.5, -0.5, -0.5]

        # action_parameter 를 해당 값들로 설정
        action_parameter = torch.tensor(values).unsqueeze(0)  # env.action_manager.action와 동일한 shape 맞추기

        obs, _, _, _, _  = env.step(action_parameter)
            
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

'''
Scene5.
1, 0.5, 0, 0.490, 0, 1, 0, 0
3,1.2,0,0,0,0,0,0
'''