"""Launch Isaac Sim Simulator first."""
import random
import argparse
import json
import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext
from isaacsim.util.debug_draw import _debug_draw

"""Standalone script: loads bottle USDs, computes and visualizes bounding boxes per environment origin."""
import os
import random
import argparse
import json
import numpy as np
import torch

import isaacsim.core.utils.bounds as bounds_utils 
from omni.usd import get_context

def grep_path():
    # base_path = "./Assets/PartManip/drawer/train"
    base_path = "./Assets/TestSuite/cabinet_suite"
    mobility_pathes = []
    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)
        mobility_pathes.append(os.path.join(path, "mobility_new.usd"))
    return mobility_pathes
def env_spacing(mobility_path):
    spacing = 3.0
    max_per_row = int(len(mobility_path)**0.5)
    origins = []
    for i in range(len(mobility_path)):
        origins.append([-5+i//max_per_row*spacing, -5+i%max_per_row*spacing, 1.0])
    return origins
def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    mobility_pathes = grep_path()
    origins = env_spacing(mobility_pathes)
    cabinet_cfgs = []
    # Origin 1
    # prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # # Origin 2
    # prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    for i, (mobility_path, origin) in enumerate(zip(mobility_pathes,origins)):
        prim_utils.create_prim(f"/World/Origin{i+1}", "Xform", translation=origin)
        cabinet_cfg = ArticulationCfg(
            prim_path=f"/World/Origin{i+1}/Cabinet",
            spawn=sim_utils.UsdFileCfg(
                usd_path=mobility_path,
                activate_contact_sensors=False,
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
                joint_pos={".*": 0.0,},
            ),
            actuators={
                "drawers": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    effort_limit_sim=87.0,
                    velocity_limit_sim=100.0,
                    stiffness=0.0,
                    damping=0.0,
                    friction=1.0,
                ),
            }
        )
        cabinet_cfgs.append(cabinet_cfg)
    scene_entities = dict()
    for i, cabinet_cfg in enumerate(cabinet_cfgs):
        cabinet = Articulation(cfg=cabinet_cfg)
        scene_entities[f"cabinet_{i+1}"] = cabinet
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    robot = entities["cabinet_1"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    # draw = _debug_draw.acquire_debug_draw_interface()
    context = get_context()
    stage = context.get_stage()
    all_paths = [prim.GetPath() for prim in stage.Traverse()]
    handle_only_paths = [stage.GetPrimAtPath(path) for path in all_paths if "handle" in str(path).split("/")[-1]]
    str_path = [i.GetPath().pathString for i in handle_only_paths]
    for path in str_path:
        bbox_cache = bounds_utils.create_bbox_cache()
        bbox = bounds_utils.compute_combined_aabb(bbox_cache, [path])
        min_corner = bbox[:3]
        max_corner = bbox[3:]

    joint_targets = robot.data.default_joint_pos.clone() # 초기 목표는 현재 조인트 위치
    max_joint_value = 0.5 # 각 조인트가 최대로 움직일 목표 값 (0.0에서 이 값까지 움직이게)
    move_speed = 0.001 # 한 스텝당 목표 위치가 증가하는 양 (조절하여 속도 조절)
    joint_direction = 1 # 1: 증가, -1: 감소 (양쪽으로 왕복하게 만들고 싶다면)

    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            count = 0
            # root_state = robot.data.default_root_state.clone()
            # root_state[:, :3] += origins
            # robot.write_root_state_to_sim(root_state)
            
            # # 리셋 시 조인트 목표도 초기화
            # joint_targets = robot.data.default_joint_pos.clone()
            # # joint_vel은 필요에 따라 0으로 설정
            # joint_vel = torch.zeros_like(robot.data.default_joint_vel) 
            # robot.write_joint_state_to_sim(joint_targets, joint_vel) 
            
            # robot.reset()
            # print("[INFO]: Resetting robot state...")
            
            # 리셋 후에는 다시 증가하는 방향으로 시작
            joint_direction = 1 

        # --- 조인트 목표 위치 업데이트 ---
        # 모든 조인트에 대해 목표 위치를 점진적으로 증가
        # (모든 조인트가 동시에 같은 속도로 같은 방향으로 움직인다고 가정)
        
        # 현재 조인트 목표에 move_speed를 더합니다.
        # joint_targets += joint_direction * move_speed
        
        # # 목표가 최대치를 넘거나 최소치(0) 아래로 내려가지 않도록 클램핑
        # joint_targets = torch.clamp(joint_targets, 0.0, max_joint_value)

        # 만약 목표가 max_joint_value에 도달하면 방향을 반대로 바꿉니다 (왕복 운동)
        # (선택 사항: 한 방향으로만 움직이게 하려면 이 부분 제거)
        # if torch.all(joint_targets >= max_joint_value - 1e-5) and joint_direction == 1:
        #     joint_direction = -1
        # elif torch.all(joint_targets <= 1e-5) and joint_direction == -1:
        #     joint_direction = 1

        # # --- 조인트 위치 목표 설정 ---
        # # robot.set_joint_position_target()을 사용하여 PID 컨트롤러가 목표 위치로 조인트를 이동시킵니다.
        # robot.set_joint_position_target(joint_targets)
        
        # -- write data to sim (이전과 동일)
        # robot.write_data_to_sim()
        # Perform step (이전과 동일)
        sim.step()
        # Increment counter (이전과 동일)
        count += 1
        # Update buffers (이전과 동일)
        # robot.update(sim_dt)

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
