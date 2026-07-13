# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unsupported legacy copy of the cabinet asset inspection script.

Use run_all_cabinet.py for the maintained flow. This file is kept only as a
historical reference during the refactor.

This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

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

import omni.isaac.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.utils import assets as asset_utils
from isaaclab.sim import build_simulation_context
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext
from omni.isaac.debug_draw import _debug_draw
##
# Pre-defined configs
##
from isaaclab_assets import CARTPOLE_CFG  # isort:skip
import typing
from pxr import Usd, UsdGeom, Gf
import omni.isaac.core.utils.bounds as bounds_utils 
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.usd import get_context
import os
import collections
from math import inf
def grep_path():
    base_path = "Assets/drawer/train"
    mobility_pathes = []
    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)
        mobility_pathes.append(os.path.join(path, "mobility.usd"))
    return mobility_pathes
def env_spacing(mobility_path):
    spacing = 3.0
    max_per_row = int(len(mobility_path)**0.5)
    origins = []
    for i in range(len(mobility_path)):
        origins.append([-40+i//max_per_row*spacing, -40+i%max_per_row*spacing, 1.0])
    return origins
def get_points_by_link(part_dict:dict):
    part_points_by_link = collections.defaultdict(list)
    for key,value in part_dict.items():
        part_points_by_link[key] = [[inf, inf, inf],[-inf, -inf, -inf]]
        for prim in value:
            prim_path = prim.GetPath().pathString
            bbox_cache = bounds_utils.create_bbox_cache()
            bbox = bounds_utils.compute_combined_aabb(bbox_cache, [prim_path])
            min_corner = bbox[:3]
            max_corner = bbox[3:]
            part_points_by_link[key][0] = np.minimum(part_points_by_link[key][0], min_corner)
            part_points_by_link[key][1] = np.maximum(part_points_by_link[key][1], max_corner)
    return part_points_by_link
def draw_points(part_points_by_link:dict):
    draw = _debug_draw.acquire_debug_draw_interface()
    colors = []
    points = []
    for key,value in part_points_by_link.items():
        min_corner = value[0]
        max_corner = value[1]
        color = (random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), 1)
        points.append((max_corner[0]+0.05,min_corner[1],min_corner[2]))
        points.append((max_corner[0]+0.05,min_corner[1],max_corner[2]))
        points.append((max_corner[0]+0.05,max_corner[1],min_corner[2]))
        points.append((max_corner[0]+0.05,max_corner[1],max_corner[2]))
        for i in range(4):
            colors.append(color)
    sizes = [10]*len(points)
    draw.draw_points(points,colors,sizes)
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
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,

            enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
            ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                ".*": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=[".*" ],
                effort_limit=1.0,
                velocity_limit=100.0,
                stiffness=0.5,
                damping=0.5,
                friction=0.1,
            ),
        },
        )
        cabinet_cfgs.append(cabinet_cfg)
    scene_entities = dict()
    for i, cabinet_cfg in enumerate(cabinet_cfgs):
        cabinet = Articulation(cfg=cabinet_cfg)
        scene_entities[f"cabinet_{i+1}"] = cabinet
    return scene_entities, origins

def sample_point(handle_points:list,part_points:list, bounds :np.ndarray=np.array([0.00,0.0])):
    x = part_points[1][0]+0.05
    min_part = part_points[0][1:]
    max_part = part_points[1][1:]
    handle_min, handle_max = handle_points[0][1:]-bounds,handle_points[1][1:]+bounds
    while True:
        yz_sample = np.random.uniform(min_part,max_part)
        in_handle = (handle_min <= yz_sample).all() and (yz_sample <= handle_max).all()
        if not in_handle :
            return np.concatenate(([x],yz_sample))
def calculate_x_rotation_quaternion(p, q):
    """
    x=0 평면 위의 두 점 p, q를 잇는 벡터의 각도만큼
    x축을 기준으로 회전하는 쿼터니언(w, x, y, z)을 계산합니다.
    """

    # p에서 q로 가는 2D 벡터 (y, z 성분) 계산
    vector = np.array([q[1] - p[1], q[2] - p[2]])

    # 벡터가 0이면 회전이 필요 없으므로 단위 쿼터니언 반환
    if np.linalg.norm(vector) == 0:
        return np.array([1.0, 0.0, 0.0, 0.0])

    # y축(양수)을 기준으로 벡터의 각도(theta)를 계산
    # arctan2(z, y)를 사용하여 올바른 사분면의 각도를 구함
    theta = np.arctan2(vector[1], vector[0])

    # x축을 기준으로 theta만큼 회전하는 쿼터니언 계산
    half_angle = theta / 2.0
    
    w = np.cos(half_angle)
    x = np.sin(half_angle) # 회전축이 x축(1, 0, 0)이므로 x 성분만 값을 가짐
    y = 0.0
    z = 0.0
    
    return np.array([w, x, y, z])
def quaternion_multiply(q1, q2):
    """두 쿼터니언을 곱합니다 (q1 * q2)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])
def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # robot = entities["cabinet_1"]
    # Define simulation stepping
    robot_states = {}
    robot_states = {}
    for keys, robot in entities.items(): # <-- 이 초기화 루프가 while 루프보다 먼저 실행됩니다.
        robot_states[keys] = {
            "joint_targets": robot.data.default_joint_pos.clone(),
            "joint_direction": 1,  # 1: 증가, -1: 감소
            "max_joint_value": 0.5,
            "move_speed": 0.001,
        }
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    context = get_context()
    stage = context.get_stage()

    for keys, robot in entities.items():
        num_env = int(keys.split('_')[-1])
        start_prim_path = f"/World/Origin{num_env}/Cabinet"
        body_names = [start_prim_path + "/" + names for names in robot.data.body_names if "link" in names]

        handles_by_link = collections.defaultdict(list)
        drawers_by_link = collections.defaultdict(list)
        cabinet_by_link = collections.defaultdict(list)
        for body_name in body_names:
            start_prim = stage.GetPrimAtPath(body_name+'/visuals')
            for prim in start_prim.GetChildren():
                if "handle" in prim.GetName():
                    handles_by_link[body_name].append(prim)
                elif "drawer_front" in prim.GetName():
                    drawers_by_link[body_name].append(prim)
                elif "cabinet_door_surface" in prim.GetName():
                    cabinet_by_link[body_name].append(prim)
        drawer_points_by_link = get_points_by_link(drawers_by_link)
        handle_points_by_link = get_points_by_link(handles_by_link)
        cabinet_points_by_link = get_points_by_link(cabinet_by_link)
        draw_points(drawer_points_by_link)
        draw_points(handle_points_by_link)
        draw_points(cabinet_points_by_link)
        right_quarternion = np.array([0.5,0.5,-0.5,-0.5])
        guide_points_by_link = dict()
        for handle in handle_points_by_link.keys():
            handle_mid = handle_points_by_link[handle][2]
            
            if handle in drawer_points_by_link:
                guide_points_by_link[handle] = sampled_point = sample_point(handle_points_by_link[handle],drawer_points_by_link[handle])
            elif handle in cabinet_points_by_link:
                guide_points_by_link[handle] = sampled_point = sample_point(handle_points_by_link[handle],cabinet_points_by_link[handle])
            else:
                raise ValueError(f"handle {handle} is not in drawer or cabinet")
    joint_targets = robot.data.default_joint_pos.clone() # 초기 목표는 현재 조인트 위치
    max_joint_value = 0.5 # 각 조인트가 최대로 움직일 목표 값 (0.0에서 이 값까지 움직이게)
    move_speed = 0.0005 # 한 스텝당 목표 위치가 증가하는 양 (조절하여 속도 조절)
    joint_direction = 1 # 1: 증가, -1: 감소 (양쪽으로 왕복하게 만들고 싶다면)

    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            count = 0
            for keys, robot in entities.items():
                origin = origins[int(keys.split('_')[-1])-1]
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origin
                robot.write_root_state_to_sim(root_state)
                
                # 각 로봇의 상태 초기화
                robot_states[keys]["joint_direction"] = 1 
                robot_states[keys]["joint_targets"] = robot.data.default_joint_pos.clone()
                joint_vel = torch.zeros_like(robot.data.default_joint_vel) 
                robot.write_joint_state_to_sim(robot_states[keys]["joint_targets"], joint_vel) 
            
                robot.reset()
                print(f"[INFO]: Resetting robot {keys} state...")
                # 리셋 후에는 다시 증가하는 방향으로 시작
                

                # --- 조인트 목표 위치 업데이트 ---
                # 모든 조인트에 대해 목표 위치를 점진적으로 증가
                # (모든 조인트가 동시에 같은 속도로 같은 방향으로 움직인다고 가정)
                
                # 현재 조인트 목표에 move_speed를 더합니다.
        for keys, robot in entities.items():
            current_state = robot_states[keys]
            
            # 현재 로봇의 조인트 목표 업데이트
            current_state["joint_targets"] += current_state["joint_direction"] * current_state["move_speed"]
            
            # 목표가 최대치를 넘거나 최소치(0) 아래로 내려가지 않도록 클램핑
            current_state["joint_targets"] = torch.clamp(
                current_state["joint_targets"], 0.0, current_state["max_joint_value"]
            )

            # 만약 목표가 max_joint_value에 도달하면 방향을 반대로 바꿉니다 (왕복 운동)
            if torch.all(current_state["joint_targets"] >= current_state["max_joint_value"] - 1e-5) and current_state["joint_direction"] == 1:
                current_state["joint_direction"] = -1
            elif torch.all(current_state["joint_targets"] <= 1e-5) and current_state["joint_direction"] == -1:
                current_state["joint_direction"] = 1

            # 각 로봇의 조인트 위치 목표 설정
            robot.set_joint_position_target(current_state["joint_targets"])
            
            # --- write data to sim ---
            robot.write_data_to_sim()
        # Perform step (이전과 동일)
        sim.step()
        # Increment counter (이전과 동일)
        count += 1
        # Update buffers (이전과 동일)
        for robot in entities.values():
            robot.update(sim_dt)

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
