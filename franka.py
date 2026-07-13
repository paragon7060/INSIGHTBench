# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka Emika robots.

The following configurations are available:

* :obj:`FRANKA_PANDA_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`FRANKA_PANDA_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""
import os

# Isaac Lab 프로젝트 루트 경로를 환경 변수에서 가져옵니다.
ISAACLAB_ROOT_DIR = os.environ.get("ISAACLAB_PATH", ".") # 환경 변수가 없으면 현재 폴더를 기준으로 함

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##
FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        #TODO: usd_path change
        usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/FrankaEmika/panda_instanceable.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -1.10,
            "panda_joint3": 0.0,
            "panda_joint4": -2.30,
            "panda_joint5": 0.0,
            "panda_joint6": 2.550,
            "panda_joint7": 0.0,
            "panda_finger_joint.*": 0.04,
        },
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=60.0, #87.0,
            velocity_limit=1.450,#2.175,
            stiffness=1200.0,
            damping=100.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=10.0, #12.0,
            velocity_limit=1.605, #2.61,
            stiffness=1200.0,
            damping=100.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=30.0,
            velocity_limit=0.5, 
            stiffness=0, #2e3,
            damping=0, #1e2,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 80.0#400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 110.0#80.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 80.0#400.0
FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 110.0#80.0
"""Configuration of Franka Emika Panda robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""
