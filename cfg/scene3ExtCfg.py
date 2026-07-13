import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

##################################################################################
######################## For Extended Bottle Scene ###############################
##################################################################################

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from scene3Cfg import DoorSceneCfg, RewTerm, joint_pos, SceneEntityCfg
from helper import sample_usd_dir_from_dir, load_bbox_and_compute_center_and_zmax

# 기본 클래스
@configclass
class SceneDoorNoguideCfg(DoorSceneCfg):
    door: ArticulationCfg = MISSING # placeholder for type check

@configclass
class SceneDoorExtCfg(DoorSceneCfg):
    door: ArticulationCfg = MISSING # placeholder for type check
    guide: AssetBaseCfg = MISSING

@configclass
class Reward5ExtCfg:
    pull_door_rew: RewTerm = MISSING  # placeholder


def make_door_scene_cfg(
    usd_path: str,
    guide_path: str,
    asset_init_pos: tuple,
    guide_init_pos: tuple,
    door_scale: tuple,
    no_guide: bool = False
):
    # 동적으로 속성 채워 넣기
    if no_guide:
        scene_cfg = SceneDoorNoguideCfg(
            door=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Door",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    activate_contact_sensors=True,
                    scale=door_scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        max_depenetration_velocity=5.0,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=asset_init_pos,
                    rot=(0.707, 0.0, 0.707, 0.0),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "door": ImplicitActuatorCfg(
                        joint_names_expr=["joint_1"],
                        effort_limit_sim=87.0,
                        velocity_limit_sim=10.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=10.0,
                    ),
                    "lever": ImplicitActuatorCfg(
                        joint_names_expr=["joint_2"],
                        effort_limit_sim=87.0,
                        velocity_limit_sim=10.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=1.0,
                    )
                }
            ),
        )
    else:
        scene_cfg = SceneDoorExtCfg(
            door=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Door",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    activate_contact_sensors=True,
                    scale=door_scale,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        max_depenetration_velocity=5.0,
                    ),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=asset_init_pos,
                    rot=(0.707, 0.0, 0.707, 0.0),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "door": ImplicitActuatorCfg(
                        joint_names_expr=["joint_1"],
                        effort_limit_sim=87.0,
                        velocity_limit_sim=10.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=1.0,
                    ),
                    "lever": ImplicitActuatorCfg(
                        joint_names_expr=["joint_2"],
                        effort_limit_sim=87.0,
                        velocity_limit_sim=10.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=1.0,
                    )
                }
            ),
            guide=AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Door/link_1/guide",
                init_state=AssetBaseCfg.InitialStateCfg(
                    pos=guide_init_pos,
                    rot=(0.5, 0.5, -0.5, 0.5)
                    ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=guide_path,
                    scale=(1.3, 1.3, 1.3),
                ),
            ),
        )
    scene_cfg.replicate_physics = False
    reward_cfg = Reward5ExtCfg(
        pull_door_rew=RewTerm(
            func=joint_pos,
            weight=1,
            params={
                "asset_cfg": SceneEntityCfg("door", joint_names=["joint_1"]),
                # "reverse": close_mode,
                # "offset": 0.005 if close_mode else 0,
            },
        )
    )
    return scene_cfg, reward_cfg
