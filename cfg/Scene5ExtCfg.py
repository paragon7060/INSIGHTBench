import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

##################################################################################
######################## For Extended Bottle Scene ###############################
##################################################################################
import random
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from scene5Cfg import BottleSceneCfg, RewTerm, joint_pos, SceneEntityCfg
from helper import sample_usd_dir_from_dir, load_bbox_and_compute_center_and_zmax

# 기본 클래스
@configclass
class SceneBottleNoguideCfg(BottleSceneCfg):
    bottle: ArticulationCfg = MISSING

@configclass
class SceneBottleExtCfg(BottleSceneCfg):
    bottle: ArticulationCfg = MISSING # placeholder for type check
    guide: AssetBaseCfg = MISSING

@configclass
class Reward5ExtCfg:
    close_bottle_rew: RewTerm = MISSING  # placeholder


def make_bottle_scene_and_reward_cfg(
    usd_path: str,
    guide_path: str,
    guide_scale: tuple,
    asset_init_pos: tuple,
    guide_init_pos: tuple,
    close_mode: bool,
    no_guide: bool
):
    # compute initial joint positions based on close_mode
    if close_mode:
        joint_1_init_pos = 0.005
        joint_2_init_pos = 6.275
    else:
        joint_1_init_pos = 0.0
        joint_2_init_pos = 0.0

    if no_guide:
        scene_cfg = SceneBottleNoguideCfg(
            bottle=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    activate_contact_sensors=True,
                    scale=(0.8, 0.8, 0.8),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=asset_init_pos,
                    rot=(0.707, 0.0, 0.707, 0.0),
                    joint_pos={
                        "joint_1": joint_1_init_pos,
                        "joint_2": joint_2_init_pos,
                    },
                    joint_vel={"joint_1": 0.0, "joint_2": 0.0},
                ),
                actuators={
                    "prismatic": ImplicitActuatorCfg(
                        joint_names_expr=["joint_1"],
                        effort_limit=200.0,
                        velocity_limit=1.0,
                        stiffness=1200.0,
                        damping=0.0,
                        friction=100.0,
                    ),
                    "cap_revolute": ImplicitActuatorCfg(
                        joint_names_expr=["joint_2"],
                        effort_limit=87.0,
                        velocity_limit=1.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=0.1,
                    ),
                },
            ),
        )
    else:
    # 동적으로 속성 채워 넣기
        scene_cfg = SceneBottleExtCfg(
            bottle=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    activate_contact_sensors=True,
                    scale=(0.8, 0.8, 0.8),
                ),
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=asset_init_pos,
                    rot=(0.707, 0.0, 0.707, 0.0),
                    joint_pos={
                        "joint_1": joint_1_init_pos,
                        "joint_2": joint_2_init_pos,
                    },
                    joint_vel={"joint_1": 0.0, "joint_2": 0.0},
                ),
                actuators={
                    "prismatic": ImplicitActuatorCfg(
                        joint_names_expr=["joint_1"],
                        effort_limit=200.0,
                        velocity_limit=1.0,
                        stiffness=1200.0,
                        damping=0.0,
                        friction=100.0,
                    ),
                    "cap_revolute": ImplicitActuatorCfg(
                        joint_names_expr=["joint_2"],
                        effort_limit=87.0,
                        velocity_limit=1.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=0.1,
                    ),
                },
            ),
            guide=AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Bottle/link_2/guide",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=guide_init_pos,
                    rot=(0.707,0,0,0.707),
                    ),
                spawn=sim_utils.UsdFileCfg(
                    usd_path=guide_path,
                    scale=guide_scale,
                ),
            ),
        )
    scene_cfg.replicate_physics = False
    reward_cfg = Reward5ExtCfg(
        close_bottle_rew=RewTerm(
            func=joint_pos,
            weight=1,
            params={
                "asset_cfg": SceneEntityCfg("bottle", joint_names=["joint_1"]),
                "reverse": close_mode,
                "offset": 0.005 if close_mode else 0,
            },
        )
    )
    return scene_cfg, reward_cfg

