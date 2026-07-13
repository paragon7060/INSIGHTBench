import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

##################################################################################
######################## For Extended Cabinet Scene ##############################
##################################################################################
import random
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from cfg.BaseCfg import BaseSceneCfg, FrontCameraBaseSceneCfg
from scene1Cfg import RewTerm, joint_pos, SceneEntityCfg
from helper import sample_usd_dir_from_dir, load_bbox_and_compute_center_and_zmax

from cfg.reward_cfg.reward import joint_pos_success, joint_pos_for_joint_id

@configclass
class SceneCabinetNoguideCfg(FrontCameraBaseSceneCfg):
    cabinet: ArticulationCfg = MISSING 


# 기본 클래스
@configclass
class SceneCabinetExtCfg(FrontCameraBaseSceneCfg):
    cabinet: ArticulationCfg = MISSING # placeholder for type check
    guide_arrow: AssetBaseCfg = MISSING

@configclass
class RewardCabinetExtCfg:
    """Reward terms for the cabinet scene."""
    joint_reward: RewTerm = MISSING

def make_cabinet_scene_and_reward_cfg(
    usd_path: str,
    asset_init_pos: tuple,
    cabinet_scale: tuple,
    joint_id: str,
    link_id: str,
    guide_init_pos: tuple,
    guide_init_quat: tuple,
    joint_type: str,
    no_guide: bool,
):
    if no_guide:
        scene_cfg = SceneCabinetNoguideCfg(
            cabinet=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    activate_contact_sensors=True,
                    scale=cabinet_scale,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=0,
                        sleep_threshold=0.005,
                        stabilization_threshold=0.001,
                    ),
                ),

                init_state=ArticulationCfg.InitialStateCfg(
                    pos=asset_init_pos,
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "drawers": ImplicitActuatorCfg(
                        joint_names_expr=[".*" ],
                        effort_limit=87.0,
                        velocity_limit=10.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=1.0,
                    ),
                },
            ),
        )
    else:
        scene_cfg = SceneCabinetExtCfg(
            cabinet=ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=usd_path,
                    activate_contact_sensors=True,
                    scale=cabinet_scale,
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False,
                        solver_position_iteration_count=4,
                        solver_velocity_iteration_count=0,
                        sleep_threshold=0.005,
                        stabilization_threshold=0.001,
                    ),
                ),

                init_state=ArticulationCfg.InitialStateCfg(
                    pos=asset_init_pos,
                    rot=(1.0, 0.0, 0.0, 0.0),
                    joint_pos={".*": 0.0},
                    joint_vel={".*": 0.0},
                ),
                actuators={
                    "drawers": ImplicitActuatorCfg(
                        joint_names_expr=[".*" ],
                        effort_limit=87.0,
                        velocity_limit=10.0,
                        stiffness=0.0,
                        damping=0.0,
                        friction=1.0,
                    ),
                },
            ),
            guide_arrow=AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Cabinet/"+f"{link_id}/guide_arrow",
                init_state= AssetBaseCfg.InitialStateCfg(pos=guide_init_pos, rot=guide_init_quat),
                spawn=sim_utils.UsdFileCfg(
                    usd_path="Assets/guides/arrows/guide_arrow_physics.usd",
                    scale=(0.02, 0.02, 0.02),
                    )
                )
        )
    scene_cfg.replicate_physics = False
    weight_joint_type = 5 if joint_type == "prismatic" else 1
    rew_cfg = RewardCabinetExtCfg(
        joint_reward = RewTerm(
            func=joint_pos_for_joint_id,
            weight=weight_joint_type,
            params={
                "joint_id": joint_id,
            }
        )
    )
    return scene_cfg, rew_cfg
