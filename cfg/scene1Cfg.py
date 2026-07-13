# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Rest everything follows."""

##################################################################################
######################## For Cabinet Scene #######################################
##################################################################################

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

# Isaac Lab 프로젝트 루트 경로를 환경 변수에서 가져옵니다.
ISAACLAB_ROOT_DIR = os.environ.get("ISAACLAB_PATH", ".") # 환경 변수가 없으면 현재 폴더를 기준으로 함

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from cfg.event_cfg.events import make_fixed_joints, reset_guide_position_with_random_flip, reset_root_state_uniform_ori
from cfg.BaseCfg import BaseSceneCfg, ObsCfg, TerminationsCfg, ContinuousJointActionsCfg
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from franka import FRANKA_PANDA_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10) 

@configclass
class CabinetBaseSceneCfg(BaseSceneCfg):
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/sektion_cabinet_instanceable.usd",
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0, 0.6),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": 0.0,
                "door_right_joint": 0.0,
                "drawer_bottom_joint": 0.0,
                "drawer_top_joint": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=87.0,
                velocity_limit_sim=100.0,
                stiffness=0.0,
                damping=0.0,
                friction=1.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.2,
            ),
        },
    )

@configclass
class Scene1aCfg(CabinetBaseSceneCfg):
    guide_arrow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_arrow",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.674,-0.15,0.915), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/arrows/guide_arrow_physics.usd",
            scale=(0.02, 0.02, 0.02),
            )
    )
    guide_open = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_open",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.674,-0.17,0.87), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/texts/text_open_black_rigid.usda",
            scale=(0.02, 0.02, 0.02),
            )
    )

@configclass
class Scene1bCfg(CabinetBaseSceneCfg):
    guide_arrow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_arrow",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.674,-0.15,0.775), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/arrows/guide_arrow_physics.usd",
            scale=(0.02, 0.02, 0.02),
            )
    )
    guide_open = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_open",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.674,-0.17,0.73), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/texts/text_open_black_rigid.usda",
            scale=(0.02, 0.02, 0.02),
            )
    )

@configclass
class Scene1cCfg(CabinetBaseSceneCfg):
    guide_arrow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_arrow",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.611,0.1026,0.65022), rot=(0.5,-0.5,-0.5,0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/arrows/guide_arrow_physics.usd",
            scale=(0.02, 0.02, 0.02),
            )
    )
    guide_open = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_open",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.611,0.12014,0.68466), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/texts/text_open_black_rigid.usda",
            scale=(0.02, 0.02, 0.02),
            )
    )

@configclass
class Scene1dCfg(CabinetBaseSceneCfg):
    guide_arrow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_arrow",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.611,-0.0826,0.65022), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/arrows/guide_arrow_physics.usd",
            scale=(0.02, 0.02, 0.02),
            )
    )
    guide_open = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_open",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.611,-0.0826,0.68466), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/texts/text_open_black_rigid.usda",
            scale=(0.02, 0.02, 0.02),
            )
    )

@configclass
class Scene1eCfg(CabinetBaseSceneCfg):
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        # articulation_root_prim_path="/link_0",  # Cabinet의 루트 링크 지정
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/PartManip/drawer/train/StorageFurniture-40147-link_1-handle_1-joint_1-handlejoint_1/mobility_new.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        scale=(0.75, 0.75, 0.75),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.75, 0, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
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
    guide_arrow = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/guide_arrow",
        init_state= RigidObjectCfg.InitialStateCfg(pos=(0.674,-0.15,0.915), rot=(0.5,0.5,-0.5,-0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/arrows/guide_arrow_physics.usd",
            scale=(0.02, 0.02, 0.02),
            )
    )

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    null_command = mdp.NullCommandCfg()

##
# Action Cfg
##

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from interact.action_batch import CuroboInteractionAction

@configclass
class CuroboInteractActionCfg(ActionTermCfg):
    """Configuration for inverse differential kinematics action term.

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        """The offset pose from parent frame to child frame.

        On many robots, end-effector frames are fictitious frames that do not have a corresponding
        rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
        For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
        "panda_hand" frame.
        """

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
        """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

    class_type: type[ActionTerm] = CuroboInteractionAction

    joint_names: list[str] = MISSING
    """List of joint names or regex expressions that the action will be mapped to."""
    body_name: str = MISSING
    """Name of the body or frame for which IK is performed."""
    body_offset: OffsetCfg | None = None
    """Offset of target frame w.r.t. to the body frame. Defaults to None, in which case no offset is applied."""
    scale: float | tuple[float, ...] = 1.0


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTermCfg = CuroboInteractActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            scale=1,
            body_offset=CuroboInteractActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
    # gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class EventCabinetCfg:
    # """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=[".*"]),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    start_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="startup",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_root_state = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="reset",
        params={
            "pose_range": {"x":(-0.11, 0.16), "y":(-0.3, 0.3), "z":(-0.12, 0.12)},
            # "pose_range": {"x":(-0.06, 0.06), "y":(-0.2, 0.2), "z":(-0.1, 0.1)},
            # "pose_range": {},
            "velocity_range":{},
            "asset_cfg": SceneEntityCfg("cabinet"),
        },
    )
 
    randomize_root_state_startup = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="startup",
        params={
            # "pose_range": {"x":(-0.11, 0.16), "y":(-0.3, 0.3), "z":(-0.12, 0.12)},
            "pose_range": {"x":(-0.06, 0.06), "y":(-0.2, 0.2), "z":(-0.1, 0.1)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("cabinet"),
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("cabinet", body_names=".*"),
            "mass_distribution_params": (0.5, 1.0),
            "operation": "abs",
            "distribution": "uniform"
        }
    )

@configclass
class EventCabinetColorCfg(EventCabinetCfg):
    randomize_cabinet_color_startup = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=[".*"]),
            "event_name": "randomize_cabinet_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )

    randomize_cabinet_color_reset = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=[".*"]),
            "event_name": "randomize_cabinet_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )

@configclass
class EventCabinetNoposCfg:

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=[".*"]),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    start_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="startup",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("cabinet", body_names=".*"),
            "mass_distribution_params": (0.2, 1.0),
            "operation": "abs",
            "distribution": "uniform"
        }
    )

@configclass
class EventCabinetNoposColorCfg(EventCabinetNoposCfg):
    randomize_cabinet_color_startup = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=[".*"]),
            "event_name": "randomize_cabinet_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )

    randomize_cabinet_color_reset = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=[".*"]),
            "event_name": "randomize_cabinet_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )


@configclass
class EventCabinetTestCfg():
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # check and backup !!TODO
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 0.35),
            "dynamic_friction_range": (0.1, 0.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_physics_material_finger = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_leftfinger","panda_rightfinger"]),
            "static_friction_range": (20, 20),
            "dynamic_friction_range": (15, 15),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    cabinet_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet", body_names=[".*"]),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    start_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="startup",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("cabinet", body_names=".*"),
            "mass_distribution_params": (0.2, 1.0),
            "operation": "abs",
            "distribution": "uniform"
        }
    )

@configclass
class Event1aCfg(EventCabinetCfg):
    randomize_guide_position_startup = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="startup",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.12, 0.03), "z":(-0.03, 0.03)},
            "do_random_flip": True,
        }
    )
    randomize_guide_position = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="reset",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.12, 0.03), "z":(-0.03, 0.03)},
            "do_random_flip": True,
        }
    )

@configclass
class Event1bCfg(EventCabinetCfg):
    randomize_guide_position_startup = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="startup",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.12, 0.03), "z":(-0.03, 0.03)},
            "do_random_flip": True,
        }
    )
    randomize_guide_position = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="reset",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.12, 0.03), "z":(-0.03, 0.03)},
            "do_random_flip": True,
        }
    )

@configclass
class Event1cCfg(EventCabinetCfg):
    randomize_guide_position_startup = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="startup",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.02, 0.1), "z":(-0.03, 0.03)},
            "do_random_flip": False,
        }
    )
    randomize_guide_position = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="reset",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.02, 0.1), "z":(-0.03, 0.03)},
            "do_random_flip": False,
        }
    )

@configclass
class Event1dCfg(EventCabinetCfg):
    randomize_guide_position_startup = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="startup",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.1, 0.02), "z":(-0.03, 0.03)},
            "do_random_flip": False,
        }
    )
    randomize_guide_position = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="reset",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.1, 0.02), "z":(-0.03, 0.03)},
            "do_random_flip": False,
        }
    )
@configclass
class Event1eCfg(EventCabinetCfg):
    randomize_guide_position_startup = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="startup",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.1, 0.02), "z":(-0.03, 0.03)},
            "do_random_flip": False,
        }
    )
    randomize_guide_position = EventTerm(
        func=reset_guide_position_with_random_flip,
        mode="reset",
        params={
            "pose_range": {"x":(-0.0, 0.0), "y":(-0.1, 0.02), "z":(-0.03, 0.03)},
            "do_random_flip": False,
        }
    )

from cfg.reward_cfg.reward import joint_pos_success, joint_pos

@configclass
class Reward1aCfg:
    """Reward terms for the MDP."""
    # 3. Open the drawer
    open_drawer_rew_1a = RewTerm(
        func=joint_pos,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"]),
            },
    )

@configclass
class Reward1bCfg:
    """Reward terms for the MDP."""
    # 3. Open the drawer
    open_drawer_rew_1b = RewTerm(
        func=joint_pos,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_bottom_joint"]),
            },
    )

@configclass
class Reward1cCfg:
    """Reward terms for the MDP."""
    # 3. Open the drawer
    open_drawer_rew_1c = RewTerm(
        func=joint_pos,
        weight=-1,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["door_left_joint"]),
            # "success_js_value": 0.1
            },
    )

@configclass
class Reward1dCfg:
    """Reward terms for the MDP."""
    # 3. Open the drawer
    open_drawer_rew_1d = RewTerm(
        func=joint_pos,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["door_right_joint"]),
            },
    )
@configclass
class Reward1eCfg:
    """Reward terms for the MDP."""
    # 3. Open the drawer
    open_drawer_rew_1e = RewTerm(
        func=joint_pos,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("cabinet", joint_names=["joint_1"]),
            },
    )

@configclass
class CabinetTopDrawerSkillEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1aCfg = Scene1aCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1aCfg = Event1aCfg()
    rewards: Reward1aCfg = Reward1aCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 300
        self.episode_length_s = 20.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.env_spacing = 2.0

@configclass
class CabinetBottomDrawerSkillEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1bCfg = Scene1bCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1bCfg = Event1bCfg()
    rewards: Reward1bCfg = Reward1bCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 300
        self.episode_length_s = 20.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

@configclass
class CabinetLeftDoorSkillEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1cCfg = Scene1cCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1cCfg = Event1cCfg()
    rewards: Reward1cCfg = Reward1cCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 300
        self.episode_length_s = 20.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

@configclass
class CabinetRightDoorSkillEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1dCfg = Scene1dCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1dCfg = Event1dCfg()
    rewards: Reward1dCfg = Reward1dCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 300
        self.episode_length_s = 20.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625

@configclass
class CabinetTopDrawerContinuousEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1aCfg = Scene1aCfg()
    actions: ContinuousJointActionsCfg = ContinuousJointActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1aCfg = Event1aCfg()
    rewards: Reward1aCfg = Reward1aCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 1
        # step_dt = 5 & self.max_episode_length = episode_length_s / step_dt
        self.episode_length_s = 20 # 5 * skill_length
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.scene.robot.actuators["panda_hand"].stiffness = 2e3 # for inference action models using gripper pos control
    
@configclass
class CabinetBottomDrawerContinuousEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1bCfg = Scene1bCfg()
    actions: ContinuousJointActionsCfg = ContinuousJointActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1bCfg = Event1bCfg()
    rewards: Reward1bCfg = Reward1bCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 1
        # step_dt = 5 & self.max_episode_length = episode_length_s / step_dt
        self.episode_length_s = 20 # 5 * skill_length
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.scene.robot.actuators["panda_hand"].stiffness = 2e3 # for inference action models using gripper pos control
    
@configclass
class CabinetLeftDoorContinuousEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1cCfg = Scene1cCfg()
    actions: ContinuousJointActionsCfg = ContinuousJointActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1cCfg = Event1cCfg()
    rewards: Reward1cCfg = Reward1cCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 1
        # step_dt = 5 & self.max_episode_length = episode_length_s / step_dt
        self.episode_length_s = 20 # 5 * skill_length
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.scene.robot.actuators["panda_hand"].stiffness = 2e3 # for inference action models using gripper pos control
   
@configclass
class CabinetRightDoorContinuousEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene1dCfg = Scene1dCfg()
    actions: ContinuousJointActionsCfg = ContinuousJointActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event1dCfg = Event1dCfg()
    rewards: Reward1dCfg = Reward1dCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 1
        # step_dt = 5 & self.max_episode_length = episode_length_s / step_dt
        self.episode_length_s = 20 # 5 * skill_length
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 60  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 2**24
        self.scene.robot.actuators["panda_hand"].stiffness = 2e3 # for inference action models using gripper pos control
   