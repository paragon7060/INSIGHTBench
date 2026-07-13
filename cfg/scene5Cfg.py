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
from isaaclab.sensors import FrameTransformerCfg, CameraCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.envs import mdp
from cfg.BaseCfg import BaseSceneCfg, TopCameraBaseSceneCfg
##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from sympy import false
from franka import FRANKA_PANDA_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
##
# Scene definition
##
@configclass
class BottleSceneCfg(TopCameraBaseSceneCfg):
    pass

@configclass
class Scene5aCfg(BottleSceneCfg):
    """
    Bottle with squeeze and rotate
    """
    bottle = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/Bottle/bottle_squeeze.usd",
            activate_contact_sensors=True,
            scale=(0.8, 0.8, 1.3),
        ),
        # spawn = sim_utils.UrdfFileCfg(asset_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility.urdf",
        #                               activate_contact_sensors=False,fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0, 0.492),
            rot=(0.707, 0.0, 0.707, 0.0),
            joint_pos={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            },
            joint_vel={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            }
            ),
        actuators={
            "prismatic":ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=1200.0,
                damping=0.0,
                friction=1.0,
            ),
            "cap_revolute":ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=1.0,
            ),
        }
    )

    # bottle_contact = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Bottle/bottle_squeeze/link_2",  # link_2에 연결
    #     update_period=0.0,  # 시뮬레이션과 같은 주기로 업데이트
    #     history_length=1,   # 최근 1 step의 contact 정보 저장
    #     debug_vis=False,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/panda_leftfinger","{ENV_REGEX_NS}/Robot/panda_rightfinger"],
    #     max_contact_data_count=100,
    # )

@configclass
class Scene5bCfg(BottleSceneCfg):
    """
    Bottle with rotate with open close guide
    """
    bottle = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/Bottle/bottle_ori_limit.usd",
            activate_contact_sensors=True,
            scale=(0.8, 0.8, 1.3),
        ),
        # spawn = sim_utils.UrdfFileCfg(asset_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility.urdf",
        #                               activate_contact_sensors=False,fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0, 0.492),
            rot=(0.707, 0.0, 0.707, 0.0),
            joint_pos={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            },
            joint_vel={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            }
            ),
        actuators={
            "prismatic":ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=1200.0,
                damping=0.0,
                friction=1.0,
            ),
            "cap_revolute":ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=1.0,
            ),
        }
    )

@configclass
class Scene5cCfg(BottleSceneCfg):
    """
    Bottle with rotate in reversed rotation
    """
    bottle = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/Bottle/bottle_ori_reversed.usd",
            activate_contact_sensors=True,
            scale=(0.9, 0.9, 1.4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0, 0.492),
            rot=(0.707, 0.0, 0.707, 0.0),
            joint_pos={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            },
            joint_vel={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            }
            ),
        actuators={
            "prismatic":ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=1200.0,
                damping=0.0,
                friction=1.0,
            ),
            "cap_revolute":ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=1.0,
            ),
        }
    )

@configclass
class Scene5dCfg(BottleSceneCfg):
    """
    Bottle with rotate to close
    """
    bottle = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/Bottle/bottle_ori_limit.usd",
            activate_contact_sensors=True,
            scale=(0.9, 0.9, 1.4),
        ),
        # spawn = sim_utils.UrdfFileCfg(asset_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility.urdf",
        #                               activate_contact_sensors=False,fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0, 0.492),
            rot=(0.707, 0.0, 0.707, 0.0),
            joint_pos={
                "joint_1" : 0.005,
                "joint_2" : 6.28,
            },
            joint_vel={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            }
            ),
        actuators={
            "prismatic":ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=1200.0,
                damping=0.0,
                friction=1.0,
            ),
            "cap_revolute":ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=1.0,
            ),
        }
    )

@configclass
class Scene5eCfg(BottleSceneCfg):
    """
    Bottle with rotate to close in reversed orientation
    """
    bottle = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Bottle",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/Bottle/bottle_ori_reversed.usd",
            activate_contact_sensors=True,
            scale=(0.8, 0.8, 1.3),
        ),
        # spawn = sim_utils.UrdfFileCfg(asset_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility.urdf",
        #                               activate_contact_sensors=False,fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.5, 0, 0.492),
            rot=(0.707, 0.0, 0.707, 0.0),
            joint_pos={
                "joint_1" : 0.005,
                "joint_2" : 6.28,
            },
            joint_vel={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
            }
            ),
        actuators={
            "prismatic":ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=1200.0,
                damping=0.0,
                friction=1.0,
            ),
            "cap_revolute":ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=0.0,
                damping=0.0,
                friction=1.0,
            ),
        }
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

from cfg.event_cfg.events import reset_root_state_uniform_ori, reset_camera_followup

@configclass
class Event5Cfg:
    # """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    robot_physics_material = EventTerm(
        func=mdp.events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_hand_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "static_friction_range": (0, 0),
            "dynamic_friction_range": (0, 0),
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

    bottle_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bottle", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )


    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("bottle", body_names=".*"),
            "mass_distribution_params": (0.2, 0.5),
            "operation": "abs",
            "distribution": "uniform"
        }
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
            "pose_range": {"x":(-0.1, 0.1), "y":(-0.2, 0.2), "z":(-0.2, 0.2), "roll":(-3.14, 3.14)},
            # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )

    randomize_root_state_startup = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="startup",
        params={
            "pose_range": {"x":(-0.1, 0.1), "y":(-0.2, 0.2), "z":(-0.2, 0.2), "roll":(-3.14, 3.14)},
            # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )

    # follow_camera = EventTerm(
    #     func=reset_camera_followup,
    #     mode="startup"
    # )

    # follow_camera_reset = EventTerm(
    #     func=reset_camera_followup,
    #     mode="reset"
    # )

@configclass
class EventBottleNoposCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    robot_physics_material = EventTerm(
        func=mdp.events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_hand_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "static_friction_range": (0, 0),
            "dynamic_friction_range": (0, 0),
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

    bottle_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bottle", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("bottle", body_names=".*"),
            "mass_distribution_params": (0.2, 0.5),
            "operation": "abs",
            "distribution": "uniform"
        }
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
            "pose_range": {"roll":(-3.14, 3.14)},
            # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )
 
    randomize_root_state_startup = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="startup",
        params={
            "pose_range": {"roll":(-3.14, 3.14)},
            # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )

@configclass
class EventBottleNoposColorCfg(EventBottleNoposCfg):
    randomize_bottle_cap_color_startup = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bottle", body_names=[".*"]),
            "event_name": "randomize_bottle_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )

    randomize_bottle_cap_color_reset = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bottle", body_names=[".*"]),
            "event_name": "randomize_bottle_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )

@configclass
class Event5RobotFricHighCfg(Event5Cfg):
    robot_physics_material = EventTerm(
        func=mdp.events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    robot_hand_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "static_friction_range": (0, 0),
            "dynamic_friction_range": (0, 0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_root_state = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="reset",
        params={
            "pose_range": {"x":(-0.08, 0.08), "y":(-0.2, 0.2), "z":(-0.12, 0.12), "roll":(-3.14, 3.14)},
            # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )
 
    randomize_root_state_startup = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="startup",
        params={
            "pose_range": {"x":(-0.08, 0.08), "y":(-0.25, 0.25), "z":(-0.12, 0.12), "roll":(-3.14, 3.14)},
            # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("bottle"),
        },
    )

@configclass
class EventBottleColorCfg(Event5RobotFricHighCfg):
    randomize_bottle_cap_color_startup = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bottle", body_names=[".*"]),
            "event_name": "randomize_bottle_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )


    randomize_bottle_cap_color_reset = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("bottle", body_names=[".*"]),
            "event_name": "randomize_bottle_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visual",
        },
    )

@configclass
class EventBottleTestCfg():
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

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

    robot_hand_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="panda_hand"),
            "static_friction_range": (0, 0),
            "dynamic_friction_range": (0, 0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    bottle_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("bottle", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.0, 1.1),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("bottle", body_names=".*"),
            "mass_distribution_params": (0.3, 0.3),
            "operation": "abs",
            "distribution": "uniform"
        }
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


from cfg.reward_cfg.reward import joint_pos_success, joint_pos

@configclass
class Reward5Cfg:
    """Reward terms for the MDP."""
    # 3. Open the drawer
    open_bottle_rew = RewTerm(
        func=joint_pos,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("bottle", joint_names=["joint_1"]),
            },
    )

@configclass
class Reward5CloseCfg:
    # Close the drawer
    close_bottle_rew = RewTerm(
        func=joint_pos,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("bottle", joint_names=["joint_1"]),
            "reverse": True,
            "offset": 0.005,
            },
    )

from isaaclab.envs import ManagerBasedRLEnvCfg
from cfg.BaseCfg import ObsCfg, TerminationsCfg

@configclass
class BottleSkillSqueezeCfg(ManagerBasedRLEnvCfg):
    scene: Scene5aCfg = Scene5aCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event5Cfg = Event5Cfg()
    rewards: Reward5Cfg = Reward5Cfg()
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
class BottleSkillOpenCfg(ManagerBasedRLEnvCfg):
    scene: Scene5bCfg = Scene5bCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event5Cfg = Event5Cfg()
    rewards: Reward5Cfg = Reward5Cfg()
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
class BottleSkillOpenReversedCfg(ManagerBasedRLEnvCfg):
    scene: Scene5cCfg = Scene5cCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event5Cfg = Event5Cfg()
    rewards: Reward5Cfg = Reward5Cfg()
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
class BottleSkillCloseCfg(ManagerBasedRLEnvCfg):
    scene: Scene5dCfg = Scene5dCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event5Cfg = Event5Cfg()
    rewards: Reward5Cfg = Reward5Cfg()
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
class BottleSkillCloseReversedCfg(ManagerBasedRLEnvCfg):
    scene: Scene5eCfg = Scene5eCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: Event5Cfg = Event5Cfg()
    rewards: Reward5Cfg = Reward5Cfg()
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