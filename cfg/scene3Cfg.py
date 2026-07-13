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
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from cfg.event_cfg.events import make_fixed_joints, reset_guide_position_with_random_flip
from cfg.BaseCfg import BaseSceneCfg, FrontDoorCameraBaseSceneCfg
from custom_lab.envs.mdp import events as custom_mdp

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from franka import FRANKA_PANDA_CFG  # isort: skip

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)

##
# Scene definition
##
@configclass
class DoorSceneCfg(FrontDoorCameraBaseSceneCfg):
    pass

@configclass
class BaseDoorSceneCfg(DoorSceneCfg):
    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility/mobility_modified_long.usd",
            activate_contact_sensors=False,
            scale=(0.9, 0.9, 0.9),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.505, 0, 0.815),
            rot=(0.707, 0, 0.707, 0.0),),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
                friction=10,
            ),
            "lever": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
                friction=1,
            )
        }
    )

@configclass
class Scene3aCfg(BaseDoorSceneCfg): # pull
    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/AdaManip/door/99690419962024/mobility_pull_ccw.usd",
            activate_contact_sensors=False,
            scale=(0.9, 0.9, 0.9),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.505, 0, 0.815),
            rot=(0.707, 0, 0.707, 0.0),),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
                friction=10,
            ),
            "lever": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,    
                friction=1,
            )
        }
    )
    guide_open = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Door/link_1/guide_pull",
        init_state= AssetBaseCfg.InitialStateCfg(pos=(0.624,-0.04,-0.02), rot=(0.5,0.5,-0.5,0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/guides/group_guide/guide_door_pull_ccw0.usd",
            scale=(1.5, 1.3, 1.5),
            )
    )

@configclass
class Scene3bCfg(BaseDoorSceneCfg):
    door = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Door",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/AdaManip/door/99690419962024/mobility_push_cw.usd",
            activate_contact_sensors=False,
            scale=(0.9, 0.9, 0.9),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.505, 0, 0.815),
            rot=(0.707, 0, 0.707, 0.0),),
        actuators={
            "door": ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
                effort_limit=87.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
                friction=10,
            ),
            "lever": ImplicitActuatorCfg(
                joint_names_expr=["joint_2"],
                effort_limit=87.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
                friction=1,
            )
        }
    )
    guide_push = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Door/link_1/guide_push",
        init_state= AssetBaseCfg.InitialStateCfg(pos=(0.624,-0.04,-0.02), rot=(0.5,0.5,-0.5,0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/guides/group_guide/guide_door_push_cw0.usd",
            scale=(1.5, 1.3, 1.5),
            )
    )

@configclass
class Scene3cCfg(BaseDoorSceneCfg):
    guide_rotate = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/guide_rotate",
        init_state= AssetBaseCfg.InitialStateCfg(pos=(0.611,0.1026,0.65022), rot=(0.5,-0.5,-0.5,0.5)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/guides/texts/text_rotate.usdc",
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

from cfg.event_cfg.events import reset_root_state_uniform_ori, reset_camera_followup

@configclass
class EventDoorCfg:
    # """Configuration for events."""
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.3, 1.75),
            "dynamic_friction_range": (1.1, 1.3),
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

    door_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=["link_2"]),
            "static_friction_range": (1.3, 1.75),
            "dynamic_friction_range": (1.1, 1.3),
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
            # "pose_range": {"x":(-0.08, 0.08), "y":(-0.2, 0.2), "z":(-0.12, 0.12), "roll":(-3.14, 3.14)},
            "pose_range": {"x":(-0.05, 0.05), "y":(-0.2, 0.2), "z":(-0.1, 0.10)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("door"),
        },
    )
 
    randomize_root_state_startup = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="startup",
        params={
            "pose_range": {"x":(-0.08, 0.08), "y":(-0.25, 0.25), "z":(-0.12, 0.12)},
            # "pose_range": {"x":(-0.05, 0.05), "y":(-0.2, 0.2), "z":(-0.1, 0.10)},
            # "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("door"),
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("door", body_names=".*"),
            "mass_distribution_params": (0.2, 0.8),
            "operation": "abs",
            "distribution": "uniform"
        }
    )

@configclass
class EventDoorNoposCfg:

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.3, 1.75),
            "dynamic_friction_range": (1.1, 1.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    door_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=["link_2"]),
            "static_friction_range": (1.3, 1.75),
            "dynamic_friction_range": (1.1, 1.3),
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
            "asset_cfg": SceneEntityCfg("door", body_names=".*"),
            "mass_distribution_params": (0.2, 0.8),
            "operation": "abs",
            "distribution": "uniform"
        }
    )

@configclass
class EventDoorNoposColorCfg(EventDoorNoposCfg):
    randomize_door_cap_color_startup = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=[".*"]),
            "event_name": "randomize_door_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visuals",
        },
    )

    randomize_door_cap_color_reset = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=[".*"]),
            "event_name": "randomize_door_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visuals",
        },
    )

@configclass
class EventDoorTestCfg():

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

    door_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=["link_2"]),
            "static_friction_range": (20, 20),
            "dynamic_friction_range": (15, 15),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("door", body_names=".*"),
            "mass_distribution_params": (0.2, 0.8),
            "operation": "abs",
            "distribution": "uniform"
        }
    )

    randomize_door_mass_door = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("door", body_names=["base","link_0","link_1"]),
            "mass_distribution_params": (1.5, 2.5),
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

@configclass
class EventDoorColorCfg(EventDoorCfg):
    randomize_door_cap_color_startup = EventTerm(
        func=mdp.randomize_visual_color,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=[".*"]),
            "event_name": "randomize_door_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visuals",
        },
    )

    randomize_door_cap_color_reset = EventTerm(
        func=mdp.randomize_visual_color,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("door", body_names=[".*"]),
            "event_name": "randomize_door_cap_color_startup",
            "colors": {"r":(0.1,1.0), "g":(0.1,1.0), "b":(0.1,1.0)},
            "mesh_name": "/link_.*/visuals",
        },
    )

from cfg.reward_cfg.reward import joint_pos_success, joint_pos

@configclass
class Reward3Cfg:
    """Reward terms for the MDP."""
    # 3. Open the drawer
    open_door_rew = RewTerm(
        func=joint_pos,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["joint_1"]),
            },
    )

from isaaclab.envs import ManagerBasedRLEnvCfg
from cfg.BaseCfg import ObsCfg, TerminationsCfg

@configclass
class DoorSkillPullEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene3aCfg = Scene3aCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: EventDoorCfg = EventDoorCfg()
    rewards: Reward3Cfg = Reward3Cfg()
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
class DoorSkillPushEnvCfg(ManagerBasedRLEnvCfg):
    scene: Scene3bCfg = Scene3bCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: EventDoorCfg = EventDoorCfg()
    rewards: Reward3Cfg = Reward3Cfg()
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