import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors.frame_transformer import OffsetCfg

from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from custom_lab.envs import mdp
from cfg.event_cfg.events import make_fixed_joints, reset_guide_position_with_random_flip

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from franka import FRANKA_PANDA_CFG

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


cam_height = 224
cam_width = 224
# cam_height = 672
# cam_width = 672

## instruction mode -> 672 x 672
# guide_cam_height = 224
# guide_cam_width = 224
guide_cam_height = 672
guide_cam_width = 672

##
# MDP settings
##

@configclass
class BaseSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="ee_tcp",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.1034),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
                    name="tool_leftfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
                    name="tool_rightfinger",
                    offset=OffsetCfg(
                        pos=(0.0, 0.0, 0.046),
                    ),
                ),
            ],
        )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(),
        spawn=sim_utils.GroundPlaneCfg(),
        collision_group=-1,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )

    # camera_top = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/cam_top",
    #     update_period=0.0,
    #     height=cam_height,
    #     width=cam_width,
    #     data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(0.6, 0, 1.0), rot=(0.707,0,0.707,0), convention="world"),
    # )
    camera_wrist = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/cam",
        update_period=0.0,
        height=cam_height,
        width=cam_width,
        data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0715, 0.02, -0.01), rot=(0.1272,0.6651,0.6962,0.2383), convention="opengl"),
    ) 
    # camera_left_front = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/cam_front1",
    #     update_period=0.0,
    #     height=cam_height,
    #     width=cam_width,
    #     data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-0.5, 0.8, 0.8), rot=(0.97,0.0,0.0,-0.25), convention="world"),
    # )

    # camera_right_front = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/cam_front2",
    #     update_period=0.0,
    #     height=cam_height,
    #     width=cam_width,
    #     data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-0.5, -0.8, 0.8), rot=(0.97,0,0,0.25), convention="world"),
    # )

    camera_right_shoulder= CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front3",
        update_period=0.0,
        height=cam_height,
        width=cam_width,
        data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.1, -0.75, 1.1), rot=(0.783,0.538,-0.179,-0.255), convention="opengl"),
    )

    camera_left_shoulder= CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front4",
        update_period=0.0,
        height=cam_height,
        width=cam_width,
        data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.1, 0.75, 1.1), rot=(0.255,0.179,-0.538,-0.783), convention="opengl"),
    )

@configclass
class TopCameraBaseSceneCfg(BaseSceneCfg):
    camera_guide = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_top",
        update_period=0.0,
        height=guide_cam_height,
        width=guide_cam_width,
        data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.506, 0, 0.9), rot=(0.707,0,0.707,0), convention="world"),
    )

@configclass
class FrontCameraBaseSceneCfg(BaseSceneCfg):
    camera_guide = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front",
        update_period=0.0,
        height=guide_cam_height,
        width=guide_cam_width,
        data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.55, 0, 0.9), rot=(0.5515,0.4426,-0.4426,-0.5515), convention="opengl"),
    )

@configclass
class FrontDoorCameraBaseSceneCfg(BaseSceneCfg):
    camera_guide = CameraCfg(
        prim_path="{ENV_REGEX_NS}/cam_front",
        update_period=0.0,
        height=guide_cam_height,
        width=guide_cam_width,
        data_types=["rgb", "distance_to_image_plane","instance_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.3, -0.25, 1.1), rot=(0.5515,0.4426,-0.4426,-0.5515), convention="opengl"),
    )

@configclass
class ObsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        ee_position = ObsTerm(func=mdp.ee_pos) # 3
        ee_quat = ObsTerm(func=mdp.ee_quat) # 4
        joint_pos = ObsTerm(func=mdp.joint_pos) # 9
        joint_vel = ObsTerm(func=mdp.joint_vel)
        actions = ObsTerm(func=mdp.last_action)

        """cam_data: rgb, depth, semantic seg"""
        wrist = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "rgb", "normalize": False})
        wrist_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "distance_to_image_plane", "normalize": False})
        wrist_semantic = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_wrist"), "data_type": "instance_segmentation_fast", "normalize": False})
        # left_front = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_left_front"), "data_type": "rgb", "normalize": False})
        # left_front_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_left_front"), "data_type": "distance_to_image_plane", "normalize": False})
        # left_front_semantic = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_left_front"), "data_type": "instance_segmentation_fast", "normalize": False})
        # right_front = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_right_front"), "data_type": "rgb", "normalize": False})
        # right_front_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_right_front"), "data_type": "distance_to_image_plane", "normalize": False})
        # right_front_semantic = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_right_front"), "data_type": "instance_segmentation_fast", "normalize": False})
        # top = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "rgb", "normalize": False})
        # top_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "distance_to_image_plane", "normalize": False})
        # top_semantic = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_top"), "data_type": "instance_segmentation_fast", "normalize": False})
        left_shoulder = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_left_shoulder"), "data_type": "rgb", "normalize": False})
        left_shoulder_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_left_shoulder"), "data_type": "distance_to_image_plane", "normalize": False})
        left_shoulder_semantic = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_left_shoulder"), "data_type": "instance_segmentation_fast", "normalize": False})
        right_shoulder = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_right_shoulder"), "data_type": "rgb", "normalize": False})
        right_shoulder_depth = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_right_shoulder"), "data_type": "distance_to_image_plane", "normalize": False})
        right_shoulder_semantic = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_right_shoulder"), "data_type": "instance_segmentation_fast", "normalize": False})
        guide = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_guide"), "data_type": "rgb", "normalize": False})
        guide_semantic = ObsTerm(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_guide"), "data_type": "instance_segmentation_fast", "normalize": False})
        # joint_pos_target = ObsTerm(func=mdp.joint_pos_target, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()

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
    
@configclass
class ContinuousJointActionsCfg:
    arm_action: ActionTermCfg = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=1.0,
            use_default_offset=False,
        )
    gripper_action: ActionTermCfg = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            use_default_offset=False,
    )

from isaaclab.envs.mdp import *

@configclass
class DifferentialIKActionsCfg:
    arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=True,
                ik_method="dls",
                )
    )
    gripper_action: ActionTermCfg = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            use_zero_offset=True,
        )
    
@configclass
class FrankaRelativeJointPosActionsCfg:
    arm_action: ActionTermCfg = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            use_zero_offset=True,
        )
    gripper_action: ActionTermCfg = RelativeJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            use_zero_offset=True,
        )
@configclass
class FrankaJointVelocityActionsCfg:
    arm_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
        )
    gripper_action: ActionTermCfg = JointVelocityActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
        )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    #(1) Time out
    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # #(2) mg_failed
    # mg_failed = DoneTerm(func=mg_failed)
    #(3) success
    # success = DoneTerm(func=mdp.success)

@configclass
class TestTerminationsCfg(TerminationsCfg):
    """Terminations for test"""
    timeout = DoneTerm(func=mdp.time_out, time_out = True)