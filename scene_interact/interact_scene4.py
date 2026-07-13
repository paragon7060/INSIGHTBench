# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab_tasks.manager_based.manipulation.sevig.config.termination import mg_failed

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from franka import FRANKA_PANDA_CFG  # isort: skip 

FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)


##
# Scene definition
##
@configclass
class CabinetSceneCfg(InteractiveSceneCfg):
    """Configuration for the cabinet scene with a robot and a cabinet.

    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the robot and end-effector frames
    """

    # robots, Will be populated by agent env cfg
    robot: ArticulationCfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # End-effector, Will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    washingmachine = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/WashingMachine",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/GAPartNet/103490_washingmachine/mobility_texture_gapartnet/mobility_texture_gapartnet.usd",
            activate_contact_sensors=True,
            scale=(0.5, 0.5, 0.5),
        ),
        # spawn = sim_utils.UrdfFileCfg(asset_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility.urdf",
        #                               activate_contact_sensors=False,fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.67, 0, 0.36674),
            rot=(1.0, 0, 0, 0.0),
            joint_pos={
                "joint_1" : 0.0,
                "joint_2" : 0.0,
                "joint_3" : 0.0,
            },
            joint_vel={
                "joint_0" : 0.0,
                "joint_1" : 0.0,
            }
            ),
        actuators={
            "button":ImplicitActuatorCfg(
                joint_names_expr=["joint_1","joint_2", "joint_3", "joint_4"],
                effort_limit=87.0,
                velocity_limit=10.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.5
            ),
            "door":ImplicitActuatorCfg(
                joint_names_expr=["joint_0"],
                effort_limit=87.0,
                velocity_limit=0.05,
                stiffness=0.0,
                damping=0.0,
                friction=0.001
            ),
        }
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

    # camera = CameraCfg(
    #     prim_path="{ENV_REGEX_NS}/cam1",
    #     update_period=0.1,
    #     height=480,
    #     width=640,
    #     data_types=["rgb", "distance_to_image_plane"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 3.0e5)
    #     ),
    #     offset=CameraCfg.OffsetCfg(pos=(-1, -1, 1.0), rot=(0.0,0.156,0.311,0.938), convention="world"),
    # )

    guide_power = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WashingMachine/link_4",
        init_state= AssetBaseCfg.InitialStateCfg(pos=(0.6,-0.265,0.495), rot=(1,0,0,0)),
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/guides/texts/text_open_black.usda",
            scale=(0.02, 0.02, 0.02),
            )
    )
# ##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    null_command = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: mdp.JointPositionActionCfg = MISSING
    # gripper_action: mdp.BinaryJointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        ee_position = ObsTerm(func=mdp.ee_pos)
        ee_quat = ObsTerm(func=mdp.ee_quat)

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        # cabinet_joint_vel = ObsTerm(
        #     func=mdp.joint_vel_rel,
        #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
        # )
        # rel_ee_drawer_distance = ObsTerm(func=mdp.rel_ee_drawer_distance)
        
        # actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # """Configuration for events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.1, 1.35),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    # cabinet_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names="drawer_handle_top"),
    #         "static_friction_range": (1.25, 1.5),
    #         "dynamic_friction_range": (1.25, 1.5),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )
    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("washingmachine", body_names=".*"),
            "mass_distribution_params": (0.5, 0.5),
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

    start_all = EventTerm(func=mdp.start_scene_to_default, mode="startup")

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # # 1. Approach the handle
    # approach_ee_handle = RewTerm(func=mdp.approach_ee_handle, weight=2.0, params={"threshold": 0.2})
    # align_ee_handle = RewTerm(func=mdp.align_ee_handle, weight=0.5)

    # # 2. Grasp the handle
    # approach_gripper_handle = RewTerm(func=mdp.approach_gripper_handle, weight=5.0, params={"offset": MISSING})
    # align_grasp_around_handle = RewTerm(func=mdp.align_grasp_around_handle, weight=0.125)
    # grasp_handle = RewTerm(
    #     func=mdp.grasp_handle,
    #     weight=0.5,
    #     params={
    #         "threshold": 0.03,
    #         "open_joint_pos": MISSING,
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=MISSING),
    #     },
    # )

    # # 3. Open the drawer
    # open_drawer_bonus = RewTerm(
    #     func=mdp.open_drawer_bonus,
    #     weight=7.5,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    # )
    # multi_stage_open_drawer = RewTerm(
    #     func=mdp.multi_stage_open_drawer,
    #     weight=1.0,
    #     params={"asset_cfg": SceneEntityCfg("cabinet", joint_names=["drawer_top_joint"])},
    # )

    # # 4. Penalize actions for cosmetic reasons
    # action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    # joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    #(1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    #(2) mg_failed
    # mg_failed = DoneTerm(func=mg_failed)


##
# Environment configuration
##


@configclass
class CabinetEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cabinet environment."""

    # Scene settings
    scene: CabinetSceneCfg = CabinetSceneCfg(num_envs=1, env_spacing=2.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
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

##
# FrankaCabinetEnv
##
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
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
class FrankaCabinetEnvAbsCfg(CabinetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Actions for the specific robot type (franka)
        self.actions.arm_action = CuroboInteractActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            scale=1,
            body_offset=CuroboInteractActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.04},
        #     close_command_expr={"panda_finger_.*": 0.0},
        # )

        # Listens to the required transforms
        # IMPORTANT: The order of the frames in the list is important. The first frame is the tool center point (TCP)
        # the other frames are the fingers
        self.scene.ee_frame = FrameTransformerCfg(
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


import torch
import matplotlib.pyplot as plt
import os

from isaaclab.envs import ManagerBasedRLStepEnv
from isaaclab.managers import ActionManager

# 이미지 저장 경로 설정
save_dir = "./output"
os.makedirs(save_dir, exist_ok=True)  # 폴더가 없으면 생성

def main():
    env_cfg = FrankaCabinetEnvAbsCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLStepEnv(cfg=env_cfg)    
    count = 0

    while simulation_app.is_running():
        # 사용자가 직접 8개의 파라미터 값을 입력하도록 요청
        print("\n[INPUT] 8개의 action parameter 값을 입력하세요 (쉼표로 구분):")
        user_input = input("예시: 0,0.45,0,1.0,-0.5,-0.5,-0.5,-0.5\n입력: ").strip()

        if user_input == "reset":
            env.reset()

        try:
            # 입력된 문자열을 쉼표 단위로 분리 후 float 리스트로 변환
            values = [float(val.strip()) for val in user_input.split(',')]
            if len(values) != 8:
                raise ValueError("8개의 파라미터가 필요합니다.")
        except Exception as e:
            print(f"[ERROR] 올바른 입력이 아닙니다: {e}")
            # 잘못된 입력이면 기본 파라미터를 사용합니다.
            values = [0, 0.5, 0.0, 0.7, -0.5, -0.5, -0.5, -0.5]

        # action_parameter 를 해당 값들로 설정
        action_parameter = torch.tensor(values).unsqueeze(0)  # env.action_manager.action와 동일한 shape 맞추기

        obs, _, _, _, _  = env.step(action_parameter)
        print("[Env 0]: Obs: ", obs["policy"][0])
        # img = env.scene["camera"].data.output["rgb"].cpu()
        # img = img.squeeze(0).numpy()
        # plot_camera(img, count)
        # count += 1
            
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()


'''
[[ command ]]
Scene1.
0, 0.6, 0.0, 0.73, -0.5, -0.5, -0.5, -0.5
1, 0.65, 0.0, 0.73, -0.5, -0.5, -0.5, -0.5
4, -0.3, 0, 0, 0, 0, 0, 0

Scene2.
{ button }
1, 0.55, -0.39, 0.456, -0.5, -0.5, -0.5, -0.5
4,0.1,0,0,0,0,0,0
{ bottom lever}
1, 0.568, -0.36, 0.53, -0.24, -0.66, -0.23, -0.67
1, 0.569, -0.35, 0.53, -0.24, -0.66, -0.23, -0.67
3,1.2,0,0,0,0,0,0
3,-1.2,0,0,0,0,0,0

'''