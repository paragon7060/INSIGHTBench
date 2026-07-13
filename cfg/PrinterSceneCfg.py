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
from cfg.event_cfg.events import reset_root_state_uniform_ori
from cfg.BaseCfg import BaseSceneCfg, TopCameraBaseSceneCfg
from custom_lab.envs.mdp import events as custom_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class PrinterSceneCfg(TopCameraBaseSceneCfg):
    pass

class PrinterSceneTestCfg(PrinterSceneCfg):
    printer = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Printer",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/GAPartNet/Printer/103811/mobility.usd",
            activate_contact_sensors=True,
            scale=(1.0, 1.0, 1.0),
        ),
        # spawn = sim_utils.UrdfFileCfg(asset_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility.urdf",
        #                               activate_contact_sensors=False,fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.7, 0, 0.38),
            rot=(1, 0.0, 0.0, 0.0),
            joint_pos={
                ".*" : 0.002,
            },
            joint_vel={
                ".*" : 0.0001,
            }
            ),
        actuators={
            "all":ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit=87.0,
                velocity_limit=1.0,
                stiffness=0.0,
                damping=1.0,
                friction=1.0,
            ),
        }
    )

@configclass
class EventPrinterCfg:
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

    bottle_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("printer", body_names=".*"),
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
            "asset_cfg": SceneEntityCfg("printer", body_names=".*"),
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

    randomize_robot_root_state = EventTerm(
        func=reset_root_state_uniform_ori,
        mode="startup",
        params={
            "pose_range": {"x":(0.0, 0.0), "y":(0.0, 0.0), "z":(0.5, 0.5)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # randomize_root_state = EventTerm(
    #     func=reset_root_state_uniform_ori,
    #     mode="reset",
    #     params={
    #         "pose_range": {"x":(-0.1, 0.1), "y":(-0.2, 0.2), "z":(-0.2, 0.2), "roll":(-3.14, 3.14)},
    #         # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
    #         # "pose_range": {},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("printer"),
    #     },
    # )

    # randomize_root_state_startup = EventTerm(
    #     func=reset_root_state_uniform_ori,
    #     mode="startup",
    #     params={
    #         "pose_range": {"x":(-0.1, 0.1), "y":(-0.2, 0.2), "z":(-0.2, 0.2), "roll":(-3.14, 3.14)},
    #         # "pose_range": {"x":(-0.05, 0.05), "y":(-0.1, 0.1), "z":(-0.05, 0.05)},
    #         # "pose_range": {},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("printer"),
    #     },
    # )

from cfg.reward_cfg.reward import joint_pos_success, joint_pos, joint_pos_for_joint_id

@configclass
class RewardPrinterCfg:
    joint_reward = RewTerm(
        func=joint_pos_for_joint_id,
        weight=1,
        params={
            "joint_id": "joint_1",
            "asset_cfg": SceneEntityCfg("printer"),
        }
    )

from cfg.BaseCfg import ActionsCfg, ObsCfg, TerminationsCfg

@configclass
class PrinterEnvCfg(ManagerBasedRLEnvCfg):
    scene: PrinterSceneCfg = PrinterSceneTestCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObsCfg = ObsCfg()
    events: EventPrinterCfg = EventPrinterCfg()
    rewards: RewardPrinterCfg = RewardPrinterCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 500
        self.episode_length_s = 20.0
        self.viewer.eye = (-2.0, 2.0, 2.0)
        self.viewer.lookat = (0.8, 0.0, 0.5)
        # simulation settings
        self.sim.dt = 1 / 120  # 60Hz
        self.sim.render_interval = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.env_spacing = 2.0