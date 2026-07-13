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
from isaaclab.sensors import FrameTransformerCfg, CameraCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from cfg.event_cfg.events import reset_root_state_uniform_ori
from cfg.BaseCfg import BaseSceneCfg, TopCameraBaseSceneCfg
from custom_lab.envs.mdp import events as custom_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg

##
# Scene definition
##
@configclass
class MicrowaveSceneCfg(TopCameraBaseSceneCfg):
    """Configuration for the microwave scene with a robot and a microwave.
    """
    microwave = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Microwave",
        spawn=sim_utils.UsdFileCfg(
            usd_path="Assets/rebut/rebut_microwave.usd",
            activate_contact_sensors=True,
            scale=(0.6, 0.6, 0.6),
        ),
        # spawn = sim_utils.UrdfFileCfg(asset_path="Assets/UniDoorManip/RoundDoor/99650069962004/mobility.urdf",
        #                               activate_contact_sensors=False,fix_base=True),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.77, 0.1, 0.21),
            rot=(1.0, 0, 0, 0.0),
            joint_pos={
                "joint_1" : -0.005,
                "joint_2" : 0.0,
                "joint_3" : 0.0,
            },
            joint_vel={
                "joint_0" : 0.0,
                "joint_1" : 0.0,
            }
            ),
        actuators={
            "levers":ImplicitActuatorCfg(
                joint_names_expr=["joint_2", "joint_3", "joint_4"],
                effort_limit=87.0,
                velocity_limit=5.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.01,
            ),
            "button":ImplicitActuatorCfg(
                joint_names_expr=["joint_1"],
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


    # guide_number_one = AssetBaseCfg(

    # )
    
@configclass
class Scene2aCfg(MicrowaveSceneCfg):
    pass

##
# Observation Config
##

from cfg.obs_cfg.observations import root_pos, pos_guide

@configclass
class Scene2ObsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        ee_position = ObsTerm(func=mdp.ee_pos) # 3
        ee_quat = ObsTerm(func=mdp.ee_quat) # 4

        joint_pos = ObsTerm(func=mdp.joint_pos) # 9

        # asset_root_pos = ObsTerm( # 7
        #     func=root_pos,
        #     params={"asset_cfg": SceneEntityCfg("microwave")}
        # )

        # guide_arrow_pos = ObsTerm( # 3
        #     func=pos_guide,
        #     params={"asset_cfg": SceneEntityCfg("guide_arrow")}
        # )

        # guide_open_pos = ObsTerm( # 3
        #     func=pos_guide,
        #     params={"asset_cfg": SceneEntityCfg("guide_open")}
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventMicrowaveCfg:
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

    microwave_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("microwave", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.1, 1.35),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    start_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="startup",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_root_state = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("microwave"),
        },
    )

    randomize_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params = {
            "asset_cfg": SceneEntityCfg("microwave", body_names=".*"),
            "mass_distribution_params": (0.5, 1.0),
            "operation": "abs",
            "distribution": "uniform"
        }
    )
 
    # randomize_root_state_startup = EventTerm(
    #     func=mdp.reset_root_state_uniform,
    #     mode="startup",
    #     params={
    #         "pose_range": {"x":(-0.2, 0.2), "y":(-0.3, 0.3), "z":(-0.2, 0.2)},
    #         "velocity_range": {},
    #         "asset_cfg": SceneEntityCfg("microwave"),
    #     },
    # )

    randomize_light = EventTerm(
        func=custom_mdp.randomize_light,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("light"),
            "pose_range": {
                "x": (-0.2, 0.2), 
                "y": (-0.2, 0.2),
                "roll": (-5.0, 5.0) # 각도 (degree or radian, Sim 설정에 따름 보통 Radian)
            },
            "intensity_range": (5000.0, 20000.0),
            "color_rgb_range": ((0.1, 0.1, 0.1), (1.0, 1.0, 1.0)),
        },
    )



from cfg.reward_cfg.reward import joint_pos_success, joint_pos, joint_pos_for_joint_id

@configclass
class RewardMicrowaveCfg:
    joint_reward = RewTerm(
        func=joint_pos_for_joint_id,
        weight=1,
        params={
            "joint_id": "joint_0",
            "asset_cfg": SceneEntityCfg("microwave"),
        }
    )

    lever_reward_bottom = RewTerm(
        func=joint_pos_for_joint_id,
        weight=1.0,
        params={
            "joint_id": "joint_2",
            "asset_cfg": SceneEntityCfg("microwave"),
        }
    )

    lever_reward_top = RewTerm(
        func=joint_pos_for_joint_id,
        weight=1.0,
        params={
            "joint_id": "joint_2",
            "asset_cfg": SceneEntityCfg("microwave"),
        }
    )

from cfg.BaseCfg import ActionsCfg, ObsCfg, TerminationsCfg
from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class MicrowaveEnvCfg(ManagerBasedRLEnvCfg):
    scene: MicrowaveSceneCfg = Scene2aCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: Scene2ObsCfg = Scene2ObsCfg()
    events: EventMicrowaveCfg = EventMicrowaveCfg()
    rewards: RewardMicrowaveCfg = RewardMicrowaveCfg()
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