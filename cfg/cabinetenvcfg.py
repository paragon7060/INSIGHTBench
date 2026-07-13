import numpy as np
import torch,os,torchvision
from torchvision.utils import save_image

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.envs import ManagerBasedEnv,ManagerBasedEnvCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.assets import ArticulationCfg,RigidObjectCfg,AssetBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg
from franka import FRANKA_PANDA_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg, Camera, CameraCfg,RayCasterCamera, TiledCamera
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg,SceneEntityCfg,TerminationTermCfg, ObservationTermCfg, ObservationGroupCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
FRAME_MARKER_SMALL_CFG = FRAME_MARKER_CFG.copy()
FRAME_MARKER_SMALL_CFG.markers["frame"].scale = (0.10, 0.10, 0.10)
# Isaac Lab 프로젝트 루트 경로를 환경 변수에서 가져옵니다.
ISAACLAB_ROOT_DIR = os.environ.get("ISAACLAB_PATH", ".") # 환경 변수가 없으면 현재 폴더를 기준으로 함


# Pre-defined configs
##
# isort: off
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.utils.quantization_config import BitsAndBytesConfig

# isort: on
def image(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    save_image_to_file: bool = False,
    image_path: str = "image",
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
        elif data_type == "normals":
            images = (images + 1.0) * 0.5

    if save_image_to_file:
        dir_path, _ = os.path.split(image_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        # Get total successful episodes
        total_successes = 0
        if hasattr(env, "recorder_manager") and env.recorder_manager is not None:
            total_successes = env.recorder_manager.exported_successful_episode_count

        for tile in range(images.shape[0]):
            tile_chw = torch.swapaxes(images[tile : tile + 1].unsqueeze(1), 1, -1).squeeze(-1)
            filename = (
                f"{image_path}_{data_type}_trial_{total_successes}_tile_{tile}_step_{env.common_step_counter}.png"
            )
            save_image(tile_chw, filename)

    return images.clone()

def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_rows = np.floor(np.sqrt(num_origins))
    num_cols = np.ceil(num_origins / num_rows)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()
@configclass
class scenecfg(InteractiveSceneCfg):
    """Designs the scene."""
    # Ground-plane
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    # origins = define_origins(num_origins=1, spacing=2.0)

    # Origin 1 with Franka Panda
    # prim_utils.create_prim("{ENV_REGEX_NS}", "Xform", translation=origins[0])
    # -- Table
    
    # -- Robot
    Robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
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

    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        articulation_root_prim_path="/link_0",  # Cabinet의 루트 링크 지정
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/sektion_cabinet_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.0, 0, 0.6),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "joint_0": 0.0,
                "joint_1": 0.0,
                "joint_2": 0.0,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["joint_0","joint_1","joint_2" ],
                effort_limit=87.0,
                velocity_limit=100.0,
                stiffness=0.0,
                damping=0.0,
                friction=0.1,
            ),
        },
    )
    # cabinet = ArticulationCfg(
    #     prim_path="{ENV_REGEX_NS}/Cabinet",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/sektion_cabinet_instanceable.usd",
    #         activate_contact_sensors=False,
    #     ),
    #     init_state=ArticulationCfg.InitialStateCfg(
    #         pos=(1.0, 0, 0.6),
    #         rot=(0.0, 0.0, 0.0, 1.0),
    #         joint_pos={
    #             "door_left_joint": 0.0,
    #             "door_right_joint": 0.0,
    #             "drawer_bottom_joint": 0.0,
    #             "drawer_top_joint": 0.0,
    #         },
    #     ),
    #     actuators={
    #         "drawers": ImplicitActuatorCfg(
    #             joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
    #             effort_limit=87.0,
    #             velocity_limit=100.0,
    #             stiffness=0.0,
    #             damping=0.0,
    #             friction=0.1,
    #         ),
    #         "doors": ImplicitActuatorCfg(
    #             joint_names_expr=["door_left_joint", "door_right_joint"],
    #             effort_limit=87.0,
    #             velocity_limit=100.0,
    #             stiffness=0.0,
    #             damping=0.0,
    #             friction=0.1,
    #         ),
    #     },
    # )
    # return the scene information
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
            usd_path=f"{ISAACLAB_ROOT_DIR}/Assets/guides/texts/text_open_black.usda",
            scale=(0.02, 0.02, 0.02),
            )
    )
    camera_left_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_left_front",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane","semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.5, 0.8, 0.8), rot=(0.97,0.0,0.0,-0.25), convention="world"),
    )
    camera_right_front = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_right_front",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane","semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.5, -0.8, 0.8), rot=(0.97,0,0,0.25), convention="world"),
    )
    camera_right_up = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_right_up",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane","semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.7, -0.7, 1.3), rot=(0.7274,  0.4233, -0.2716, -0.4668), convention="world"),
    )#0.7265,  0.4790, -0.2712, -0.4113
    #[-0.7000, -0.6500,  1.4500]], device='cuda:0'), tensor([[ 0.7274,  0.4233, -0.2716, -0.4668]]
    camera_right_side = CameraCfg(
        prim_path="{ENV_REGEX_NS}/camera_right_side",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane","semantic_segmentation"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 2.5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(-0.7, -0.7, 1.3), rot=(0.7274,  0.4233, -0.2716, -0.4668), convention="world"),
    )
def apply_articulation_root_api(env: ManagerBasedEnv, env_ids: torch.Tensor = None):
    """Apply ArticulationRootAPI to cabinet if not present."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdPhysics, PhysxSchema
    
    stage = stage_utils.get_current_stage()
    cabinet_prim_path = "/World/envs/env_0/Cabinet/link_0"
    cabinet_prim = stage.GetPrimAtPath(cabinet_prim_path)
    
    if cabinet_prim.IsValid() and not cabinet_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        print(f"Applying ArticulationRootAPI to {cabinet_prim_path}")
        UsdPhysics.ArticulationRootAPI.Apply(cabinet_prim)
        PhysxSchema.PhysxArticulationAPI.Apply(cabinet_prim)

@configclass
class eventcfg :
    """Designs the event."""
    # event_manager = EventManager(cfg=EventManagerCfg(
    #     terms=[
    #         EventTermCfg(func=lambda env: env.get_time() > 10.0, mode="interval", interval_range_s=(10.0, 10.0)),
    #     ]
    # ))
    
    apply_cabinet_articulation_api = EventTermCfg(
        func=apply_articulation_root_api,
        mode="startup",
        params={},
    )
    
    robot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("Robot", body_names=".*"),
            "static_friction_range": (1.1, 1.35),
            "dynamic_friction_range": (1.1, 1.35),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    # cabinet_physics_material = EventTermCfg(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("cabinet", body_names=["drawer_handle_top","drawer_handle_bottom","door_right_nob_link","door_left_nob_link"]),
    #         "static_friction_range": (1.1, 1.35),
    #         "dynamic_friction_range": (1.1, 1.35),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )

    start_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="startup",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTermCfg(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

@configclass
class rewardcfg :
    pass
@configclass
class donecfg:
    """Termination terms for the MDP."""

    #(1) Time out
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    # #(2) mg_failed
    # mg_failed = DoneTerm(func=mg_failed)

@configclass
class obscfg:
    """Observation specifications for the MDP."""
    @configclass
    class RGBCameraPolicyCfg(ObservationGroupCfg):
        """Observations for policy group with RGB images."""
        cam_image = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("camera_left_front"),"data_type": "rgb", "normalize": False})

        # table_cam_normals = ObservationTermCfg(
        #     func=image,
        #     params={
        #         "sensor_cfg": SceneEntityCfg("camera_left_front"),
        #         "data_type": "rgb",
        #         "normalize": False,
        #         "save_image_to_file": False,
        #         "image_path": "camera_left_front",
        #     },
        # )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False
    # observation groups
    rgb_camera: RGBCameraPolicyCfg = RGBCameraPolicyCfg()
    
@configclass
class acioncfg():
    arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="Robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.2,  # 스케일을 줄여서 더 안정적으로
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
        )
@configclass
class vlacfg(ManagerBasedEnvCfg):
    scene: scenecfg = scenecfg(num_envs=1, env_spacing=2.0)
    event: eventcfg = eventcfg()
    reward: rewardcfg = rewardcfg()
    done: donecfg = donecfg()
    observations: obscfg = obscfg()
    actions: acioncfg = acioncfg()
    def __post_init__(self):
        """Post initialization."""
        # viewer settings

        # step settings
        self.decimation = 4  # env step every 10 sim steps: 100Hz / 10 = 10Hz
        # simulation settings
        self.sim.dt = 1/60  # sim step every 10ms: 100Hz