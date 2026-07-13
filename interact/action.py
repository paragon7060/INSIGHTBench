# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as f
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik_new import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from custom_lab.managers.action_counter_manager import ActionCounterTerm

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# Third Party
import carb
import numpy as np
from omni.isaac.core import World
from omni.isaac.core.objects import cuboid, sphere

########### OV #################
from omni.isaac.core.utils.types import ArticulationAction

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.logger import log_error, setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import (
    get_assets_path,
    get_filename,
    get_path_of_dir,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.camera import CameraObservation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class CuroboInteractionAction(ActionCounterTerm):
    r"""Inverse Kinematics action term.

    This action term performs pre-processing of the raw actions using scaling transformation.

    .. math::
        \text{action} = \text{scaling} \times \text{input action}
        \text{joint position} = J^{-} \times \text{action}

    where :math:`\text{scaling}` is the scaling applied to the input action, and :math:`\text{input action}`
    is the input action from the user, :math:`J` is the Jacobian over the articulation's actuated joints,
    and \text{joint position} is the desired joint position command for the articulation's joints.
    """

    cfg: actions_cfg.CuroboActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""

    def __init__(self, cfg: actions_cfg.HybridDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        self._full_joint_ids = self._joint_ids.copy()
        self._full_joint_ids.append(self._joint_ids[-1]+1)
        self._full_joint_ids.append(self._joint_ids[-1]+1)
        # panda
        self._full_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']
        
        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one match for the body name: {self.cfg.body_name}. Found {len(body_ids)}: {body_names}."
            )
        # save only the first body index
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        # check if articulation is fixed-base
        # if fixed-base then the jacobian for the base is not computed
        # this means that number of bodies is one less than the articulation's number of bodies
        if self._asset.is_fixed_base:
            self._jacobi_body_idx = self._body_idx - 1
        else:
            self._jacobi_body_idx = self._body_idx

        # log info for debugging
        carb.log_info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        carb.log_info(
            f"Resolved body name for the action term {self.__class__.__name__}: {self._body_name} [{self._body_idx}]"
        )
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # save the scale as tensors
        self._scale = torch.zeros((self.num_envs, self.action_dim), device=self.device)
        self._scale[:] = torch.tensor(self.cfg.scale, device=self.device)

        # convert the fixed offsets to torch tensors of batched shape
        if self.cfg.body_offset is not None:
            self._offset_pos = torch.tensor(self.cfg.body_offset.pos, device=self.device).repeat(self.num_envs, 1)
            self._offset_rot = torch.tensor(self.cfg.body_offset.rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._offset_pos, self._offset_rot = None, None
        
        self._skill_type = None

        self._gripper_joint_ids = [7,8]
        self._gripper_open_command = torch.tensor([[0.0400, 0.0400]], device=self.device)
        self._gripper_close_command = torch.tensor([[0.0000, 0.0000]], device=self.device)

        # CuRobo setting
        setup_curobo_logger("info")

        self.usd_help = UsdHelper()
        self.target_pose = None
        
        self.tensor_args = TensorDeviceType()

        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]

        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.robot_cfg["kinematics"]["ee_link"] = "ee_link"

        self.reactive_mode = False
        self.current_js = None
        self.zero_command = torch.zeros((7),device=self.device)

        self.cmd_plan = [None for _ in range(self.num_envs)]
        self.ik_goal = None
        self.solver = "mg" # "mg" or "mpc"
        self.pose_cost_metric = None

        self.use_debug_draw = True

        self.curobo_mg_prev()

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return 8

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        # self._joint_pos_des = torch.zeros(self.num_envs, 7) # 7 is # of joint
        self._raw_actions[:] = actions
        self._skill_type = self._raw_actions[:,0][0].item() # 0: abs / 1: rel controller
        # if not (self._skill_type == 0 or self._skill_type == 1):
        #     raise ValueError(
        #         f"The first action should be 0 or 1"
        #     )
        self._processed_actions[:] = self.raw_actions[:] * self._scale
        # obtain quantities from simulation
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_pos_des = joint_pos[:, self._joint_ids]
        self.curobo_mg_reset()
        self.curobo_update_js()
        self.pose_cost_metric = None
        self.plan_config.pose_cost_metric = None
        # # for test with target box
        # self._processed_actions[:,1:4] = self._env.scene["target"].get_local_poses()[0][0]
        self._joint_pos_des[0, :] = self.current_js.position
        self._asset.set_joint_position_target(self._joint_pos_des, joint_ids=self._joint_ids)

        self.curobo_goal_setting(self._processed_actions[:,1:])
        if self._skill_type == 0:
            print("Approach skill : Move to the target")
            self.curobo_mg_compute(self.ik_goal)
        if self._skill_type == 1:
            print("Grasp skill : Open gripper and move to the target and close gripper")
            self.current_js
            self.curobo_mg_compute(self.ik_goal)
        if self._skill_type == 2:
            print("Constrained movement skill")
            print("Constrained: Holding tool Orientation and automatically calculate pose_cost_metric")
            self.plan_config.pose_cost_metric = self.pose_cost_metric
            self.curobo_mg_compute(self.ik_goal)
        if self._skill_type == 3:
            print("Rotate skill")
            self.current_rt = self._asset.data.joint_pos[0][6]
            self.goal_rt = self.current_rt + self._processed_actions[0][1]
        if self._skill_type == 4:
            print("Pull/Push skill along ee_quat")
            self.plan_config.pose_cost_metric = self.pose_cost_metric
            self.curobo_mg_local_compute(self.ik_goal)



    def apply_actions(self, counter, decimation):
    # compute the delta in joint-space
        if self._skill_type == 0: # Approach
            self.curobo_mg_step(counter)
        elif self._skill_type == 1: # Grasp
            ### gripper open abs IK & gripper close ###
            # counter: ~ decimation/4 : open gripper
            if counter < decimation/4:
                self.gripper_open()
            # counter: d/4 ~ 3/4d : IK move
            elif counter < decimation*3/4:
                self.curobo_mg_step(counter, decimation/4)
            else:
                self.gripper_close()
        elif self._skill_type == 2: # Constrained Movement
            self.curobo_mg_step(counter)
        elif self._skill_type == 3: # Rotate
            self._asset.set_joint_position_target(self.goal_rt, joint_ids=6)
        elif self._skill_type in {4}: # Push/Pull
            self.curobo_mg_step(counter)


    # def reset(self, env_ids: Sequence[int] | None = None) -> None:
    #     self._raw_actions[env_ids] = 0.0
    #     self._asset.set_joint_position_target(self.init_joint_pos_des, joint_ids=self._joint_ids)

    """
    Helper functions.
    """

    def _compute_frame_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the pose of the target frame in the root frame.

        Returns:
            A tuple of the body's position and orientation in the root frame.
        """
        # obtain quantities from simulation
        ee_pose_w = self._asset.data.body_state_w[:, self._body_idx, :7]
        root_pose_w = self._asset.data.root_state_w[:, :7]
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        # account for the offset
        if self.cfg.body_offset is not None:
            ee_pose_b, ee_quat_b = math_utils.combine_frame_transforms(
                ee_pose_b, ee_quat_b, self._offset_pos, self._offset_rot
            )

        return ee_pose_b, ee_quat_b

    def _compute_ee_from_root(self) -> tuple[torch.Tensor, torch.Tensor]:
        # obtain quantities from simulation
        ee_pose_w = self._asset.data.body_state_w[:, self._body_idx, :7]
        root_pose_w = self._asset.data.root_state_w[:, :7]
        # compute the pose of the body in the root frame
        ee_pose_b, ee_quat_b = math_utils.subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )

        return ee_pose_b, ee_quat_b
    
    def _compute_frame_jacobian(self):
        """Computes the geometric Jacobian of the target frame in the root frame.

        This function accounts for the target frame offset and applies the necessary transformations to obtain
        the right Jacobian from the parent body Jacobian.
        """
        # read the parent jacobian
        jacobian = self._asset.root_physx_view.get_jacobians()[:, self._jacobi_body_idx, :, self._joint_ids]
        # account for the offset
        if self.cfg.body_offset is not None:
            # Modify the jacobian to account for the offset
            # -- translational part
            # v_link = v_ee + w_ee x r_link_ee = v_J_ee * q + w_J_ee * q x r_link_ee
            #        = (v_J_ee + w_J_ee x r_link_ee ) * q
            #        = (v_J_ee - r_link_ee_[x] @ w_J_ee) * q
            jacobian[:, 0:3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(self._offset_pos), jacobian[:, 3:, :])
            # -- rotational part
            # w_link = R_link_ee @ w_ee
            jacobian[:, 3:, :] = torch.bmm(math_utils.matrix_from_quat(self._offset_rot), jacobian[:, 3:, :])

        return jacobian

    def gripper_open(self):
        self._asset.set_joint_position_target(self._gripper_open_command, joint_ids=self._gripper_joint_ids)
    
    def gripper_close(self):
        self._asset.set_joint_position_target(self._gripper_close_command, joint_ids=self._gripper_joint_ids)
    
    def curobo_mg_prev(self):
        from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere

        # collision world 생성
        # world_cfg는 기존 코드에서 로드한 collision world config 객체라고 가정
        world_cfg = WorldConfig.from_dict({
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.02,
                }
            }
        })
        # collision use voxel viewer
        if self.use_debug_draw:
            render_voxel_size = 0.02
            # self.voxel_viewer = VoxelManager(100, size=render_voxel_size)

        # motion_gen
        if self.solver == "mg":
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg,
                self.tensor_args,
                collision_checker_type=CollisionCheckerType.BLOX,
                trim_steps=[1,-2],
                project_pose_to_goal_frame=False,
                ee_link_name="right_gripper",
            )
            motion_gen_config_local = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg,
                self.tensor_args,
                collision_checker_type=CollisionCheckerType.BLOX,
                trim_steps=[1,-2],
                # interpolation_dt=0.03,
                # collision_activation_distance=0.025,
                fixed_iters_trajopt=True,
                finetune_trajopt_iters=300,
                project_pose_to_goal_frame=True,
                ee_link_name="right_gripper",
            )
            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen_local = MotionGen(motion_gen_config_local)

            if not self.reactive_mode:
                print("warming up...")
                self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
                self.motion_gen_local.warmup(enable_graph=True, warmup_js_trajopt=False)

            self.world_model = self.motion_gen.world_collision
            print("Curobo is Ready")

            self.plan_config = MotionGenPlanConfig(
                enable_graph=False, enable_graph_attempt=10, max_attempts=60, enable_finetune_trajopt=True, time_dilation_factor=0.5, finetune_attempts=15
                )

        if self.solver == "mpc":
            self.mpc_config = MpcSolverConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg,
                use_cuda_graph=True,
                use_cuda_graph_metrics=True,
                use_cuda_graph_full_step=False,
                self_collision_check=True,
                collision_checker_type=CollisionCheckerType.MESH,
                use_mppi=True,
                use_lbfgs=False,
                use_es=False,
                store_rollouts=True,
                step_dt=0.02,
            )
            self.mpc = MpcSolver(self.mpc_config)

    def curobo_goal_setting(self, action):
        '''
        Action 을 IK goal 에 맞는 형식으로 변경
        skill 0 : IK goal directly
        skill 1 : Pull -> current EE pose + action
        '''
        if self.solver == "mg":
            if self._skill_type in {0,1}:
                self.ik_goal = Pose(
                    position=self.tensor_args.to_device(action[:,:3]),
                    quaternion=self.tensor_args.to_device(action[:,3:7])
                )
            # Constrained Movement skill
            if self._skill_type == 2:
                ee_pose = self.motion_gen.compute_kinematics(self.current_js).ee_pose.clone()
                ee_pos_curr, ee_quat_curr = ee_pose.position, ee_pose.quaternion
                
                linear_heading = action[:,:3]
                distance = action[:,3]
                direction_vector = linear_heading / torch.norm(linear_heading, dim=1, keepdim=True)
                movement = direction_vector * distance.unsqueeze(1)
                ee_pos_final = ee_pos_curr + movement
                self.ik_goal = Pose(
                    position=self.tensor_args.to_device(ee_pos_final),
                    quaternion=self.tensor_args.to_device(ee_quat_curr),
                    normalize_rotation=True
                )
                # make pose_cost_metric for ik_goal
                projected_position = ee_pos_curr - ee_pos_final
                cost_list = torch.ones(6)
                cost_list[3:] = (projected_position < 0.005).int()
                self.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(cost_list),
                    reach_full_pose=True,                
                )
            
            # Pull/Push Skill
            if self._skill_type in {4}: 
                ee_pose = self.motion_gen_local.compute_kinematics(self.current_js).ee_pose.clone()
                ee_pos_curr, ee_quat_curr = ee_pose.position, ee_pose.quaternion

                distance = action[0][0]
                self.ik_goal = self.compute_push_pull_ik_goal(ee_pos_curr, ee_quat_curr, distance)
                
                self.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
                    reach_full_pose=True,
                )



        # if self.solver == "mpc":
        #     retract_cfg = mpc.rollout_fn.dynamics_model.retract_config.clone().unsqueeze(0)
        #     joint_names = mpc.rollout_fn.joint_names

        #     state = mpc.rollout_fn.compute_kinematics(
        #         JointState.from_position(retract_cfg, joint_names=joint_names)
        #     )
        #     current_state = JointState.from_position(retract_cfg, joint_names=joint_names)
        #     retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        #     goal = Goal(
        #         current_state=current_state,
        #         goal_state=JointState.from_position(retract_cfg, joint_names=joint_names),
        #         goal_pose=retract_pose,
        #     )            
        #     if self._skill_type == 0:

    def curobo_mg_compute(self, ik_goal):
        '''
        MotionGeneration
        self.cmd_plan 에 생성된 trajectory 입력
        '''
        # self.curobo_update_world()
        result = self.motion_gen.plan_single(self.current_js, ik_goal, self.plan_config.clone())

        succ = result.success.item()
        if succ:
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(cmd_plan)
            self.cmd_plan = cmd_plan

        else:
            print(" MG Failed ")
            print(result.status)

    def curobo_mg_local_compute(self, ik_goal):
        '''
        MotionGeneration
        self.cmd_plan 에 생성된 trajectory 입력
        '''
        # self.curobo_update_world()
        result = self.motion_gen_local.plan_single(self.current_js, ik_goal, self.plan_config.clone())

        succ = result.success.item()
        if succ:
            cmd_plan = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(cmd_plan)
            self.cmd_plan = cmd_plan

        else:
            print(" MG Failed ")
            print(result.status)

    def curobo_mg_step(self, counter, start=0):
        '''
        self.cmd_plan 에 입력된 trajectory 를 따라서 counter에 맞게 robot control
        '''
        joint_pos_des = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel_des = torch.zeros_like(self._asset.data.joint_vel[:, self._joint_ids])
        joint_acc_des = torch.zeros_like(self._asset.data.joint_acc[:, self._joint_ids])
        # for s in range(len(self.cmd_plan)):
        counter -= start
        counter = int(counter) -1
        if self.cmd_plan[0] is not None and counter < len(self.cmd_plan.position):
            joint_pos_des[0, :] = self.cmd_plan[counter].position[:7].clone()
            joint_vel_des[0, :] = self.cmd_plan[counter].velocity[:7].clone()
            joint_acc_des[0, :] = self.cmd_plan[counter].acceleration[:7].clone()
            self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)
            # self._asset.set_joint_velocity_target(joint_vel_des, joint_ids=self._joint_ids)
            # self._asset.set_joint_effort_target(joint_acc_des, joint_ids=self._joint_ids)
        elif self.cmd_plan[0] is not None:
            joint_pos_des[0, :] = self.cmd_plan[-1,:].position[:7].clone()
            joint_vel_des[0, :] = self.cmd_plan[-1,:].velocity[:7].clone()
            joint_acc_des[0, :] = self.cmd_plan[-1,:].acceleration[:7].clone()
            self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)
            # self._asset.set_joint_velocity_target(joint_vel_des, joint_ids=self._joint_ids)
            # self._asset.set_joint_effort_target(joint_acc_des, joint_ids=self._joint_ids)


    def curobo_mg_reset(self):
        '''
        after mg, reset cmd_plan
        '''
        self.cmd_plan = [None for _ in range(self.num_envs)]
        self.ik_goal = None
        self.pose_cost_metric = None

    def curobo_ik_step(self, ik_goal):
        '''
        IK solution 생성
        '''
        result = self.motion_gen.ik_solver.solve_batch_env(ik_goal)
        # when set_joint_target_pose, you should give full_joint_ids
        ik_sol = result.js_solution.position.squeeze()
        return ik_sol

    def curobo_update_js(self):
        '''
        return : joint state
        '''
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        joint_vel = self._asset.data.joint_vel[:, self._joint_ids].clone()
        joint_acc = self._asset.data.joint_acc[:, self._joint_ids].clone()
        # why?
        self.current_js = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=joint_acc ,
            jerk=joint_vel * 1.0,
            joint_names=self._joint_names,
        )

    # def curobo_update_world(self):
    #     from curobo.geom.types import Cuboid, WorldConfig
    #     from curobo.types.base import TensorDeviceType
    #     from curobo.types.camera import CameraObservation
    #     from curobo.types.math import Pose

    #     # tensor device 및 voxel 관련 파라미터 설정
    #     tensor_args = TensorDeviceType()
    #     voxel_size = 0.05
    #     '''
    #     batch env 에 대해서 CollisionWorld update
    #     '''
    #     print("Updating world ...")
    #     self.world_model.decay_layer("world")
    #     # camera data
    #     camera_list = ["camera1", "camera2", "camera3"]
    #     for camera in camera_list:
    #         data = self._env.scene[camera].data
    #         tensor_args = TensorDeviceType()
    #         data_cam = self.convert_sim_camera_data(data, tensor_args)
    #         self.world_model.add_camera_frame(data_cam, "world")
    #     self.world_model.process_camera_frames(None, False)
    #     torch.cuda.synchronize()
    #     self.world_model.update_blox_hashes()
    #     bounding = Cuboid("t", dims=[10, 10, 10.0], pose=[0, 0, 0, 1, 0, 0, 0])
    #     voxels = self.world_model.get_voxels_in_bounding_box(bounding, voxel_size)
    #     if voxels.shape[0] > 0:
    #         voxels = voxels[voxels[:, 2] > voxel_size]
    #         voxels = voxels[voxels[:, 0] > 0.0]
    #         if self.use_debug_draw:
    #             self.draw_points(voxels)
    #         else:
    #             voxels = voxels.cpu().numpy()
    #             self.voxel_viewer.update_voxels(voxels[:, :3])
    #     else:
    #         if not self.use_debug_draw:
    #             self.voxel_viewer.clear()


    def convert_sim_camera_data(self, camera_data, tensor_args):
        """
        simulation 내의 CameraData 객체를 CameraObservation으로 변환
        
        Parameters:
            camera_data: 시뮬레이터에서 받아온 CameraData 객체
            tensor_args: TensorDeviceType 등 장치 정보를 포함하는 객체
        Returns:
            CameraObservation 객체 (device에 맞게 변환)
        """
        # Depth 이미지 추출: 
        depth_image = camera_data.output["distance_to_image_plane"].squeeze(0)
        
        # Intrinsics 추출:
        intrinsics = camera_data.intrinsic_matrices[0]
        
        # Pose 정보 추출:
        position = camera_data.pos_w[0]
        quaternion = camera_data.quat_w_world[0]
        # Pose 객체 생성
        pose = Pose(position=position, quaternion=quaternion)
        
        # CameraObservation 생성
        observation = CameraObservation(
            depth_image=depth_image,
            intrinsics=intrinsics,
            pose=pose
        )
        
        # 필요에 따라 device 변환
        return observation.to(device=tensor_args.device)
    
    def draw_points(self,voxels):
        # Third Party
        from matplotlib import cm
        # Third Party
        from omni.isaac.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
        # if draw.get_num_points() > 0:
        draw.clear_points()
        if len(voxels) == 0:
            return

        jet = cm.get_cmap("plasma").reversed()

        cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
        z_val = cpu_pos[:, 0]

        jet_colors = jet(z_val)

        b, _ = cpu_pos.shape
        point_list = []
        colors = []
        for i in range(b):
            # get list of points:
            point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
            colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 0.8)]
        sizes = [20.0 for _ in range(b)]

        draw.draw_points(point_list, colors, sizes)

    def quaternion_to_rotation_matrix(self, q):
        """
        q: torch.Tensor of shape (4,) representing quaternion.
        여기서는 q가 [w, x, y, z] 순서라고 가정
        Returns:
            3x3 rotation matrix.
        """
        # 만약 q의 shape가 (1,4)라면 squeeze해서 (4,)로
        q = q.squeeze()
        w, x, y, z = q[0], q[1], q[2], q[3]
        # 회전 행렬 공식 (단위 quaternion을 가정)
        R = torch.tensor([
            [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)]
        ], device=q.device)
        return R


    def compute_push_pull_ik_goal(self, ee_pos_curr, ee_quat_curr, distance, local_axis=torch.tensor([0.0, 0.0, 1.0])):
        """
        매개변수:
        ee_pos_curr: 현재 end-effector 위치, shape: (3,) 또는 (1,3)
        ee_quat_curr: 현재 end-effector orientation (quaternion), shape: (4,) 또는 (1,4)
        distance: 이동할 거리 (양수면 push, 음수면 pull)
        local_axis: 로컬 좌표계에서의 이동 축 (예: [1,0,0]이면 x축)
        
        반환:
        ik_goal: Pose 객체 (여기서는 curobo.types.math.Pose)로, 목표 위치와 orientation을 포함
        """
        # local_axis를 device에 맞추기
        local_axis = local_axis.to(ee_pos_curr.device)
        # quaternion을 회전 행렬로 변환
        R = self.quaternion_to_rotation_matrix(ee_quat_curr)
        # 월드 좌표계의 이동 방향: R * local_axis
        world_direction = torch.matmul(R, local_axis)
        # 단위 벡터로 정규화
        world_direction = world_direction / torch.norm(world_direction)
        # 목표 위치 = 현재 위치 + distance * world_direction
        target_pos = ee_pos_curr + distance * world_direction
        # 목표 orientation은 그대로 유지
        ik_goal = Pose(position=target_pos, quaternion=ee_quat_curr)
        return ik_goal
