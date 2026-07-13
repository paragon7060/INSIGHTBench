# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Trajectory generator for robot manipulation tasks using CuRobo motion planning."""

import torch
import math
import signal
import time
from contextlib import contextmanager

from isaaclab.assets.articulation import Articulation
from custom_lab.envs.manager_based_rl_step_env import ManagerBasedContinuousEnv, ManagerBasedRLStepEnv
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)
from curobo.rollout.cost.pose_cost import PoseCostMetric


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class TrajectoryGenerator:
    """Generates robot trajectories for different manipulation skills."""
    
    ARM_DOF = 7
    HOLD_STEPS_DEFAULT = 10
    GRIPPER_OPEN_POSITION = 0.04
    GRIPPER_CLOSE_POSITION = 0.0

    def __init__(self, robot_name: str, env: ManagerBasedRLStepEnv):
        """Initialize trajectory generator.
        
        Args:
            robot_name: Name of the robot configuration
            env: Isaac Sim environment
        """
        # Load robot configuration
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.num_envs = env.num_envs
        self.device = env.device
        self.tensor_args = TensorDeviceType()
        self._asset: Articulation = env.scene["robot"]

        # Get joint information
        self._joint_ids, self._joint_names = self._asset.find_joints(env.action_manager.cfg.arm_action.joint_names)
        self._num_joints = len(self._joint_ids)
        gripper_cfg = getattr(env.action_manager.cfg, "gripper_action", None)
        gripper_names = gripper_cfg.joint_names if gripper_cfg is not None else ["panda_finger_joint.*"]
        self._gripper_joint_ids, self._gripper_joint_names = self._asset.find_joints(gripper_names)

        self._full_joint_ids = self._joint_ids.copy()
        self._full_joint_ids.append(self._joint_ids[-1]+1)
        self._full_joint_ids.append(self._joint_ids[-1]+1)
        # panda
        self._full_joint_names = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2']

        # Gripper control
        self.gripper_effort_plan = [None for _ in range(self.num_envs)]
        self.gripper_effort_cmd = torch.zeros((self.num_envs, len(self._gripper_joint_ids)), device=self.device)

        # Setup CuRobo
        setup_curobo_logger("info")
        self._skill_type = None

        # Robot configuration
        self.j_names = self.robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.default_config = self.robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.robot_cfg["kinematics"]["ee_link"] = "ee_link"

        # State tracking
        self.reactive_mode = False
        self.current_js = None
        self.cmd_plan = [None for _ in range(self.num_envs)]
        self.ik_goal = [None for _ in range(self.num_envs)]
        self.last_plan_debug = {}
        self.solver = "mg"
        self.pose_cost_metric = None

        # Rotation skill state
        self._rotate_remaining = torch.zeros(self.num_envs, device=self.device)
        self._rotate_dir = torch.zeros(self.num_envs, device=self.device)
        self._rotate_step_size = 0.05
        self.fallback_steps = 10
        
        # Gripper state tracking (default: open)
        self._gripper_states = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)  # True: closed, False: open
        self._gripper_positions = torch.zeros((self.num_envs, len(self._gripper_joint_ids)), device=self.device)
        
        # Initialize all environments to open gripper state
        for i in range(self.num_envs):
            self.set_gripper_state(i, False)  # False = open
        
        self.curobo_mg_prev()

    @property
    def action_dim(self) -> int:
        return 8

    def command(self, actions: torch.Tensor):
        """Generate trajectory based on skill type and action parameters.
        
        Args:
            actions: Action tensor with skill type and parameters
        """
        plan_start = time.perf_counter()
        self._skill_type = actions[:,0][0].item()
        self.last_plan_debug = {
            "skill_type": int(self._skill_type),
            "action_env0": self._debug_tensor_list(actions[0, :8]),
            "pre_grasp_success": None,
            "final_grasp_success": None,
            "standard_attempts": 0,
            "local_attempts": 0,
            "success": None,
            "fallback_used": [False for _ in range(self.num_envs)],
            "plan_lengths": [0 for _ in range(self.num_envs)],
            "elapsed_s": 0.0,
            "exception": None,
            "timeout": False,
        }
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        self._joint_pos_des = joint_pos[:, self._joint_ids]
        
        # Reset and update state
        self.curobo_mg_reset()
        self.curobo_update_js()
        self.pose_cost_metric = None
        self.plan_config.pose_cost_metric = None
        self._joint_pos_des[:, :] = self.current_js.position

        # Set goal based on skill type
        self.curobo_goal_setting(actions[:,1:])
        
        # Generate trajectory for each skill type
        if self._skill_type == 0:
            # Reach skill: direct motion to target pose
            self.curobo_mg_compute(self.ik_goal)
            
        elif self._skill_type == 1:
            # Grasp skill: pre-grasp -> grasp -> close gripper
            self._generate_grasp_trajectory()
            
        elif self._skill_type == 2:
            # Push skill: linear motion with pose cost metric
            self.plan_config.pose_cost_metric = self.pose_cost_metric
            self.curobo_mg_compute(self.ik_goal)
            
        elif self._skill_type == 3:
            # Rotate skill: wrist joint rotation
            self._generate_rotate_trajectory()
            
        elif self._skill_type == 4:
            # Local motion skill
            self.plan_config.pose_cost_metric = self.pose_cost_metric
            self.curobo_mg_local_compute(self.ik_goal)

        # Add gripper control to all trajectories
        self._add_gripper_control()

        # Update gripper state after grasp skill completion
        if self._skill_type == 1:
            self.update_gripper_state_after_grasp()

        self._finalize_last_plan_debug(plan_start)
        return self.cmd_plan

    def _debug_tensor_list(self, value, digits: int = 4):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().flatten().tolist()
        return [round(float(v), digits) for v in value]

    def _set_last_plan_success(self, success):
        if isinstance(success, torch.Tensor):
            self.last_plan_debug["success"] = [bool(v) for v in success.detach().cpu().flatten().tolist()]
        elif success is not None:
            self.last_plan_debug["success"] = [bool(v) for v in success]

    def _finalize_last_plan_debug(self, plan_start: float):
        lengths = []
        fallback_used = []
        for cmd in self.cmd_plan:
            if cmd is None:
                lengths.append(0)
                fallback_used.append(True)
                continue
            length = int(cmd.position.shape[0])
            lengths.append(length)
            fallback_used.append(length <= self.fallback_steps)
        self.last_plan_debug["plan_lengths"] = lengths
        self.last_plan_debug["fallback_used"] = fallback_used
        if self.last_plan_debug.get("success") is None:
            self.last_plan_debug["success"] = [cmd is not None for cmd in self.cmd_plan]
        self.last_plan_debug["elapsed_s"] = round(time.perf_counter() - plan_start, 4)

    def gripper_reset(self):
        for i in range(self.num_envs):
            self.set_gripper_state(i, False)

    def _generate_grasp_trajectory(self):
        """Generate grasp trajectory: pre-grasp -> grasp -> close gripper."""
        joint_dim = self.ARM_DOF
        offset_distance = -0.12
        
        # Generate pre-grasp pose
        pre_grasp_goal_pose = self.compute_push_pull_ik_goal(
            ee_pos_curr=self.ik_goal.position.squeeze(0),
            ee_quat_curr=self.ik_goal.quaternion.squeeze(0),
            distance=offset_distance,
        )

        # Plan pre-grasp motion
        pre_grasp_success = self.curobo_mg_compute(pre_grasp_goal_pose)
        self.last_plan_debug["pre_grasp_success"] = [bool(v) for v in pre_grasp_success.detach().cpu().flatten().tolist()]

        # Get current end-effector poses
        ee_pose_all = self.motion_gen.compute_kinematics(self.current_js).ee_pose.clone()
        ee_pos_all = ee_pose_all.position
        ee_quat_all = ee_pose_all.quaternion

        # Prepare final joint states for grasp planning
        pregrasp_final_positions = torch.zeros((self.num_envs, joint_dim), device=self.tensor_args.device)
        pregrasp_final_velocities = torch.zeros((self.num_envs, joint_dim), device=self.tensor_args.device)
        pregrasp_final_accelerations = torch.zeros((self.num_envs, joint_dim), device=self.tensor_args.device)

        for s in range(self.num_envs):
            if pre_grasp_success[s] and self.cmd_plan[s] is not None and len(self.cmd_plan[s].position) > 0:
                last = self.cmd_plan[s][-1]
                pregrasp_final_positions[s, :] = last.position[:joint_dim].clone().to(self.tensor_args.device)
            else:
                pregrasp_final_positions[s, :] = self.current_js.position[s, :joint_dim].clone().to(self.tensor_args.device)

        js_pregrasp_final = JointState(
            position=pregrasp_final_positions,
            velocity=pregrasp_final_velocities,
            acceleration=pregrasp_final_accelerations,
            joint_names=self._joint_names,
        )

        # Set pose cost metric for grasp planning
        self.pose_cost_metric = PoseCostMetric(
            hold_partial_pose=True,
            hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
            reach_full_pose=True,
        )
        self.plan_config.pose_cost_metric = self.pose_cost_metric

        # Update goals for failed pre-grasp environments
        for s in range(self.num_envs):
            if not pre_grasp_success[s]:
                self.ik_goal.position[s] = ee_pos_all[s]
                self.ik_goal.quaternion[s] = ee_quat_all[s]

        # Plan final grasp motion
        result_grasp = None
        try:
            self.last_plan_debug["local_attempts"] += 1
            result_grasp = self.motion_gen_local.plan_batch_env(
                js_pregrasp_final,
                self.ik_goal.clone(),
                self.plan_config.clone(),
            )
            if result_grasp is not None:
                self.last_plan_debug["final_grasp_success"] = [
                    bool(v) for v in result_grasp.success.detach().cpu().flatten().tolist()
                ]
        except Exception as e:
            import traceback as _tb
            print(f"[Error] Final grasp planning exception: {e}")
            _tb.print_exc()
            self.last_plan_debug["exception"] = f"final_grasp: {type(e).__name__}: {e}"
            result_grasp = None

        # Generate complete grasp trajectory
        self._combine_grasp_trajectories(pre_grasp_success, result_grasp)

    def _combine_grasp_trajectories(self, pre_grasp_success, result_grasp):
        """Combine pre-grasp, grasp, and gripper control trajectories."""
        hold_steps = self.HOLD_STEPS_DEFAULT

        if result_grasp is not None and torch.count_nonzero(result_grasp.success) > 0:
            pos_trajs = result_grasp.get_paths() if self.num_envs > 1 else result_grasp.get_interpolated_plan()

            for s in range(self.num_envs):
                if result_grasp.success[s]:
                    # Generate grasp trajectory
                    cmd_plan_grasp = self._trim_js7(self.motion_gen.get_full_js(pos_trajs[s] if self.num_envs > 1 else pos_trajs))

                    # Generate intermediate hold trajectory
                    if pre_grasp_success[s] and self.cmd_plan[s] is not None and len(self.cmd_plan[s].position) > 0:
                        src = self._trim_js7(self.cmd_plan[s])
                        last_js = JointState(
                            position=src[-1].position.clone(),
                            velocity=torch.zeros_like(src[-1].velocity.clone()),
                            acceleration=torch.zeros_like(src[-1].acceleration.clone()),
                            jerk=torch.zeros_like(src[-1].jerk.clone()) if src.jerk is not None else torch.zeros_like(src[-1].velocity.clone()),
                            joint_names=self._joint_names
                        )
                    else:
                        last_js = JointState(
                            position=self.current_js.position[s, :self.ARM_DOF].clone(),
                            velocity=torch.zeros(self.ARM_DOF, device=self.tensor_args.device),
                            acceleration=torch.zeros(self.ARM_DOF, device=self.tensor_args.device),
                            jerk=torch.zeros(self.ARM_DOF, device=self.tensor_args.device),
                            joint_names=self._joint_names
                        )

                    cmd_plan_inter = self._make_hold_from_js(last_js, hold_steps)

                    # Generate gripper open trajectory (start)
                    start_hold_pos = self.current_js.position[s, :self.ARM_DOF].unsqueeze(0).repeat(hold_steps, 1)
                    start_hold_vel = torch.zeros_like(start_hold_pos)
                    start_hold_acc = torch.zeros_like(start_hold_pos)
                    start_hold_jerk = torch.zeros_like(start_hold_pos)
                    cmd_plan_gripper_open = JointState(
                        position=start_hold_pos,
                        velocity=start_hold_vel,
                        acceleration=start_hold_acc,
                        jerk=start_hold_jerk,
                        joint_names=self._joint_names,
                    )

                    # Generate gripper close trajectory (end)
                    end_js_last = cmd_plan_grasp[-1]
                    end_hold_pos = end_js_last.position.unsqueeze(0).repeat(hold_steps, 1)
                    end_hold_vel = torch.zeros_like(end_hold_pos)
                    end_hold_acc = torch.zeros_like(end_hold_pos)
                    end_hold_jerk = torch.zeros_like(end_hold_pos)
                    cmd_plan_gripper_close = JointState(
                        position=end_hold_pos,
                        velocity=end_hold_vel,
                        acceleration=end_hold_acc,
                        jerk=end_hold_jerk,
                        joint_names=self._joint_names,
                    )

                    # Generate gripper effort plan
                    total_len = (
                        cmd_plan_gripper_open.position.shape[0]
                        + (self._trim_js7(self.cmd_plan[s]).position.shape[0] if self.cmd_plan[s] is not None else 0)
                        + cmd_plan_inter.position.shape[0]
                        + cmd_plan_grasp.position.shape[0]
                        + cmd_plan_gripper_close.position.shape[0]
                    )
                    effort_plan = torch.zeros((total_len, len(self._gripper_joint_ids)), device=self.device)
                    idx = 0
                    idx += hold_steps
                    if self.cmd_plan[s] is not None:
                        idx += self._trim_js7(self.cmd_plan[s]).position.shape[0]
                    idx += hold_steps
                    idx += cmd_plan_grasp.position.shape[0]
                    effort_plan[0: idx, :] = self.GRIPPER_OPEN_POSITION
                    effort_plan[idx: idx + hold_steps, :] = self.GRIPPER_CLOSE_POSITION
                    self.gripper_effort_plan[s] = effort_plan

                    # Combine all trajectory segments
                    self._combine_trajectory_segments(s, cmd_plan_gripper_open, cmd_plan_inter, cmd_plan_grasp, cmd_plan_gripper_close)

                else:
                    # Fallback for failed grasp planning
                    print(f"[Warn] Final grasp MG failed for env {s}; using simple fallback.")
                    fallback_steps = 10
                    self.cmd_plan[s] = self._make_hold_from_single(self.current_js, s, fallback_steps)
                    self.gripper_effort_plan[s] = torch.zeros((fallback_steps, len(self._gripper_joint_ids)), device=self.device)
        else:
            # Fallback for all environments
            print("[Warn] Final grasp planning returned no successful paths; using simple fallbacks for all envs.")
            for s in range(self.num_envs):
                fallback_steps = 10
                self.cmd_plan[s] = self._make_hold_from_single(self.current_js, s, fallback_steps)
                self.gripper_effort_plan[s] = torch.zeros((fallback_steps, len(self._gripper_joint_ids)), device=self.device)

    def _combine_trajectory_segments(self, s, cmd_plan_gripper_open, cmd_plan_inter, cmd_plan_grasp, cmd_plan_gripper_close):
        """Combine trajectory segments into final command plan."""
        if (self.cmd_plan[s] is not None and cmd_plan_grasp is not None and 
            cmd_plan_inter is not None and cmd_plan_gripper_open is not None and cmd_plan_gripper_close is not None):
            
            pre_grasp_trimmed = self._trim_js7(self.cmd_plan[s])
            
            combined_position = torch.cat([
                cmd_plan_gripper_open.position,
                pre_grasp_trimmed.position,
                cmd_plan_inter.position,
                cmd_plan_grasp.position,
                cmd_plan_gripper_close.position,
            ], dim=0)
            combined_velocity = torch.cat([
                cmd_plan_gripper_open.velocity,
                pre_grasp_trimmed.velocity,
                cmd_plan_inter.velocity,
                cmd_plan_grasp.velocity,
                cmd_plan_gripper_close.velocity,
            ], dim=0)
            combined_acceleration = torch.cat([
                cmd_plan_gripper_open.acceleration,
                pre_grasp_trimmed.acceleration,
                cmd_plan_inter.acceleration,
                cmd_plan_grasp.acceleration,
                cmd_plan_gripper_close.acceleration,
            ], dim=0)
            combined_jerk = torch.cat([
                cmd_plan_gripper_open.jerk,
                pre_grasp_trimmed.jerk,
                cmd_plan_inter.jerk,
                cmd_plan_grasp.jerk,
                cmd_plan_gripper_close.jerk,
            ], dim=0)

            self.cmd_plan[s] = JointState(
                position=combined_position,
                velocity=combined_velocity,
                acceleration=combined_acceleration,
                jerk=combined_jerk,
                joint_names=self._joint_names
            )
        else:
            print(f"[Warn] Combining plans failed for env {s}; using fallback hold only.")
            self.cmd_plan[s] = self._make_hold_from_single(self.current_js, s, self.HOLD_STEPS_DEFAULT)

    def _generate_rotate_trajectory(self):
        """Generate rotation trajectory for wrist joint."""
        joint_dim = self.ARM_DOF
        
        # Set gripper to closed state for rotation (grasp object and rotate)
        for s in range(self.num_envs):
            self.set_gripper_state(s, True)  # True = closed
        
        # Generate rotation trajectory for each environment
        for s in range(self.num_envs):
            # Get current wrist joint position
            cur_wrist = self._asset.data.joint_pos[s, 6]  # wrist_3 joint
            goal_wrist = self.goal_rt[s]
            dir_wrist = self._rotate_dir[s]
            goal_rotation_amount = self._rotate_remaining[s].clone()
            remaining_rotation = abs(self._rotate_remaining[s])

            # Get joint limits
            joint_limit_upper = self._asset.data.joint_pos_limits[s, 6][1]
            joint_limit_lower = self._asset.data.joint_pos_limits[s, 6][0]

            rel_start_step = []
            rel_end_step = []

            traj = torch.empty((1), device=cur_wrist.device)
            while remaining_rotation > 1e-2:
                if dir_wrist > 0:
                    max_possible_rotation = joint_limit_upper - cur_wrist
                else:
                    max_possible_rotation = cur_wrist - joint_limit_lower
                
                rotation_this_segment = min(remaining_rotation, max_possible_rotation)
                # 메인 이동 세그먼트
                if rotation_this_segment > 1e-2:
                    start_pos = cur_wrist
                    end_pos = start_pos + dir_wrist * rotation_this_segment
                    step_size_with_dir = dir_wrist * self._rotate_step_size
                    
                    segment_traj = torch.arange(
                        start=start_pos,
                        end=end_pos + (step_size_with_dir * 0.5), 
                        step=step_size_with_dir, 
                        device=cur_wrist.device
                    )

                    if len(segment_traj) == 0 or torch.abs(segment_traj[-1] - end_pos) > 1e-6:
                        # np.append -> torch.cat (텐서를 합치는 함수)
                        segment_traj = torch.cat([segment_traj, end_pos.unsqueeze(0)])
                    
                    traj = torch.cat([traj, segment_traj], dim=0)
                    cur_wrist = end_pos
                    remaining_rotation -= rotation_this_segment        
                elif remaining_rotation > 1e-2:
                    # make release trajectory
                    rel_start_step.append(traj.shape[0]) 
                    start_pos = traj[-1]
                    end_pos = start_pos - dir_wrist * math.pi
                    step_size_with_dir = dir_wrist * self._rotate_step_size
                    segment_traj = torch.arange(
                        start=start_pos - step_size_with_dir, 
                        end=end_pos - (step_size_with_dir * 0.5), 
                        step=-step_size_with_dir, 
                        device=cur_wrist.device
                    )
                    if len(segment_traj) == 0 or torch.abs(segment_traj[-1] - end_pos) > 1e-6:
                        # np.append -> torch.cat (텐서를 합치는 함수)
                        segment_traj = torch.cat([segment_traj, end_pos.unsqueeze(0)])

                    traj = torch.cat([traj, segment_traj], dim=0)
                    cur_wrist = end_pos
                    remaining_rotation -= rotation_this_segment
                    rel_end_step.append(traj.shape[0])
            
            # trajectory를 JointState로 변환
            positions = traj.unsqueeze(1).repeat(1, joint_dim)
            initial_other_joints = self.current_js.position[s, :6]
            steps = positions.shape[0]
            positions[:, :6] = initial_other_joints.unsqueeze(0).repeat(steps, 1)
            
            velocities = torch.zeros((steps, joint_dim), device=self.device)
            accelerations = torch.zeros((steps, joint_dim), device=self.device)
            jerks = torch.zeros((steps, joint_dim), device=self.device)
            
            main_trajectory = JointState(
                position=positions,
                velocity=velocities,
                acceleration=accelerations,
                jerk=jerks,
                joint_names=self._joint_names
            )

            effort_plan = torch.zeros((positions.shape[0], len(self._gripper_joint_ids)), device=self.device)
            for i in range(len(rel_start_step)):
                effort_plan[rel_start_step[i]:rel_end_step[i], :] = self.GRIPPER_OPEN_POSITION
            self.gripper_effort_plan[s] = effort_plan
            self.cmd_plan[s] = main_trajectory


    def _add_gripper_control(self):
        """Add gripper control to all trajectories."""
        for s in range(self.num_envs):
            if self.cmd_plan[s] is None:
                continue
                
            # Safety guard: trim to 7-DOF
            try:
                if self.cmd_plan[s].position.shape[1] > self.ARM_DOF:
                    self.cmd_plan[s] = self._trim_js7(self.cmd_plan[s])
            except Exception:
                self.cmd_plan[s] = self._trim_js7(self.cmd_plan[s])
                
            T = self.cmd_plan[s].position.shape[0]
            
            # Generate gripper position trajectory based on skill type
            if self.gripper_effort_plan[s] is not None and self.gripper_effort_plan[s].shape[0] == T:
                # Skills with gripper_effort_plan (grasp, rotate with release)
                effort_traj = self.gripper_effort_plan[s]
                grip_pos_traj = torch.where(
                    effort_traj > 0.02,  # open threshold
                    torch.tensor([self.GRIPPER_OPEN_POSITION] * len(self._gripper_joint_ids), device=self.device),
                    torch.tensor([self.GRIPPER_CLOSE_POSITION] * len(self._gripper_joint_ids), device=self.device)
                )
            else:
                # Other skills: maintain current gripper state
                current_grip_pos = self._gripper_positions[s]
                grip_pos_traj = current_grip_pos.unsqueeze(0).repeat(T, 1)
                
            self.cmd_plan[s] = self._append_gripper_pos_to_js9(self.cmd_plan[s], grip_pos_traj)

    def curobo_mg_prev(self):
        """Initialize CuRobo motion generation."""
        world_cfg_list = []
        for _ in range(self.num_envs):
            world_cfg_list.append(WorldConfig())
            
        if self.solver == "mg":
            # Standard motion generation config
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg_list,
                self.tensor_args,
                collision_checker_type=CollisionCheckerType.MESH,
                trim_steps=[1,-2],
                interpolation_dt=0.0166,
                project_pose_to_goal_frame=False,
                ee_link_name="ee_link",
                high_precision=True,
                num_trajopt_seeds=4,
                num_graph_seeds=1,
            )
            
            # Local motion generation config (for grasp)
            motion_gen_config_local = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg_list,
                self.tensor_args,
                collision_checker_type=CollisionCheckerType.MESH,
                trim_steps=[1,-2],
                interpolation_dt=0.0166,
                fixed_iters_trajopt=True,
                finetune_trajopt_iters=100,
                project_pose_to_goal_frame=True,
                ee_link_name="ee_link",
                high_precision=False,
                num_trajopt_seeds=4,
                num_graph_seeds=1,
            )
            
            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen_local = MotionGen(motion_gen_config_local)
            self.world_model = self.motion_gen.world_collision
            self.plan_config = MotionGenPlanConfig(
                enable_graph=False, 
                max_attempts=60, 
                enable_finetune_trajopt=True, 
                finetune_attempts=5
            )

    def curobo_mg_compute(self, ik_goal):
        """Compute motion using standard motion generation."""
        print(f"[MG] ik_goal.position={ik_goal.position} quaternion={ik_goal.quaternion}", flush=True)
        print(f"[MG] current_js.position={self.current_js.position}", flush=True)
        retry_count = 3
        result = None

        for attempt in range(retry_count):
            self.last_plan_debug["standard_attempts"] += 1
            try:
                with time_limit(10):
                    result = self.motion_gen.plan_batch_env(self.current_js, ik_goal, self.plan_config.clone())
                
                if result.success.any():
                    break
                
                # If all failed, add noise and retry
                if attempt < retry_count - 1:
                    print(f"[Info] MG failed constraints (Attempt {attempt+1}). Retrying with noise...")
                    noise = torch.randn_like(ik_goal.position) * 0.01
                    ik_goal.position += noise
            
            except TimeoutError:
                print(f"[Warn] MG Timeout (Attempt {attempt+1}). Retrying with noise...")
                self.last_plan_debug["timeout"] = True
                if attempt < retry_count - 1:
                    noise = torch.randn_like(ik_goal.position) * 0.01
                    ik_goal.position += noise
                continue
            except Exception as e:
                print(f"[Error] MG Exception: {e}")
                self.last_plan_debug["exception"] = f"standard: {type(e).__name__}: {e}"
                break
        
        # Check result
        if result is not None and torch.count_nonzero(result.success) > 0:
            if self.num_envs == 1:
                trajs = result.get_interpolated_plan()
                if result.success.item():
                    self.cmd_plan[0] = self._trim_js7(self.motion_gen.get_full_js(trajs))
                else:
                    print("MG Failed; using fallback hold for env 0")
                    self.cmd_plan[0] = self._make_hold_from_single(self.current_js, 0, self.fallback_steps)
            else:
                trajs = result.get_paths()
                for i in range(len(result.success)):
                    if result.success[i]:
                        self.cmd_plan[i] = self._trim_js7(self.motion_gen.get_full_js(trajs[i]))
                    else:
                        print(f" MG Failed for environment {i}; using fallback hold")
                        self.cmd_plan[i] = self._make_hold_from_single(self.current_js, i, self.fallback_steps)
            self._set_last_plan_success(result.success)
            return result.success
        else:
            print("[Warn] MG failed for all envs (after retries); using fallback holds")
            for i in range(self.num_envs):
                self.cmd_plan[i] = self._make_hold_from_single(self.current_js, i, self.fallback_steps)
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._set_last_plan_success(success)
            return success

    def curobo_mg_local_compute(self, ik_goal):
        """Compute motion using local motion generation."""
        print(f"[MG_LOCAL] ik_goal.position={ik_goal.position} quaternion={ik_goal.quaternion}", flush=True)
        print(f"[MG_LOCAL] current_js.position={self.current_js.position}", flush=True)
        retry_count = 3
        result = None

        for attempt in range(retry_count):
            self.last_plan_debug["local_attempts"] += 1
            try:
                with time_limit(10):
                    result = self.motion_gen_local.plan_batch_env(self.current_js, ik_goal, self.plan_config.clone())
                
                if result.success.any():
                    break
                    
                # If all failed, add noise and retry
                if attempt < retry_count - 1:
                    print(f"[Info] MG (local) failed constraints (Attempt {attempt+1}). Retrying with noise...")
                    noise = torch.randn_like(ik_goal.position) * 0.01
                    ik_goal.position += noise

            except TimeoutError:
                print(f"[Warn] MG (local) Timeout (Attempt {attempt+1}). Retrying with noise...")
                self.last_plan_debug["timeout"] = True
                if attempt < retry_count - 1:
                    noise = torch.randn_like(ik_goal.position) * 0.01
                    ik_goal.position += noise
                continue
            except Exception as e:
                import traceback as _tb
                print(f"[Error] MG (local) Exception: {e}")
                _tb.print_exc()
                self.last_plan_debug["exception"] = f"local: {type(e).__name__}: {e}"
                break

        if result is not None and torch.count_nonzero(result.success) > 0:
            if self.num_envs == 1:
                trajs = result.get_interpolated_plan()
                if result.success.item():
                    self.cmd_plan[0] = self._trim_js7(self.motion_gen.get_full_js(trajs))
                else:
                    print("MG (local) Failed; using fallback hold for env 0")
                    self.cmd_plan[0] = self._make_hold_from_single(self.current_js, 0, self.fallback_steps)
            else:
                trajs = result.get_paths()
                for i in range(len(result.success)):
                    if result.success[i]:
                        self.cmd_plan[i] = self._trim_js7(self.motion_gen.get_full_js(trajs[i]))
                    else:
                        print(f" MG (local) Failed for environment {i}; using fallback hold")
                        self.cmd_plan[i] = self._make_hold_from_single(self.current_js, i, self.fallback_steps)
            self._set_last_plan_success(result.success)
            return result.success
        else:
            print("[Warn] MG (local) failed for all envs (after retries); using fallback holds")
            for i in range(self.num_envs):
                self.cmd_plan[i] = self._make_hold_from_single(self.current_js, i, self.fallback_steps)
            success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            self._set_last_plan_success(success)
            return success

    def curobo_mg_reset(self):
        """Reset motion generation state."""
        self.cmd_plan = [None for _ in range(self.num_envs)]
        self.ik_goal = None
        self.pose_cost_metric = None
        self.goal_rt = torch.zeros(self.num_envs, device=self.device)
        self.current_rt = torch.zeros(self.num_envs, device=self.device)
        self.gripper_effort_plan = [None for _ in range(self.num_envs)]

    def get_gripper_position_for_state(self, env_idx: int, target_state: bool) -> torch.Tensor:
        """Get gripper position for target state."""
        if target_state:  # closed
            return torch.tensor([self.GRIPPER_CLOSE_POSITION] * len(self._gripper_joint_ids), device=self.device)
        else:  # open
            return torch.tensor([self.GRIPPER_OPEN_POSITION] * len(self._gripper_joint_ids), device=self.device)
            
    def set_gripper_state(self, env_idx: int, target_state: bool):
        """Set gripper state for specific environment."""
        self._gripper_states[env_idx] = target_state
        self._gripper_positions[env_idx] = self.get_gripper_position_for_state(env_idx, target_state)
        
    def update_gripper_state_after_grasp(self):
        """Update gripper state to closed after grasp skill completion."""
        for s in range(self.num_envs):
            if self._skill_type == 1:  # Grasp skill
                self.set_gripper_state(s, True)  # True = closed

    def curobo_ik_step(self, ik_goal):
        """Solve IK for given goal."""
        result = self.motion_gen.ik_solver.solve_batch_env(ik_goal)
        ik_sol = result.js_solution.position.squeeze()
        return ik_sol

    def curobo_update_js(self):
        """Update current joint state."""
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)
        joint_acc = torch.zeros_like(joint_pos)
        self.current_js = JointState(
            position=joint_pos,
            velocity=joint_vel,
            acceleration=joint_acc,
            jerk=joint_vel,
            joint_names=self._joint_names,
        )

    def curobo_goal_setting(self, action):
        """Set IK goal based on skill type and action parameters."""
        if self.solver == "mg":
            if self._skill_type in {0,1}:
                # Reach and Grasp: direct pose target
                self.ik_goal = Pose(
                    position=self.tensor_args.to_device(action[:,:3].clone()),  # 복사본 사용
                    quaternion=self.tensor_args.to_device(action[:,3:7].clone())  # 복사본 사용
                )
            if self._skill_type == 2:
                # Push: linear motion from current pose
                ee_pose = self.motion_gen.compute_kinematics(self.current_js).ee_pose.clone()
                ee_pos_curr, ee_quat_curr = ee_pose.position, ee_pose.quaternion
                linear_heading = action[:,:3].clone()  # 복사본 사용
                distance = action[:,3].clone()  # 복사본 사용
                direction_vector = linear_heading / torch.norm(linear_heading, dim=1, keepdim=True)
                movement = direction_vector * distance.unsqueeze(1)
                ee_pos_final = ee_pos_curr + movement
                self.ik_goal = Pose(
                    position=self.tensor_args.to_device(ee_pos_final),
                    quaternion=self.tensor_args.to_device(ee_quat_curr),
                    normalize_rotation=True
                )
                projected_position = ee_pos_curr - ee_pos_final
                cost_list = torch.ones(6)
                cost_list[3:] = (projected_position < 0.005).int()
                self.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device(cost_list),
                    reach_full_pose=True,
                )
            if self._skill_type == 3:
                # Rotate: wrist joint target
                self.current_rt = self._asset.data.joint_pos[:,6]
                delta = action[:,0].clone()  # 복사본 사용
                self.goal_rt = self.current_rt + delta
                self._rotate_remaining = delta.abs()  # Total rotation amount needed
                self._rotate_dir = torch.sign(delta)  # Direction of rotation
            if self._skill_type in {4}:
                # Local motion: push/pull along current orientation
                ee_pose = self.motion_gen_local.compute_kinematics(self.current_js).ee_pose.clone()
                ee_pos_curr, ee_quat_curr = ee_pose.position, ee_pose.quaternion
                distance = action[0][0].clone()  # 복사본 사용
                self.ik_goal = self.compute_push_pull_ik_goal(ee_pos_curr, ee_quat_curr, distance)
                self.pose_cost_metric = PoseCostMetric(
                    hold_partial_pose=True,
                    hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
                    reach_full_pose=True,
                )

    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        was_1d = False
        if q.dim() == 1:
            q = q.unsqueeze(0)
            was_1d = True
        N = q.shape[0]
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]
        R = torch.zeros((N, 3, 3), device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)
        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 2] = 2 * (y * z - x * w)
        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)
        if was_1d:
            R = R.squeeze(0)
        return R

    def compute_push_pull_ik_goal(self, ee_pos_curr, ee_quat_curr, distance, local_axis=torch.tensor([0.0, 0.0, 1.0])):
        """Compute IK goal for push/pull motion along local axis."""
        local_axis = local_axis.to(ee_pos_curr.device)
        R = self.quaternion_to_rotation_matrix(ee_quat_curr)
        world_direction = torch.matmul(R, local_axis)
        world_direction = world_direction / torch.norm(world_direction)
        target_pos = ee_pos_curr + distance * world_direction
        ik_goal = Pose(position=target_pos, quaternion=ee_quat_curr)
        return ik_goal

    def update_gripper_cmd(self, counter: int):
        """Update gripper command based on effort plan."""
        step_idx = max(int(counter), 0)
        if self.num_envs > 1:
            for i in range(self.num_envs):
                if self.gripper_effort_plan[i] is not None and step_idx < self.gripper_effort_plan[i].shape[0]:
                    self.gripper_effort_cmd[i, :] = self.gripper_effort_plan[i][step_idx]
                elif self.gripper_effort_plan[i] is not None:
                    self.gripper_effort_cmd[i, :] = self.gripper_effort_plan[i][-1]
                else:
                    self.gripper_effort_cmd[i, :] = 0.0
        else:
            if self.gripper_effort_plan[0] is not None and step_idx < self.gripper_effort_plan[0].shape[0]:
                self.gripper_effort_cmd[0, :] = self.gripper_effort_plan[0][step_idx]
            elif self.gripper_effort_plan[0] is not None:
                self.gripper_effort_cmd[0, :] = self.gripper_effort_plan[0][-1]
            else:
                self.gripper_effort_cmd[0, :] = 0.0

    def _make_hold_from_single(self, js_batched: JointState, env_idx: int, steps: int) -> JointState:
        """Create hold trajectory from single environment joint state."""
        pos = js_batched.position[env_idx, :self.ARM_DOF].unsqueeze(0).repeat(steps, 1)
        vel = torch.zeros_like(pos)
        acc = torch.zeros_like(pos)
        jerk = torch.zeros_like(pos)
        return JointState(position=pos, velocity=vel, acceleration=acc, jerk=jerk, joint_names=self._joint_names)

    def _make_hold_from_js(self, js: JointState, steps: int) -> JointState:
        """Create hold trajectory from joint state."""
        pos = js.position[:self.ARM_DOF].unsqueeze(0).repeat(steps, 1) if js.position.dim() == 1 else js.position[:, :self.ARM_DOF].unsqueeze(0).repeat(steps, 1)
        vel = torch.zeros_like(pos)
        acc = torch.zeros_like(pos)
        jerk = torch.zeros_like(pos)
        return JointState(position=pos, velocity=vel, acceleration=acc, jerk=jerk, joint_names=self._joint_names)

    def _trim_js7(self, js: JointState) -> JointState:
        """Trim joint state to 7-DOF arm."""
        if js.position.dim() == 1:
            pos = js.position[:self.ARM_DOF].unsqueeze(0)
            vel = js.velocity[:self.ARM_DOF].unsqueeze(0) if js.velocity is not None else torch.zeros_like(pos)
            acc = js.acceleration[:self.ARM_DOF].unsqueeze(0) if js.acceleration is not None else torch.zeros_like(pos)
            jerk = js.jerk[:self.ARM_DOF].unsqueeze(0) if getattr(js, 'jerk', None) is not None else torch.zeros_like(pos)
        else:
            pos = js.position[:, :self.ARM_DOF]
            vel = js.velocity[:, :self.ARM_DOF] if js.velocity is not None else torch.zeros_like(pos)
            acc = js.acceleration[:, :self.ARM_DOF] if js.acceleration is not None else torch.zeros_like(pos)
            jerk = js.jerk[:, :self.ARM_DOF] if getattr(js, 'jerk', None) is not None else torch.zeros_like(pos)
        return JointState(position=pos, velocity=vel, acceleration=acc, jerk=jerk, joint_names=self._joint_names)

    def _append_gripper_pos_to_js9(self, js7: JointState, gripper_pos_traj: torch.Tensor) -> JointState:
        """Append gripper position to 7-DOF joint state to create 9-DOF."""
        T = js7.position.shape[0]
        pos9 = torch.cat([js7.position[:, :self.ARM_DOF], gripper_pos_traj], dim=1)
        vel_base = js7.velocity[:, :self.ARM_DOF] if js7.velocity is not None else torch.zeros_like(js7.position[:, :self.ARM_DOF])
        acc_base = js7.acceleration[:, :self.ARM_DOF] if js7.acceleration is not None else torch.zeros_like(js7.position[:, :self.ARM_DOF])
        jerk_base = js7.jerk[:, :self.ARM_DOF] if getattr(js7, 'jerk', None) is not None else torch.zeros_like(js7.position[:, :self.ARM_DOF])
        # Set gripper velocity/acceleration/jerk to zero
        zeros2 = torch.zeros((T, 2), device=pos9.device)
        vel9 = torch.cat([vel_base, zeros2], dim=1)
        acc9 = torch.cat([acc_base, zeros2], dim=1)
        jerk9 = torch.cat([jerk_base, zeros2], dim=1)
        return JointState(position=pos9, velocity=vel9, acceleration=acc9, jerk=jerk9, joint_names=self._joint_names)
    
