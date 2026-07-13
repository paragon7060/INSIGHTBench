# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as f
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from custom_lab.managers.action_counter_manager import ActionCounterTerm

# Third Party
import carb
import numpy as np

_CUROBO_LOADED = False


def _require_curobo() -> None:
    """Load collection-only CuRobo dependencies when the action is constructed."""
    global _CUROBO_LOADED
    if _CUROBO_LOADED:
        return

    try:
        from curobo.geom.sdf.world import CollisionCheckerType as _CollisionCheckerType
        from curobo.geom.types import WorldConfig as _WorldConfig
        from curobo.rollout.rollout_base import Goal as _Goal
        from curobo.types.base import TensorDeviceType as _TensorDeviceType
        from curobo.types.camera import CameraObservation as _CameraObservation
        from curobo.types.math import Pose as _Pose
        from curobo.types.state import JointState as _JointState
        from curobo.util.logger import log_error as _log_error
        from curobo.util.logger import setup_curobo_logger as _setup_curobo_logger
        from curobo.util.usd_helper import UsdHelper as _UsdHelper
        from curobo.util_file import get_assets_path as _get_assets_path
        from curobo.util_file import get_filename as _get_filename
        from curobo.util_file import get_path_of_dir as _get_path_of_dir
        from curobo.util_file import get_robot_configs_path as _get_robot_configs_path
        from curobo.util_file import get_world_configs_path as _get_world_configs_path
        from curobo.util_file import join_path as _join_path
        from curobo.util_file import load_yaml as _load_yaml
        from curobo.wrap.reacher.motion_gen import MotionGen as _MotionGen
        from curobo.wrap.reacher.motion_gen import MotionGenConfig as _MotionGenConfig
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig as _MotionGenPlanConfig
        from curobo.wrap.reacher.motion_gen import PoseCostMetric as _PoseCostMetric
        from curobo.wrap.reacher.mpc import MpcSolver as _MpcSolver
        from curobo.wrap.reacher.mpc import MpcSolverConfig as _MpcSolverConfig
    except ModuleNotFoundError as exc:
        if exc.name == "curobo" or (exc.name is not None and exc.name.startswith("curobo.")):
            raise ModuleNotFoundError(
                "CuroboInteractionAction requires CuRobo. Install the full "
                "InsightBench environment with install.sh for data collection."
            ) from exc
        raise

    globals().update(
        {
            "CameraObservation": _CameraObservation,
            "CollisionCheckerType": _CollisionCheckerType,
            "Goal": _Goal,
            "JointState": _JointState,
            "MotionGen": _MotionGen,
            "MotionGenConfig": _MotionGenConfig,
            "MotionGenPlanConfig": _MotionGenPlanConfig,
            "MpcSolver": _MpcSolver,
            "MpcSolverConfig": _MpcSolverConfig,
            "Pose": _Pose,
            "PoseCostMetric": _PoseCostMetric,
            "TensorDeviceType": _TensorDeviceType,
            "UsdHelper": _UsdHelper,
            "WorldConfig": _WorldConfig,
            "get_assets_path": _get_assets_path,
            "get_filename": _get_filename,
            "get_path_of_dir": _get_path_of_dir,
            "get_robot_configs_path": _get_robot_configs_path,
            "get_world_configs_path": _get_world_configs_path,
            "join_path": _join_path,
            "load_yaml": _load_yaml,
            "log_error": _log_error,
            "setup_curobo_logger": _setup_curobo_logger,
        }
    )
    _CUROBO_LOADED = True

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from custom_lab.envs.mdp.actions import actions_cfg


class CuroboInteractionAction(ActionCounterTerm):
    """
    """

    cfg: actions_cfg.CuroboActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor
    """The scaling factor applied to the input action. Shape is (1, action_dim)."""

    def __init__(self, cfg: actions_cfg.HybridDifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv):
        _require_curobo()
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
        # self._gripper_open_command = torch.tensor([[0.0400, 0.0400]], device=self.device)
        # self._gripper_close_command = torch.tensor([[0.0000, 0.0000]], device=self.device)
        # self._gripper_close_effort = torch.tensor([[-0.0400, -0.0400]], device=self.device)
        # pre-allocate the two commands
        self._gripper_open_effort  = torch.tensor([[0.5, 0.5]], device=self.device).repeat(self.num_envs, 1)
        self._gripper_close_effort = torch.tensor([[-15.0, -15.0]], device=self.device).repeat(self.num_envs, 1)

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
        self.ik_goal = [None for _ in range(self.num_envs)]
        self.solver = "mg" # "mg" or "mpc"
        self.pose_cost_metric = None

        self.use_debug_draw = True
        self.gripper_state = torch.zeros((self.num_envs, 1), dtype=torch.bool, device=self.device) # 1 : open, 2 : close

        self._rotate_remaining = torch.zeros(self.num_envs, device=self.device)
        self._rotate_dir       = torch.zeros(self.num_envs, device=self.device)
        self._release_phase       = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rotate_step_size = 0.3
        self._release_target = torch.zeros(self.num_envs, device=self.device)

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
        self._joint_pos_des[:, :] = self.current_js.position
        self._asset.set_joint_position_target(self._joint_pos_des, joint_ids=self._joint_ids)

        self.curobo_goal_setting(self._processed_actions[:,1:])
        if self._skill_type == 0:
            print("Approach skill : Move to the target")
            self.curobo_mg_compute(self.ik_goal)
        if self._skill_type == 1:
            joint_dim = 7
            print("Grasp skill : Open gripper and move to the target and close gripper")
            # compute pre-grasp point
            offset_distance = -0.12
            pre_grasp_goal_pose = self.compute_push_pull_ik_goal(
                ee_pos_curr = self.ik_goal.position.squeeze(0),
                ee_quat_curr = self.ik_goal.quaternion.squeeze(0),
                distance = offset_distance,
            )

            # 2) Pre‐grasp MG 실행 (batch 크기는 num_envs 그대로)
            pre_grasp_success = self.curobo_mg_compute(pre_grasp_goal_pose)
            # pre_grasp_success: (num_envs,) bool 텐서

            # 3) “현재 joint state” 와 “현재 end‐effector pose” 를 미리 얻어 둔다.
            #    (pre‐grasp 실패 env에 이 값을 채워 줄 예정)
            ee_pose_all = self.motion_gen.compute_kinematics(self.current_js).ee_pose.clone()
            ee_pos_all  = ee_pose_all.position    # shape = (num_envs, 3)
            ee_quat_all = ee_pose_all.quaternion  # shape = (num_envs, 4)

            # 4) Pre‐grasp 성공 여부에 따라 js_pregrasp_final을 채운다.
            #    - 성공한 env: cmd_plan[s] 마지막 JointState에서 값을 가져옴
            #    - 실패한 env: 그냥 “현재 joint state” 를 넣어 주면 되므로 current_js.position 사용
            pregrasp_final_positions     = torch.zeros((self.num_envs, joint_dim), device=self.tensor_args.device)
            pregrasp_final_velocities    = torch.zeros((self.num_envs, joint_dim), device=self.tensor_args.device)
            pregrasp_final_accelerations = torch.zeros((self.num_envs, joint_dim), device=self.tensor_args.device)

            for s in range(self.num_envs):
                if pre_grasp_success[s] and self.cmd_plan[s] is not None and len(self.cmd_plan[s].position) > 0:
                    # “Pre‐grasp” 성공한 env: 마지막 trajectory에서 joint state 추출
                    last = self.cmd_plan[s][-1]
                    pregrasp_final_positions[s, :]     = last.position[:joint_dim].clone().to(self.tensor_args.device)
                    pregrasp_final_velocities[s, :]    = last.velocity[:joint_dim].clone().to(self.tensor_args.device)
                    pregrasp_final_accelerations[s, :] = last.acceleration[:joint_dim].clone().to(self.tensor_args.device)
                else:
                    # “Pre‐grasp” 실패한 env: 그냥 현재 joint state를 사용
                    print(f"Pre-grasp MG failed for env{s}")
                    pregrasp_final_positions[s, :]     = self.current_js.position[s, :].clone().to(self.tensor_args.device)
                    pregrasp_final_velocities[s, :]    = torch.zeros(joint_dim, device=self.tensor_args.device)
                    pregrasp_final_accelerations[s, :] = torch.zeros(joint_dim, device=self.tensor_args.device)

            js_pregrasp_final = JointState(
                position=pregrasp_final_positions,
                velocity=pregrasp_final_velocities,
                acceleration=pregrasp_final_accelerations,
                joint_names=self._joint_names,
            )

            # 5) PoseCostMetric 설정 (필요 시)
            self.pose_cost_metric = PoseCostMetric(
                hold_partial_pose=True,
                hold_vec_weight=self.motion_gen.tensor_args.to_device([1, 1, 1, 1, 1, 0]),
                reach_full_pose=True,
            )
            self.plan_config.pose_cost_metric = self.pose_cost_metric

            # 6) “Pre‐grasp” 실패한 env에 대해서는 self.ik_goal 을 “현재 end‐effector pose” 로 덮어쓴다.
            #    (이렇게 하면 final‐grasp MG 호출 시 start==goal이 되어 trivial한 trajectory만 반환)
            for s in range(self.num_envs):
                if not pre_grasp_success[s]:
                    # 현재 EE pose를 ik_goal[s]에 덮어쓰기
                    self.ik_goal.position[s]    = ee_pos_all[s]
                    self.ik_goal.quaternion[s]  = ee_quat_all[s]

            # 7) 모든 env(batch) 크기를 그대로 유지한 채로 final‐grasp MG 수행
            try:
                result_grasp = self.motion_gen_local.plan_batch_env(
                    js_pregrasp_final,
                    self.ik_goal.clone(),
                    self.plan_config.clone(),
                )
            except Exception as e:
                print(f"[Error] Final grasp planning exception: {e}")
                result_grasp = None

            # 8) 결과를 전체 num_envs 길이의 리스트로 매핑
            cmd_plan_grasp = [None] * self.num_envs
            cmd_plan_inter = [None] * self.num_envs  # Hold 단계용 intermediate
            hold_steps = 60

            if result_grasp is not None and torch.count_nonzero(result_grasp.success) > 0:
                if self.num_envs > 1:
                    pos_trajs = result_grasp.get_paths()
                else:
                    pos_trajs = result_grasp.get_interpolated_plan()

                for s in range(self.num_envs):
                    if result_grasp.success[s]:
                        # Final‐grasp MG가 성공한 env: trajectory 생성
                        if self.num_envs > 1:
                            cmd_plan_grasp[s] = self.motion_gen.get_full_js(pos_trajs[s])
                        else:
                            cmd_plan_grasp[s] = self.motion_gen.get_full_js(pos_trajs)

                        # Hold 단계용으로 “Pre‐grasp”에서 끝난 위치를 복사하여 중간 정지 모션 생성
                        if pre_grasp_success[s] and self.cmd_plan[s] is not None and len(self.cmd_plan[s].position) > 0:
                            last_js = JointState(
                                position=self.cmd_plan[s][-1].position.clone(),
                                velocity=torch.zeros_like(self.cmd_plan[s][-1].velocity.clone()),
                                acceleration=torch.zeros_like(self.cmd_plan[s][-1].acceleration.clone()),
                                jerk=torch.zeros_like(self.cmd_plan[s][-1].jerk.clone())
                                     if self.cmd_plan[s].jerk is not None
                                     else torch.zeros_like(self.cmd_plan[s][-1].velocity.clone()),
                                joint_names=self._joint_names
                            )
                        else:
                            # “Pre‐grasp” 실패 env: current_js 기준으로 trivial Hold 준비
                            last_js = JointState(
                                position=self.current_js.position[s].clone(),
                                velocity=torch.zeros_like(self.current_js.velocity[s].clone()),
                                acceleration=torch.zeros_like(self.current_js.acceleration[s].clone()),
                                jerk=torch.zeros_like(self.current_js.velocity[s].clone()),
                                joint_names=self._joint_names
                            )

                        intermediate_positions    = last_js.position.unsqueeze(0).repeat(hold_steps, 1)
                        intermediate_velocities   = last_js.velocity.unsqueeze(0).repeat(hold_steps, 1)
                        intermediate_accelerations= last_js.acceleration.unsqueeze(0).repeat(hold_steps, 1)
                        intermediate_jerks         = last_js.jerk.unsqueeze(0).repeat(hold_steps, 1)

                        cmd_plan_inter[s] = JointState(
                            position=intermediate_positions,
                            velocity=intermediate_velocities,
                            acceleration=intermediate_accelerations,
                            jerk=intermediate_jerks,
                            joint_names=self._joint_names
                        )

                    else:
                        # MG가 실패한 env: None으로 남겨둔다
                        cmd_plan_grasp[s] = None
                        cmd_plan_inter[s] = None
                        # print(f"[Env {s}] Final grasp MG failed; status={result_grasp.status[s] if result_grasp is not None else 'No result'}")
            else:
                print("Final grasp planning returned no successful paths or planning entirely failed.")
                # cmd_plan_grasp, cmd_plan_inter 모두 None 상태

            # 9) “Pre‐grasp + Hold + Final‐grasp” trajectory를 합쳐서 self.cmd_plan[s]으로 최종 저장
            for s in range(self.num_envs):
                if self.cmd_plan[s] is not None and cmd_plan_grasp[s] is not None and cmd_plan_inter[s] is not None:
                    combined_position = torch.cat([
                        self.cmd_plan[s].position,
                        cmd_plan_inter[s].position,
                        cmd_plan_grasp[s].position
                    ], dim=0)
                    combined_velocity = torch.cat([
                        self.cmd_plan[s].velocity,
                        cmd_plan_inter[s].velocity,
                        cmd_plan_grasp[s].velocity
                    ], dim=0)
                    combined_acceleration = torch.cat([
                        self.cmd_plan[s].acceleration,
                        cmd_plan_inter[s].acceleration,
                        cmd_plan_grasp[s].acceleration
                    ], dim=0)
                    combined_jerk = torch.cat([
                        self.cmd_plan[s].jerk,
                        cmd_plan_inter[s].jerk,
                        cmd_plan_grasp[s].jerk
                    ], dim=0)

                    self.cmd_plan[s] = JointState(
                        position=combined_position,
                        velocity=combined_velocity,
                        acceleration=combined_acceleration,
                        jerk=combined_jerk,
                        joint_names=self._joint_names
                    )
                else:
                    # 하나라도 결손된 env는 애초에 cmd_plan[s]을 None으로 둬서 apply_actions 단계에서 무시됨
                    self.cmd_plan[s] = None

        if self._skill_type == 2:
            print("Constrained movement skill")
            print("Constrained: Holding tool Orientation and automatically calculate pose_cost_metric")
            self.plan_config.pose_cost_metric = self.pose_cost_metric
            self.curobo_mg_compute(self.ik_goal)
        if self._skill_type == 3:
            print("Rotate skill")
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
            if counter < decimation/6:
                self.gripper_open()
            # counter: d/4 ~ 3/4d : IK move
            elif counter < decimation*5/6:
                self.curobo_mg_step(counter, decimation/5)
            else:
                self.gripper_close()
        elif self._skill_type == 2: # Constrained Movement
            self.curobo_mg_step(counter)
        elif self._skill_type == 3: # Rotate
            cur = self._asset.data.joint_pos[:, 6]  # (num_envs,)
            goal = self.goal_rt
            dir = self._rotate_dir
            # 2) 풀(back-off) 단계가 아닌 env 에 대해 next 목표값 계산
            nomask = ~self._release_phase
            next_val = cur + self._rotate_dir * self._rotate_step_size  # (num_envs,)

            # 3) limit 체크
            over  = next_val > self._asset.data.joint_limits[0,6][1]
            under = next_val < self._asset.data.joint_limits[0,6][0]
            to_release = nomask & (over | under)

            # 목표 각도 도달
            mask_neg = (dir < 0) & (next_val < goal) & nomask & ~to_release
            mask_pos = (dir > 0) & (next_val > goal) & nomask & ~to_release
            mask_step = nomask & ~(mask_neg | mask_pos | to_release)

            # target 계산
            target = next_val.clone()
            target[mask_neg] = goal[mask_neg]
            target[mask_pos] = goal[mask_pos]

            # 3. 일반 회전 step
            if mask_step.any():
                idxs = mask_step.nonzero(as_tuple=True)[0]
                target = next_val[mask_step]
                self._asset.set_joint_effort_target(self._gripper_close_effort[0], joint_ids=self._gripper_joint_ids, env_ids=idxs)
                self._asset.set_joint_position_target(target, joint_ids=[6]*len(idxs), env_ids=idxs)
                self._rotate_remaining[mask_step] -= self._rotate_step_size

            # 4. 목표 각도에 맞춘 env
            if (mask_neg | mask_pos).any():
                idxs_goal = (mask_neg | mask_pos).nonzero(as_tuple=True)[0]
                self._asset.set_joint_effort_target(self._gripper_close_effort[0], joint_ids=self._gripper_joint_ids, env_ids=idxs_goal)
                self._asset.set_joint_position_target(goal[idxs_goal], joint_ids=[6]*len(idxs_goal), env_ids=idxs_goal)
                self._rotate_remaining[mask_neg | mask_pos] -= (goal[mask_neg | mask_pos] - cur[mask_neg | mask_pos]).abs()

            # 5) 한계 도달(env마다) → gripper open and 3.14 release
            if to_release.any():
                self._release_phase[to_release] = True
                slack = 3.1415 * (-self._rotate_dir[to_release])  # 1rad 뒤로
                self._release_target[to_release] = cur[to_release] + slack

            # 6) release_phase env 처리
            if self._release_phase.any():
                idxs = to_release.nonzero(as_tuple=True)[0]
                self._asset.set_joint_effort_target(self._gripper_open_effort[0], joint_ids=self._gripper_joint_ids, env_ids=idxs)
                self._asset.set_joint_position_target(self._release_target[to_release], joint_ids=[6]*len(idxs), env_ids=idxs)

                rel_mask = self._release_phase
                cur_rel   = self._asset.data.joint_pos[:, 6][rel_mask]
                tgt_rel   = self._release_target[rel_mask]
                # release 완료 감지 (방향에 따라 비교)
                dir_rel   = self._rotate_dir[rel_mask]
                reached_upper = (dir_rel > 0) & (cur_rel <= tgt_rel + 1e-2)
                reached_lower = (dir_rel < 0) & (cur_rel >= tgt_rel - 1e-2)
                reached = reached_upper | reached_lower

                if reached.any():
                    # 6a) release_phase 해제
                    idxs = rel_mask.nonzero(as_tuple=True)[0][reached]
                    self._release_phase[idxs] = False
                    # 6b) 회전량에서 180° 차감
                    # self._rotate_remaining[idxs] = (self._rotate_remaining[idxs] - 3.1415).clamp(min=0.0)
                    self.goal_rt[idxs] -= self._rotate_dir[idxs] * 3.1415
                    # 이후 7)번에서 남은 회전량에 따라 다시 4)번 분기로 복귀

        elif self._skill_type in {4}: # Push/Pull
            self.curobo_mg_step(counter)
        elif self._skill_type in {5}:
            self.gripper_open()
        elif self._skill_type in {6}:
            self.gripper_close()


    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        # 1) 기존 reset 동작 실행
        super().reset(env_ids)

        # 2) 추가적으로 초기화할 텐서 예시
        #    예: gripper state 초기화
        if env_ids is None:
            # 모든 env
            self.gripper_state[:] = True
        else:
            # slice 또는 list 인덱싱
            self.gripper_state[env_ids, 0] = True

        # 3) 필요하다면 다른 멤버도 초기화
        #    self.some_other_buffer[env_ids] = 0.0

        return {}
    
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

    def update_gripper_cmd(self):
        mask = self.gripper_state  # shape: (num_envs,1)
        self.gripper_effort_cmd = torch.where(mask, self._gripper_open_effort, self._gripper_close_effort)

    def gripper_open(self):
        """
        Opens the gripper by setting effort targets for the gripper joints.
        """
        # 1) Mark gripper as open
        self.gripper_state[:, 0] = True
        # 2) Build batched effort command based on state
        self.update_gripper_cmd()
        # 3) Apply effort to gripper joints
        self._asset.set_joint_effort_target(self.gripper_effort_cmd, joint_ids=self._gripper_joint_ids)

    def gripper_close(self):
        """
        Closes the gripper by setting effort targets for the gripper joints.
        """
        # 1) Mark gripper as closed
        self.gripper_state[:, 0] = False
        # 2) Build batched effort command based on state
        self.update_gripper_cmd()
        # 3) Apply effort to gripper joints
        self._asset.set_joint_effort_target(self.gripper_effort_cmd, joint_ids=self._gripper_joint_ids)

    def curobo_mg_prev(self):
        from curobo.geom.types import WorldConfig, Cuboid, Mesh, Capsule, Cylinder, Sphere

        world_cfg_list = []
        for i in range(self.num_envs):
            world_cfg = WorldConfig()
            world_cfg_list.append(world_cfg)
        # # collision world 생성
        # # world_cfg는 기존 코드에서 로드한 collision world config 객체라고 가정
        # # CHECK : need to change world_cfg for other scenario
        # for i in range(self.num_envs):
        #     world_cfg = WorldConfig.from_dict({
        #         "cuboid": {
        #             "table": {
        #                 "dims": [0.35, 0.6, 0.33],
        #                 "pose": [0.9, 0, 0.6, 1, 0, 0, 0],
        #             }
        #         }
        #     })            
        #     # world_cfg = WorldConfig.from_dict({
        #     #     "cuboid": {
        #     #         "table": {
        #     #             "dims": [0.6, 0.75, 0.8],
        #     #             "pose": [1, 0, 0.4, 1, 0, 0, 0],
        #     #         }
        #     #     }
        #     # }) # cabinet
        #     world_cfg_list.append(world_cfg)

        # motion_gen
        if self.solver == "mg":
            motion_gen_config = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg_list,
                self.tensor_args,
                collision_checker_type=CollisionCheckerType.MESH,
                trim_steps=[1,-2],
                interpolation_dt=0.0333,
                project_pose_to_goal_frame=False,
                ee_link_name="ee_link", #"right_gripper",
                high_precision=True,
            )
            motion_gen_config_local = MotionGenConfig.load_from_robot_config(
                self.robot_cfg,
                world_cfg_list,
                self.tensor_args,
                collision_checker_type=CollisionCheckerType.MESH,
                trim_steps=[1,-2],
                interpolation_dt=0.0333,
                # collision_activation_distance=0.025,
                fixed_iters_trajopt=True,
                finetune_trajopt_iters=300,
                project_pose_to_goal_frame=True,
                ee_link_name="ee_link", #"right_gripper",
                high_precision=True
            )
            self.motion_gen = MotionGen(motion_gen_config)
            self.motion_gen_local = MotionGen(motion_gen_config_local)

            # if not self.reactive_mode:
            #     print("warming up...")
            #     self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
            #     self.motion_gen_local.warmup(enable_graph=True, warmup_js_trajopt=False)

            self.world_model = self.motion_gen.world_collision
            # print("Curobo is Ready")

            self.plan_config = MotionGenPlanConfig(
                enable_graph=False, max_attempts=60, enable_finetune_trajopt=True, finetune_attempts=5,
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
            # Rotate Skill
            if self._skill_type == 3:
                self.current_rt = self._asset.data.joint_pos[:,6]
                self.goal_rt = self.current_rt + self._processed_actions[:,1]
                delta = self._processed_actions[:,1]      # shape (num_envs,)
                self._rotate_remaining = delta.abs()
                self._rotate_dir       = torch.sign(delta)  # +1 or -1 or 0
                self._release_phase[:]    = False              # 모두 초기화

                # if abs(self.goal_rt) > 2.8973:
                #     self.rotate_js = self.current_js.clone()  
                #     self.rotate_js.position[:,6] = torch.where(self._rotate_dir, self._asset.data.joint_limits[0,6][0],self._asset.data.joint_limits[0,6][1])
                #     ee_pose = self.motion_gen_local.compute_kinematics(self.rotate_js).ee_pose.clone()
                #     ee_pos_curr, ee_quat_curr = ee_pose.position, ee_pose.quaternion
                #     distance = -0.1
                #     self.ik_goal = self.compute_push_pull_ik_goal(ee_pos_curr, ee_quat_curr, distance)
                # else:
                #     pass

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
        result = self.motion_gen.plan_batch_env(self.current_js, ik_goal, self.plan_config.clone())

        if torch.count_nonzero(result.success) > 0:
            if self.num_envs == 1:
                trajs = result.get_interpolated_plan()
                if result.success.item():
                    self.cmd_plan[0] = self.motion_gen.get_full_js(trajs)
                else:
                    print("MG Failed")
                    print(result.status)
            else:
                trajs = result.get_paths()

                for i in range(len(result.success)):
                    if result.success[i]:
                        self.cmd_plan[i] = self.motion_gen.get_full_js(trajs[i])
                    else:
                        print(f" MG Failed for environment {i}")
                        print(result.status)
        return result.success

    def curobo_mg_local_compute(self, ik_goal):
        '''
        MotionGeneration
        self.cmd_plan 에 생성된 trajectory 입력
        '''
        # self.curobo_update_world()
        result = self.motion_gen_local.plan_batch_env(self.current_js, ik_goal, self.plan_config.clone())

        if torch.count_nonzero(result.success) > 0:
            if self.num_envs == 1:
                trajs = result.get_interpolated_plan()
                if result.success.item():
                    self.cmd_plan[0] = self.motion_gen.get_full_js(trajs)
                else:
                    print("MG Failed")
                    print(result.status)
            else:
                trajs = result.get_paths()

                for i in range(len(result.success)):
                    if result.success[i]:
                        self.cmd_plan[i] = self.motion_gen.get_full_js(trajs[i])
                    else:
                        print(f" MG Failed for environment {i}")
                        print(result.status)

    def curobo_mg_step(self, counter, start=0):
        '''
        self.cmd_plan 에 입력된 trajectory 를 따라서 counter에 맞게 robot control
        counter start : 
        '''
        self.update_gripper_cmd()
        joint_pos_des = self._asset.data.joint_pos[:, self._joint_ids]
        joint_vel_des = torch.zeros_like(self._asset.data.joint_vel[:, self._joint_ids])
        joint_acc_des = torch.zeros_like(self._asset.data.joint_acc[:, self._joint_ids])
        # for s in range(len(self.cmd_plan)):
        counter -= start
        counter = int(counter) -1 if counter >= 0 else 0
        if self.num_envs > 1:
            for i in range(self.num_envs):
                if self.cmd_plan[i] is not None and counter < len(self.cmd_plan[i].position):
                    joint_pos_des[i, :] = self.cmd_plan[i][counter].position[:7].clone()
                    joint_vel_des[i, :] = self.cmd_plan[i][counter].velocity[:7].clone()
                    joint_acc_des[i, :] = self.cmd_plan[i][counter].acceleration[:7].clone()
                elif self.cmd_plan[i] is not None:
                    joint_pos_des[i, :] = self.cmd_plan[i][-1,:].position[:7].clone()
                    # joint_vel_des[i, :] = self.cmd_plan[-1,:].velocity[:7].clone()
                    # joint_acc_des[i, :] = self.cmd_plan[-1,:].acceleration[:7].clone()
        else:
            if self.cmd_plan[0] is not None and counter < len(self.cmd_plan[0].position):
                joint_pos_des[0, :] = self.cmd_plan[0][counter].position[:7].clone()
                joint_vel_des[0, :] = self.cmd_plan[0][counter].velocity[:7].clone()
                joint_acc_des[0, :] = self.cmd_plan[0][counter].acceleration[:7].clone()
            elif self.cmd_plan[0] is not None:
                joint_pos_des[0, :] = self.cmd_plan[0][-1,:].position[:7].clone()
                # joint_vel_des[0, :] = self.cmd_plan[-1,:].velocity[:9].clone()
                # joint_acc_des[0, :] = self.cmd_plan[-1,:].acceleration[:9].clone()
        self._asset.set_joint_position_target(joint_pos_des, joint_ids=self._joint_ids)
        self._asset.set_joint_velocity_target(joint_vel_des, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(joint_acc_des, joint_ids=self._joint_ids)
        self._asset.set_joint_effort_target(self.gripper_effort_cmd, joint_ids=self._gripper_joint_ids)



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
            jerk=joint_vel, #* 1.0,
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

    def quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """
        Converts a batch of quaternions to a batch of rotation matrices.
        q: torch.Tensor of shape (N, 4) or (4,) representing quaternions (w, x, y, z order).
           It's assumed that input quaternions are normalized.
        Returns:
            Rotation matrices R of shape (N, 3, 3) or (3, 3).
        """
        was_1d = False
        if q.dim() == 1:  # 입력이 (4,) 형태인 경우
            q = q.unsqueeze(0)  # (1, 4) 형태로 변경하여 일관된 처리
            was_1d = True
        
        # 이제 q는 (N, 4) 형태라고 가정
        N = q.shape[0]

        # w, x, y, z 컴포넌트 추출 (각각 shape: (N,))
        w = q[:, 0]
        x = q[:, 1]
        y = q[:, 2]
        z = q[:, 3]

        # 출력 텐서 초기화
        R = torch.zeros((N, 3, 3), device=q.device, dtype=q.dtype)

        # 회전 행렬 공식 적용 (단위 쿼터니언 가정)
        # 대각선 요소
        R[:, 0, 0] = 1 - 2 * (y * y + z * z)  # w*w + x*x - y*y - z*z (다른 공식 형태도 있음)
        R[:, 1, 1] = 1 - 2 * (x * x + z * z)  # w*w - x*x + y*y - z*z
        R[:, 2, 2] = 1 - 2 * (x * x + y * y)  # w*w - x*x - y*y + z*z

        # 비대각선 요소
        R[:, 0, 1] = 2 * (x * y - z * w)
        R[:, 0, 2] = 2 * (x * z + y * w)

        R[:, 1, 0] = 2 * (x * y + z * w)
        R[:, 1, 2] = 2 * (y * z - x * w)

        R[:, 2, 0] = 2 * (x * z - y * w)
        R[:, 2, 1] = 2 * (y * z + x * w)

        if was_1d:
            R = R.squeeze(0)  # 원래 입력이 (4,)였으면 (3,3)으로 복원
            
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
    
