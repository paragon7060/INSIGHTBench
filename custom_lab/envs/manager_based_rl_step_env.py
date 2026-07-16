# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations
from dataclasses import MISSING

import gymnasium as gym
import math
import numpy as np
import time
import torch
from collections.abc import Sequence
from typing import Any, ClassVar

from isaacsim.core.version import get_version
from isaaclab.managers import ActionManager, ObservationManager, EventManager, CommandManager, TerminationManager, RewardManager, CurriculumManager, RecorderManager
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, quat_inv, quat_conjugate, euler_xyz_from_quat
from custom_lab.managers.action_counter_manager import ActionCounterManager

from isaaclab.envs.common import VecEnvStepReturn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.manager_based_rl_env_cfg import ManagerBasedRLEnvCfg

def wrap_to_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + torch.pi) % (2 * torch.pi) - torch.pi

def near_0_or_pi(a: torch.Tensor, tol_rad: float) -> torch.Tensor:
    """각도 a가 0 또는 π에 가깝다면 True (랩핑 고려)"""
    d0  = torch.abs(wrap_to_pi(a))            # 0과의 거리
    dpi = torch.abs(wrap_to_pi(a - torch.pi)) # π와의 거리
    return torch.minimum(d0, dpi) < tol_rad

def near_equal_mod_2pi(a: torch.Tensor, b: torch.Tensor, tol_rad: float) -> torch.Tensor:
    """a ≈ b (mod 2π) 이면 True"""
    return torch.abs(wrap_to_pi(a + b)) < tol_rad

class ManagerBasedRLStepEnv(ManagerBasedRLEnv):

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""
    metadata: ClassVar[dict[str, Any]] = {
        "render_modes": [None, "human", "rgb_array"],
        "isaac_sim_version": get_version(),
    }
    """Metadata for the environment."""

    cfg: ManagerBasedRLEnvCfg
    """Configuration for the environment."""
    
    scene_key: str | None

    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, scene_key: str | None = None,  **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode)
        self._bottle_effort_flag = None
        self._bottle_effort_value = None
        self._sim_counter_per_action = 0
        # Collection smoke runs may set a per-action deadline. It remains disabled
        # for normal evaluation and production collection.
        self._collect_step_deadline_s: float | None = None
        self.scene_key = scene_key
        self._cache_articulations()
        # bottle effort flag/value 버퍼 초기화
        self.asset_ori_delta = torch.zeros(self.num_envs, 3, device=self.device)

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # -- action manager
        self.action_manager = ActionCounterManager(self.cfg.actions, self)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- event manager
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def _cache_articulations(self):
        self._art_handles = self.scene.articulations.copy()
        self._art_names = set(self._art_handles.keys())
        if 'door' in self._art_names:
            art = self._art_handles["door"]
            art.write_joint_friction_coefficient_to_sim(5e10, joint_ids=[0])
            self._released = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.scene.device)
            joint_1_effort = -1 * torch.ones(self.num_envs, device=art.device)
            joint_2_effort = -1 * torch.ones(self.num_envs, device=art.device)
            art.set_joint_effort_target(joint_1_effort.unsqueeze(1), joint_ids=[0])
            art.set_joint_effort_target(joint_2_effort.unsqueeze(1), joint_ids=[1])
        if 'bottle' in self._art_names:
            if self.scene_key in {"5g", "5a"}:
                # squeeze scene
                art = self._art_handles["bottle"]
                art.write_joint_friction_coefficient_to_sim(1e5,joint_ids=[0,1])
                # 예: 클래스 멤버로 보관
                self._released = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.scene.device)
                self.N_STEPS_FOR_RELEASE = 3
                self.condition_history = torch.zeros(self.num_envs,self.N_STEPS_FOR_RELEASE, dtype=torch.bool, device=self.device)

    def _check_articulations(self):
        if 'microwave' in self._art_names:
            art = self._art_handles["microwave"]
            # 조건: joint_pos[:, 1] > 0.005 → 0번 joint에 +1, 아니면 -1
            mask = art.data.joint_pos[:, 1] > 0.001
            effort = torch.where(mask, torch.ones(self.num_envs, device=art.device), -torch.ones(self.num_envs, device=art.device))
            # 0번 joint에 effort 할당
            art.set_joint_effort_target(effort.unsqueeze(1), joint_ids=[0])
        if 'door' in self._art_names:
            art = self._art_handles["door"]
            # [수정 1] 2번 관절(문고리, 인덱스 1)의 위치와 상한계 값을 가져옵니다. (기존 문제 수정)
            joint_2_pos = art.data.joint_pos[:, 1]
            joint_2_upper = art.data.joint_pos_limits[:, 1, 1]

            # 문고리가 30% 이상 돌아갔는지 확인하는 마스크를 생성합니다.
            is_pulled_mask = joint_2_pos > 0.3 * joint_2_upper
            is_not_pulled_mask = ~is_pulled_mask

            # -- 상태 변경이 필요한 환경 식별 --
            # 1. 잠금 해제할 환경: 문고리가 돌아갔고, 현재 잠겨있는 상태인 경우
            to_release_mask = is_pulled_mask & (~self._released)
            # 2. 다시 잠글 환경: 문고리가 원위치이고, 현재 풀려있는 상태인 경우
            to_lock_mask = is_not_pulled_mask & self._released

            # -- 잠금 해제 로직 --
            if to_release_mask.any():
                env_ids_to_release = torch.where(to_release_mask)[0]
                # 문(joint 0)의 마찰력을 낮춰 부드럽게 움직이도록 설정
                art.write_joint_friction_coefficient_to_sim(
                    joint_friction_coeff=20.0, 
                    joint_ids=[0],
                    env_ids=env_ids_to_release
                )
                # 상태를 '풀림'으로 업데이트
                self._released[env_ids_to_release] = True
            
            # -- 잠금 로직 --
            if to_lock_mask.any():
                env_ids_to_lock = torch.where(to_lock_mask)[0]
                # 문(joint 0)의 마찰력을 높여 움직이지 않도록 잠금
                art.write_joint_friction_coefficient_to_sim(
                    joint_friction_coeff=5e10, # 초기화 시 사용된 높은 마찰 계수
                    joint_ids=[0],
                    env_ids=env_ids_to_lock
                )
                # 상태를 '잠김'으로 업데이트
                self._released[env_ids_to_lock] = False

        if 'bottle' in self._art_names:
            art = self._art_handles["bottle"]
            joint2_pos = art.data.joint_pos[:, 1]  # (num_envs,)
            joint2_lower = art.data.joint_pos_limits[:, 1, 0]
            joint2_upper = art.data.joint_pos_limits[:, 1, 1]

            # joint_2를 0~1로 정규화
            joint2_norm = (joint2_pos - joint2_lower) / (joint2_upper - joint2_lower)

            joint1_upper = art.data.joint_pos_limits[:, 0, 1]
            joint1_target = torch.zeros(self.num_envs, device=art.device)
            joint1_effort = torch.zeros(self.num_envs, device=art.device)


            # 30% 이하: 0
            mask_30 = joint2_norm <= 0.3
            joint1_target[mask_30] = 0.0
            joint1_effort[mask_30] = -1.0  # 원하는 힘 값으로 조정

            # 30~70%: 선형 증가
            mask_linear = (joint2_norm > 0.3) & (joint2_norm < 0.6)
            # 0~1로 다시 정규화
            alpha = (joint2_norm[mask_linear] - 0.3) / 0.3
            joint1_target[mask_linear] = joint1_upper[mask_linear] * alpha
            joint1_effort[mask_linear] = 0.0

            # 70% 이상: upper limit
            mask_70 = joint2_norm >= 0.6
            joint1_target[mask_70] = joint1_upper[mask_70]
            joint1_effort[mask_70] = 1000.0

            # position target 적용
            art.set_joint_position_target(joint1_target.unsqueeze(1), joint_ids=[0])
            art.set_joint_effort_target(joint1_effort.unsqueeze(1), joint_ids=[0])

    def _check_bottle(self):
        '''
        squeeze mode 에 대해서, 원하는 조건이 만족됐을 떄 valid_env_filter 를 지정해서 joint 를 풀어준다.
        '''
        if self.scene_key in {"5g", "5a"}:
            art = self._art_handles["bottle"]
            # calculate eef rotation
            eef_quat = self.scene['ee_frame'].data.target_quat_w[:,0,:].clone()
            eef_xyz  = torch.stack(euler_xyz_from_quat(eef_quat), dim=-1)
            # calculate asset root rotation
            asset_xyz  = self.asset_ori_delta

            ### check rotation alignment ###
            # tolerances
            tol_xy_deg = 5.0   # x,y는 0 또는 π 근처 허용 오차(도)
            tol_z_deg  = 3.0   # z는 asset z와의 차이 허용 오차(도)
            tol_xy     = torch.deg2rad(torch.tensor(tol_xy_deg, device=eef_xyz.device))
            tol_z      = torch.deg2rad(torch.tensor(tol_z_deg,  device=eef_xyz.device))
            # conditions
            x_ok = near_0_or_pi(eef_xyz[:, 0], tol_xy)
            y_ok = near_0_or_pi(eef_xyz[:, 1], tol_xy)
            z_ok = near_equal_mod_2pi(eef_xyz[:, 2], asset_xyz[:, 0], tol_z)
            release_mask = x_ok & y_ok & z_ok
            to_release = release_mask & (~self._released)
            # to_lock    = (~release_mask) & self._released

            # 해제해야 할 env들
            if to_release.any():
                env_ids = torch.where(to_release)[0]
                # 특정 조인트만 풀고 싶다면 joint_ids 지정 (예: joint_2)
                art.write_joint_friction_coefficient_to_sim(
                    joint_friction_coeff=0.002,
                    joint_ids=[0,1],
                    env_ids=env_ids,
                )

            # 다시 잠가야 할 env들
            # if to_lock.any():
            #     env_ids = torch.where(to_lock)[0].tolist()
            #     art.write_joint_friction_coefficient_to_sim(
            #         joint_friction=1e5,
            #         joint_ids=[0,1],
            #         env_ids=env_ids,
            #     )

            # 상태 갱신
            self._released[to_release] = True
            # self._released[to_lock]    = False

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if 'door' in self._art_names:
            art = self._art_handles["door"]
            art.write_joint_friction_coefficient_to_sim(5e6, joint_ids=[0], env_ids=env_ids)
            self._cache_articulations()
        if 'bottle' in self._art_names:
            self._cache_articulations()
            if self.scene_key in {"5g", "5a"}:
                # squeeze scene
                art = self._art_handles["bottle"]
                art.write_joint_friction_coefficient_to_sim(1e5, joint_ids=[0,1], env_ids=env_ids)
                # 예: 클래스 멤버로 보관
                #TODO: fix the reset for individual env
                self._released = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.scene.device)
                self.N_STEPS_FOR_RELEASE = 5
                self.condition_history = torch.zeros(self.num_envs,self.N_STEPS_FOR_RELEASE, dtype=torch.bool, device=self.device)

    """
    Operations - MDP
    """
    def _check_collect_step_deadline(self, stage: str) -> None:
        """Raise at a physics-loop boundary when a collection smoke step expires."""
        deadline = self._collect_step_deadline_s
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError(f"collect env.step deadline exceeded during {stage}")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        self._check_collect_step_deadline("action processing")
        # process actions
        self.action_manager.process_action(action.to(self.device))
        self._check_collect_step_deadline("action processing")
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._sim_counter_per_action = 0

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._check_collect_step_deadline("physics stepping")
            self._sim_step_counter += 1
            self._sim_counter_per_action += 1
            # set actions into buffers
            self.action_manager.apply_action(self._sim_counter_per_action, self.cfg.decimation)
            # articulation: conditioned movement
            self._check_articulations()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            self._check_collect_step_deadline("simulation")
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
                self._check_collect_step_deadline("rendering")
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            self._check_collect_step_deadline("scene update")

        # post-step:
        self._check_bottle()
        self._check_collect_step_deadline("post-physics checks")
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # Mine -- command skill step + 1
        try:
            skill_term = self.command_manager.get_term('skill_sequence')
            if hasattr(skill_term, 'next_skill'):
                skill_term.next_skill()
        except:
            pass
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self._check_collect_step_deadline("termination computation")
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._check_collect_step_deadline("reward computation")
        # print(f"Reward : {self.reward_buf.tolist()}")

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self._cache_articulations()
            print(f"Resetting env_ids: {reset_env_ids.tolist()}")
        self._check_collect_step_deadline("environment reset")
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        self._check_collect_step_deadline("command update")
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self._check_collect_step_deadline("interval events")
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self._check_collect_step_deadline("observation computation")
        self.obs_buf = self.observation_manager.compute()
        self._check_collect_step_deadline("observation computation")

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

class ManagerBasedContinuousEnv(ManagerBasedRLStepEnv):

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- event manager
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)
        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    """
    Operations - MDP
    """
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnv.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        self._check_collect_step_deadline("action processing")
        # process actions
        self.action_manager.process_action(action.to(self.device))
        self._check_collect_step_deadline("action processing")
        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._sim_counter_per_action = 0

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._check_collect_step_deadline("physics stepping")
            self._sim_step_counter += 1
            self._sim_counter_per_action += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # articulation: conditioned movement
            self._check_articulations()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=not getattr(self, "eval_single_render", False))
            self._check_collect_step_deadline("simulation")
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
                self._check_collect_step_deadline("rendering")
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)
            self._check_collect_step_deadline("scene update")

        # post-step:
        self._check_bottle()
        self._check_collect_step_deadline("post-physics checks")
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # Mine -- command skill step + 1
        try:
            skill_term = self.command_manager.get_term('skill_sequence')
            if hasattr(skill_term, 'next_skill'):
                skill_term.next_skill()
        except:
            pass
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self._check_collect_step_deadline("termination computation")
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)
        self._check_collect_step_deadline("reward computation")
        # print(f"Reward : {self.reward_buf.tolist()}")

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            self._cache_articulations()
            print(f"Resetting env_ids: {reset_env_ids.tolist()}")
        self._check_collect_step_deadline("environment reset")
        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        self._check_collect_step_deadline("command update")
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self._check_collect_step_deadline("interval events")
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self._check_collect_step_deadline("observation computation")
        self.obs_buf = self.observation_manager.compute()
        self._check_collect_step_deadline("observation computation")

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _check_bottle(self):
        '''
        squeeze mode 에 대해서, 원하는 조건이 만족됐을 떄 valid_env_filter 를 지정해서 joint 를 풀어준다.
        '''
        if self.scene_key in {"5g", "5a"}:
            art = self._art_handles["bottle"]
            # calculate eef rotation
            eef_quat = self.scene['ee_frame'].data.target_quat_w[:,0,:].clone()
            eef_xyz  = torch.stack(euler_xyz_from_quat(eef_quat), dim=-1)
            # calculate asset root rotation
            asset_xyz  = self.asset_ori_delta

            ### check rotation alignment ###
            # tolerances
            tol_xy_deg = 5.0   # x,y는 0 또는 π 근처 허용 오차(도)
            tol_z_deg  = 5.0   # z는 asset z와의 차이 허용 오차(도)
            tol_xy     = torch.deg2rad(torch.tensor(tol_xy_deg, device=eef_xyz.device))
            tol_z      = torch.deg2rad(torch.tensor(tol_z_deg,  device=eef_xyz.device))
            # conditions
            x_ok = near_0_or_pi(eef_xyz[:, 0], tol_xy)
            y_ok = near_0_or_pi(eef_xyz[:, 1], tol_xy)
            z_ok = near_equal_mod_2pi(eef_xyz[:, 2], asset_xyz[:, 0], tol_z)
            orientation_ok = x_ok & y_ok & z_ok
            # to_release = orientation_ok

            # Is near cap?
            eef_pos = self.scene['ee_frame'].data.target_pos_w[:,0,:]
            bottle_cap_pos = art.data.body_pos_w[:,2,:]
            is_near_cap = torch.norm(eef_pos - bottle_cap_pos, dim=1) < 0.05
            
            condition_met_mask = is_near_cap & orientation_ok
            self.condition_history[:,:-1] = self.condition_history[:,1:].clone()
            self.condition_history[:,-1] = condition_met_mask
            log_sum = torch.sum(self.condition_history, dim=1)
            release_trigger_mask = (log_sum == self.N_STEPS_FOR_RELEASE)

            to_release = release_trigger_mask & (~self._released)

            # 해제해야 할 env들
            if to_release.any():
                env_ids = torch.where(to_release)[0]
                # 특정 조인트만 풀고 싶다면 joint_ids 지정 (예: joint_2)
                art.write_joint_friction_coefficient_to_sim(
                    joint_friction_coeff=0.002,
                    joint_ids=[0,1],
                    env_ids=env_ids,
                )
                self._released[to_release] = True
