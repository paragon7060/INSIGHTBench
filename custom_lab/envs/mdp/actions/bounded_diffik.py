"""Bounded PhysX-Jacobian differential IK action for INSIGHT evaluation."""

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


class BoundedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """PhysX-Jacobian DiffIK with direction-preserving joint-step bounds."""

    cfg: "BoundedDifferentialInverseKinematicsActionCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.max_joint_step_rad <= 0.0:
            raise ValueError(
                f"max_joint_step_rad must be positive, got {cfg.max_joint_step_rad}"
            )
        self.last_joint_position = None
        self.last_unbounded_joint_target = None
        self.last_joint_target = None
        self.last_joint_step_scale = None

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            unbounded_target = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )
        else:
            unbounded_target = joint_pos.clone()

        delta = unbounded_target - joint_pos
        max_abs_delta = delta.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(
            self.cfg.max_joint_step_rad / (max_abs_delta + 1.0e-8), max=1.0
        )
        joint_target = joint_pos + scale * delta

        if self.cfg.clamp_joint_limits:
            joint_target = torch.clamp(
                joint_target,
                min=self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0],
                max=self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1],
            )

        self.last_joint_position = joint_pos.detach().clone()
        self.last_unbounded_joint_target = unbounded_target.detach().clone()
        self.last_joint_target = joint_target.detach().clone()
        self.last_joint_step_scale = scale.detach().clone()
        self._asset.set_joint_position_target(joint_target, self._joint_ids)


@configclass
class BoundedDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg
):
    """Configuration for :class:`BoundedDifferentialInverseKinematicsAction`."""

    class_type: type[ActionTerm] = BoundedDifferentialInverseKinematicsAction
    max_joint_step_rad: float = 0.05
    clamp_joint_limits: bool = True


@configclass
class BoundedPhysxDiffIKActionsCfg:
    """Absolute TCP pose plus two-finger position actions."""

    arm_action: ActionTermCfg = BoundedDifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        body_offset=BoundedDifferentialInverseKinematicsActionCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.1034),
        ),
        scale=1.0,
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": 1.0e-3},
        ),
        max_joint_step_rad=0.05,
        clamp_joint_limits=True,
    )
    gripper_action: ActionTermCfg = JointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        use_default_offset=False,
    )
