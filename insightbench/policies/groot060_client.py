"""Python 3.11 client wrapper for the isolated LeRobot 0.6 GR00T server."""

from __future__ import annotations

import base64
import hashlib
import json
import sys
import time
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import torch

from insightbench.policies.base import PolicyBase


_PANDA_LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973)
_PANDA_UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973)
_LULA_GRIPPER_TO_TCP_POS = np.array((0.0, 0.0, 0.0034), dtype=np.float64)
_LULA_GRIPPER_TO_TCP_ROT = np.diag((-1.0, -1.0, 1.0))


def _encode_array(value: np.ndarray) -> dict:
    buffer = BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return {
        "format": "npy_base64",
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _decode_array(value: dict) -> np.ndarray:
    if value.get("format") != "npy_base64":
        return np.asarray(value, dtype=np.float32)
    return np.load(BytesIO(base64.b64decode(value["data"])), allow_pickle=False)


def _normalize_vectors(value: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.clip(norm, 1e-8, None)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_vectors(quat)
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(-1, 3, 3)


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_quat()
    return quat_xyzw[..., (3, 0, 1, 2)]


def _rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    col0 = _normalize_vectors(rotation_6d[:, 0:3])
    raw_col1 = rotation_6d[:, 3:6]
    col1 = _normalize_vectors(raw_col1 - np.sum(col0 * raw_col1, axis=-1, keepdims=True) * col0)
    col2 = np.cross(col0, col1)
    return np.stack((col0, col1, col2), axis=-1)


def _matrix_to_axis_angle(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    angle = np.arccos(np.clip((trace - 1.0) * 0.5, -1.0, 1.0))
    vee = np.stack(
        (
            matrix[:, 2, 1] - matrix[:, 1, 2],
            matrix[:, 0, 2] - matrix[:, 2, 0],
            matrix[:, 1, 0] - matrix[:, 0, 1],
        ),
        axis=-1,
    )
    result = np.empty_like(vee)
    regular = np.abs(np.sin(angle)) > 1e-5
    result[regular] = vee[regular] * (angle[regular] / (2.0 * np.sin(angle[regular])))[:, None]
    result[~regular] = 0.5 * vee[~regular]

    near_pi = (~regular) & (angle > 1e-3)
    for index in np.flatnonzero(near_pi):
        rotation = matrix[index]
        axis = np.sqrt(np.maximum((np.diag(rotation) + 1.0) * 0.5, 0.0))
        largest = int(np.argmax(axis))
        if axis[largest] > 1e-6:
            if largest == 0:
                axis[1] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * axis[0])
                axis[2] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * axis[0])
            elif largest == 1:
                axis[0] = (rotation[0, 1] + rotation[1, 0]) / (4.0 * axis[1])
                axis[2] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * axis[1])
            else:
                axis[0] = (rotation[0, 2] + rotation[2, 0]) / (4.0 * axis[2])
                axis[1] = (rotation[1, 2] + rotation[2, 1]) / (4.0 * axis[2])
        result[index] = _normalize_vectors(axis[None, :])[0] * angle[index]
    return result


def _clip_vector_norm(value: np.ndarray, limit: float) -> np.ndarray:
    if limit <= 0:
        return value
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    scale = np.minimum(1.0, limit / np.clip(norm, 1e-8, None))
    return value * scale


def _axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(np.asarray(axis_angle, dtype=np.float64)).as_matrix()


def _load_lula_kinematics(extension_path: str) -> tuple[object, str]:
    extension = Path(extension_path).expanduser().resolve()
    if not extension.is_dir():
        raise FileNotFoundError(f"Lula motion-generation extension not found: {extension}")
    if str(extension) not in sys.path:
        sys.path.insert(0, str(extension))
    lula_prebundle = extension.parent / "isaacsim.robot_motion.lula" / "pip_prebundle"
    if not lula_prebundle.is_dir():
        raise FileNotFoundError(f"Lula Python binding not found: {lula_prebundle}")
    if str(lula_prebundle) not in sys.path:
        sys.path.insert(0, str(lula_prebundle))

    import lula

    config_path = extension / "motion_policy_configs" / "franka" / "rmpflow" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    relative = config["relative_asset_paths"]
    config_dir = config_path.parent
    robot_description_path = config_dir / relative["robot_description_path"]
    urdf_path = (config_dir / relative["urdf_path"]).resolve()
    description = lula.load_robot(str(robot_description_path), str(urdf_path))
    return description.kinematics(), str(config["end_effector_frame_name"])


def _load_official_lula_components(extension_path: str) -> tuple[type, type, dict]:
    extension = Path(extension_path).expanduser().resolve()
    if not extension.is_dir():
        raise FileNotFoundError(f"Lula motion-generation extension not found: {extension}")
    if str(extension) not in sys.path:
        sys.path.insert(0, str(extension))
    lula_prebundle = extension.parent / "isaacsim.robot_motion.lula" / "pip_prebundle"
    if str(lula_prebundle) not in sys.path:
        sys.path.insert(0, str(lula_prebundle))

    import isaacsim

    extension_namespace = str(extension / "isaacsim")
    if extension_namespace not in isaacsim.__path__:
        isaacsim.__path__.append(extension_namespace)
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, RmpFlow

    config_path = extension / "motion_policy_configs" / "franka" / "rmpflow" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    relative = config["relative_asset_paths"]
    config_dir = config_path.parent
    paths = {
        "robot_description_path": str((config_dir / relative["robot_description_path"]).resolve()),
        "urdf_path": str((config_dir / relative["urdf_path"]).resolve()),
        "rmpflow_config_path": str((config_dir / relative["rmpflow_config_path"]).resolve()),
        "end_effector_frame_name": str(config["end_effector_frame_name"]),
        "maximum_substep_size": float(config["maximum_substep_size"]),
    }
    return LulaKinematicsSolver, RmpFlow, paths


def _request_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600.0) as response:  # noqa: S310 - configured localhost bridge
        return json.loads(response.read().decode("utf-8"))


class Groot060ClientWrapper(PolicyBase):
    """Map INSIGHTBench observations to the local Python 3.12 GR00T server."""

    def __init__(self, policy_cfg, device: torch.device):
        super().__init__(policy_cfg, device)
        self.endpoint = str(policy_cfg.endpoint).rstrip("/")
        self.reset_endpoint = str(policy_cfg.reset_endpoint)
        self.health_endpoint = str(policy_cfg.health_endpoint)
        self.checkpoint_kind = str(policy_cfg.get("checkpoint_kind", "base_n1_7"))
        self.action_space = str(policy_cfg.get("action_space", "ee_abs_rot6d"))
        image_mapping = policy_cfg.get("image_mapping")
        if image_mapping:
            self.image_mapping = {str(key): str(value) for key, value in image_mapping.items()}
        else:
            self.image_mapping = {
                "exterior_image_1_left": str(
                    policy_cfg.get("external_image_key", "observation.images.right_shoulder")
                ),
                "wrist_image_left": str(policy_cfg.get("wrist_image_key", "observation.images.wrist")),
            }
        self.position_delta_limit = float(policy_cfg.get("position_delta_limit", 0.0))
        self.rotation_delta_limit = float(policy_cfg.get("rotation_delta_limit", 0.0))
        self.gripper_delta_limit = float(policy_cfg.get("gripper_delta_limit", 0.0))
        self.target_smoothing_alpha = float(policy_cfg.get("target_smoothing_alpha", 1.0))
        if not 0.0 < self.target_smoothing_alpha <= 1.0:
            raise ValueError(
                "target_smoothing_alpha must be in (0, 1], got "
                f"{self.target_smoothing_alpha}"
            )
        self.raw_action_mode = str(policy_cfg.get("raw_action_mode", "online"))
        if self.raw_action_mode not in {"online", "record", "replay"}:
            raise ValueError(
                "raw_action_mode must be one of online, record, replay; got "
                f"{self.raw_action_mode!r}"
            )
        raw_action_path = policy_cfg.get("raw_action_path")
        self._raw_action_path = (
            Path(str(raw_action_path)).expanduser() if raw_action_path else None
        )
        if self.raw_action_mode != "online" and self._raw_action_path is None:
            raise ValueError(
                f"raw_action_path is required for raw_action_mode={self.raw_action_mode!r}"
            )
        self._raw_action_replay: dict[tuple[int, int], np.ndarray] = {}
        self._raw_action_episode = -1
        self._raw_action_step = 0
        self.ee_to_joint_solver = str(policy_cfg.get("ee_to_joint_solver", "local_lula_dls"))
        self.lula_extension_path = str(
            policy_cfg.get(
                "lula_extension_path",
                Path.home()
                / "isaacsim"
                / "exts"
                / "isaacsim.robot_motion.motion_generation",
            )
        )
        self.ik_max_iterations = int(policy_cfg.get("ik_max_iterations", 50))
        self.ik_position_tolerance_m = float(
            policy_cfg.get("ik_position_tolerance_m", 1e-7)
        )
        self.ik_rotation_tolerance_rad = float(
            policy_cfg.get("ik_rotation_tolerance_rad", 1e-6)
        )
        self.ik_damping = float(policy_cfg.get("ik_damping", 1e-4))
        self.ik_max_joint_step_rad = float(policy_cfg.get("ik_max_joint_step_rad", 0.1))
        self.diff_ik_damping = float(policy_cfg.get("diff_ik_damping", 0.01))
        self.rmpflow_frame_duration = float(policy_cfg.get("rmpflow_frame_duration", 0.1))
        self._kinematics = None
        self._kinematics_frame = None
        self._joint_limits = None
        self._official_kinematics = None
        self._diff_ik_controller = None
        self._rmpflow_type = None
        self._rmpflow_paths = None
        self._rmpflows = None
        self._rmpflow_prev_joints = None
        self._ik_failure_count = 0
        self._smoothed_target_position = None
        self._smoothed_target_rotation = None
        trace_path = policy_cfg.get("trace_path")
        self._trace_path = Path(str(trace_path)).expanduser() if trace_path else None
        self._trace_step = 0
        self._trace_prev_position = None
        self._trace_prev_rotation = None
        self._trace_prev_joint_target = None
        self._trace_prev_actual_joint = None
        self._pending_physx_trace = None
        self._supports_cached_only = False
        self._needs_full_query = True

    def load(self) -> None:
        health = _request_json("GET", self.health_endpoint)
        if not health.get("ok"):
            raise RuntimeError(f"GR00T 0.6 server health check failed: {health}")
        if int(health["state_dim"]) != int(self.cfg.state_dim):
            raise ValueError(f"Server state_dim={health['state_dim']} but config has {self.cfg.state_dim}")
        if int(health["action_dim"]) != int(self.cfg.action_dim):
            raise ValueError(f"Server action_dim={health['action_dim']} but config has {self.cfg.action_dim}")
        server_kind = str(health.get("checkpoint_kind", self.checkpoint_kind))
        if server_kind != self.checkpoint_kind:
            raise ValueError(f"Server checkpoint_kind={server_kind} but config has {self.checkpoint_kind}")
        if self.checkpoint_kind == "insight_finetuned":
            if self.action_space == "joint_abs":
                if int(self.cfg.action_dim) != 8:
                    raise ValueError(
                        f"joint_abs requires action_dim=8, got {self.cfg.action_dim}"
                    )
            elif self.action_space == "ee_abs_rot6d":
                if int(self.cfg.action_dim) != 10:
                    raise ValueError(
                        f"ee_abs_rot6d requires action_dim=10, got {self.cfg.action_dim}"
                    )
                supported_solvers = {
                    "local_lula_dls",
                    "isaaclab_diffik_dls",
                    "isaaclab_diffik_physx_bounded",
                    "lula_kinematics",
                    "rmpflow",
                }
                if self.ee_to_joint_solver not in supported_solvers:
                    raise ValueError(
                        f"Unsupported ee_to_joint_solver={self.ee_to_joint_solver!r}; "
                        f"choose from {sorted(supported_solvers)}"
                    )
                if self.ee_to_joint_solver != "isaaclab_diffik_physx_bounded":
                    self._kinematics, self._kinematics_frame = _load_lula_kinematics(
                        self.lula_extension_path
                    )
                    self._joint_limits = np.asarray(
                        [
                            (
                                self._kinematics.c_space_coord_limits(index).lower,
                                self._kinematics.c_space_coord_limits(index).upper,
                            )
                            for index in range(7)
                        ],
                        dtype=np.float64,
                    )
                    if self.ee_to_joint_solver in {"lula_kinematics", "rmpflow"}:
                        kinematics_type, self._rmpflow_type, self._rmpflow_paths = (
                            _load_official_lula_components(self.lula_extension_path)
                        )
                        if self.ee_to_joint_solver == "lula_kinematics":
                            self._official_kinematics = kinematics_type(
                                robot_description_path=self._rmpflow_paths["robot_description_path"],
                                urdf_path=self._rmpflow_paths["urdf_path"],
                            )
            else:
                raise ValueError(
                    f"Unsupported INSIGHT action_space={self.action_space!r}"
                )
        self._prepare_raw_action_io()
        self._supports_cached_only = bool(health.get("supports_cached_only", False))
        self._needs_full_query = True
        self.policy = True

    def _prepare_raw_action_io(self) -> None:
        if self.raw_action_mode == "online":
            return
        assert self._raw_action_path is not None
        if self.raw_action_mode == "record":
            self._raw_action_path.parent.mkdir(parents=True, exist_ok=True)
            if self._raw_action_path.exists():
                raise FileExistsError(
                    f"Raw action record already exists: {self._raw_action_path}"
                )
            return

        if not self._raw_action_path.is_file():
            raise FileNotFoundError(
                f"Raw action replay file not found: {self._raw_action_path}"
            )
        episode_steps: dict[int, set[int]] = {}
        for line_number, raw_line in enumerate(
            self._raw_action_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            episode = int(record["episode"])
            step = int(record["step"])
            key = (episode, step)
            if key in self._raw_action_replay:
                raise ValueError(
                    f"Duplicate raw action replay key {key} at line {line_number}"
                )
            action = _decode_array(record["action"]).astype(np.float32, copy=False)
            if action.ndim != 2 or action.shape[1] != int(self.cfg.action_dim):
                raise ValueError(
                    f"Raw action replay {key} has shape {action.shape}; expected "
                    f"[B, {self.cfg.action_dim}]"
                )
            digest = hashlib.sha256(np.ascontiguousarray(action).tobytes()).hexdigest()
            if digest != record.get("sha256"):
                raise ValueError(
                    f"Raw action replay checksum mismatch for {key}: "
                    f"expected {record.get('sha256')}, got {digest}"
                )
            self._raw_action_replay[key] = action
            episode_steps.setdefault(episode, set()).add(step)
        if not self._raw_action_replay:
            raise ValueError(f"Raw action replay file is empty: {self._raw_action_path}")
        expected_episodes = set(range(max(episode_steps) + 1))
        if set(episode_steps) != expected_episodes:
            raise ValueError(
                f"Raw action replay episodes are not contiguous: {sorted(episode_steps)}"
            )
        for episode, steps in episode_steps.items():
            expected_steps = set(range(max(steps) + 1))
            if steps != expected_steps:
                raise ValueError(
                    f"Raw action replay episode {episode} steps are not contiguous"
                )

    def _record_raw_action(self, action: np.ndarray) -> None:
        assert self._raw_action_path is not None
        contiguous = np.ascontiguousarray(action, dtype=np.float32)
        record = {
            "episode": self._raw_action_episode,
            "step": self._raw_action_step,
            "action_space": self.action_space,
            "shape": list(contiguous.shape),
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
            "action": _encode_array(contiguous),
        }
        with self._raw_action_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    def _replay_raw_action(self) -> np.ndarray:
        key = (self._raw_action_episode, self._raw_action_step)
        action = self._raw_action_replay.get(key)
        if action is None:
            raise KeyError(
                f"Raw action replay has no action for episode={key[0]} step={key[1]}"
            )
        return action.copy()

    def reset(self) -> None:
        if self.raw_action_mode == "replay":
            return
        response = _request_json("POST", self.reset_endpoint, {})
        if not response.get("ok"):
            raise RuntimeError(f"GR00T 0.6 reset failed: {response}")
        self._needs_full_query = True

    def reset_episode(self) -> None:
        self._raw_action_episode += 1
        self._raw_action_step = 0
        self._smoothed_target_position = None
        self._smoothed_target_rotation = None
        self._trace_prev_position = None
        self._trace_prev_rotation = None
        self._trace_prev_joint_target = None
        self._trace_prev_actual_joint = None
        self._pending_physx_trace = None
        self._diff_ik_controller = None
        self._rmpflow_prev_joints = None
        if self._rmpflows is not None:
            for rmpflow in self._rmpflows:
                rmpflow.reset()
        self.reset()

    def _smooth_targets(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        alpha = self.target_smoothing_alpha
        if alpha >= 1.0 or self._smoothed_target_position is None:
            smoothed_position = target_position.copy()
            smoothed_rotation = target_rotation.copy()
        else:
            smoothed_position = (
                (1.0 - alpha) * self._smoothed_target_position
                + alpha * target_position
            )
            relative_rotation = target_rotation @ np.swapaxes(
                self._smoothed_target_rotation, -1, -2
            )
            rotation_step = alpha * _matrix_to_axis_angle(relative_rotation)
            smoothed_rotation = (
                _axis_angle_to_matrix(rotation_step) @ self._smoothed_target_rotation
            )
        self._smoothed_target_position = smoothed_position.copy()
        self._smoothed_target_rotation = smoothed_rotation.copy()
        return smoothed_position, smoothed_rotation

    def _solve_local_lula_ik(
        self,
        tcp_positions: np.ndarray,
        tcp_rotations: np.ndarray,
        seed_joints: np.ndarray,
    ) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        if self._kinematics is None or self._joint_limits is None:
            raise RuntimeError("Lula kinematics was not initialized")

        solved = np.empty_like(seed_joints, dtype=np.float64)
        for env_index, (tcp_pos, tcp_rot, seed_joint) in enumerate(
            zip(tcp_positions, tcp_rotations, seed_joints)
        ):
            target_rot = tcp_rot @ _LULA_GRIPPER_TO_TCP_ROT.T
            target_pos = tcp_pos - target_rot @ _LULA_GRIPPER_TO_TCP_POS
            joint = np.clip(
                np.asarray(seed_joint, dtype=np.float64),
                self._joint_limits[:, 0],
                self._joint_limits[:, 1],
            )
            converged = False

            for _ in range(self.ik_max_iterations):
                pose = self._kinematics.pose(joint[:, None], self._kinematics_frame)
                current_pos = np.asarray(pose.translation, dtype=np.float64)
                current_rot = np.asarray(pose.rotation.matrix(), dtype=np.float64)
                position_error = target_pos - current_pos
                rotation_error = Rotation.from_matrix(target_rot @ current_rot.T).as_rotvec()
                if (
                    np.linalg.norm(position_error) <= self.ik_position_tolerance_m
                    and np.linalg.norm(rotation_error) <= self.ik_rotation_tolerance_rad
                ):
                    converged = True
                    break

                jacobian = np.asarray(
                    self._kinematics.jacobian(joint[:, None], self._kinematics_frame),
                    dtype=np.float64,
                )
                residual = np.concatenate((position_error, rotation_error))
                regularized = (
                    jacobian @ jacobian.T + (self.ik_damping**2) * np.eye(6)
                )
                try:
                    delta_joint = jacobian.T @ np.linalg.solve(regularized, residual)
                except np.linalg.LinAlgError:
                    break
                max_step = float(np.max(np.abs(delta_joint)))
                if max_step > self.ik_max_joint_step_rad:
                    delta_joint *= self.ik_max_joint_step_rad / max_step
                if not np.isfinite(delta_joint).all():
                    break
                joint = np.clip(
                    joint + delta_joint,
                    self._joint_limits[:, 0],
                    self._joint_limits[:, 1],
                )

            if not converged:
                self._ik_failure_count += 1
                if self._ik_failure_count <= 5 or self._ik_failure_count % 100 == 0:
                    print(
                        "[Groot060ClientWrapper] local Lula DLS did not converge "
                        f"(env={env_index}, total_failures={self._ik_failure_count}); "
                        "using the best finite joint solution.",
                        flush=True,
                    )
            solved[env_index] = joint
        return solved

    @staticmethod
    def _tcp_to_gripper_targets(
        tcp_positions: np.ndarray, tcp_rotations: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        gripper_rotations = tcp_rotations @ _LULA_GRIPPER_TO_TCP_ROT.T
        gripper_positions = tcp_positions - np.einsum(
            "bij,j->bi", gripper_rotations, _LULA_GRIPPER_TO_TCP_POS
        )
        return gripper_positions, gripper_rotations

    def _solve_isaaclab_diffik(
        self,
        tcp_positions: np.ndarray,
        tcp_rotations: np.ndarray,
        seed_joints: np.ndarray,
    ) -> np.ndarray:
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        if self._kinematics is None or self._joint_limits is None:
            raise RuntimeError("Lula kinematics was not initialized")
        num_envs = seed_joints.shape[0]
        if self._diff_ik_controller is None or self._diff_ik_controller.num_envs != num_envs:
            cfg = DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": self.diff_ik_damping},
            )
            self._diff_ik_controller = DifferentialIKController(
                cfg, num_envs=num_envs, device=str(self.device)
            )

        target_positions, target_rotations = self._tcp_to_gripper_targets(
            tcp_positions, tcp_rotations
        )
        current_positions = np.empty((num_envs, 3), dtype=np.float64)
        current_rotations = np.empty((num_envs, 3, 3), dtype=np.float64)
        jacobians = np.empty((num_envs, 6, 7), dtype=np.float64)
        for env_index, joint in enumerate(seed_joints):
            pose = self._kinematics.pose(joint.astype(np.float64)[:, None], self._kinematics_frame)
            current_positions[env_index] = np.asarray(pose.translation, dtype=np.float64)
            current_rotations[env_index] = np.asarray(pose.rotation.matrix(), dtype=np.float64)
            jacobians[env_index] = np.asarray(
                self._kinematics.jacobian(joint.astype(np.float64)[:, None], self._kinematics_frame),
                dtype=np.float64,
            )

        command = np.concatenate(
            [target_positions, _matrix_to_quat_wxyz(target_rotations)], axis=1
        )
        device = self.device
        command_t = torch.as_tensor(command, dtype=torch.float32, device=device)
        current_position_t = torch.as_tensor(current_positions, dtype=torch.float32, device=device)
        current_quat_t = torch.as_tensor(
            _matrix_to_quat_wxyz(current_rotations), dtype=torch.float32, device=device
        )
        jacobian_t = torch.as_tensor(jacobians, dtype=torch.float32, device=device)
        joints_t = torch.as_tensor(seed_joints, dtype=torch.float32, device=device)
        self._diff_ik_controller.set_command(command_t)
        solved = self._diff_ik_controller.compute(
            current_position_t, current_quat_t, jacobian_t, joints_t
        ).detach().cpu().numpy().astype(np.float64)
        delta = np.clip(
            solved - seed_joints,
            -self.ik_max_joint_step_rad,
            self.ik_max_joint_step_rad,
        )
        return np.clip(
            seed_joints + delta, self._joint_limits[:, 0], self._joint_limits[:, 1]
        )

    def _solve_lula_kinematics(
        self,
        tcp_positions: np.ndarray,
        tcp_rotations: np.ndarray,
        seed_joints: np.ndarray,
    ) -> np.ndarray:
        if self._official_kinematics is None or self._joint_limits is None:
            raise RuntimeError("Official LulaKinematicsSolver was not initialized")
        target_positions, target_rotations = self._tcp_to_gripper_targets(
            tcp_positions, tcp_rotations
        )
        target_quaternions = _matrix_to_quat_wxyz(target_rotations)
        solved = np.empty_like(seed_joints, dtype=np.float64)
        for env_index, (position, quaternion, seed_joint) in enumerate(
            zip(target_positions, target_quaternions, seed_joints)
        ):
            joint, success = self._official_kinematics.compute_inverse_kinematics(
                self._kinematics_frame,
                position,
                quaternion,
                warm_start=seed_joint.astype(np.float64),
            )
            if not success:
                self._ik_failure_count += 1
                if self._ik_failure_count <= 5 or self._ik_failure_count % 100 == 0:
                    print(
                        "[Groot060ClientWrapper] LulaKinematicsSolver did not converge "
                        f"(env={env_index}, total_failures={self._ik_failure_count}); "
                        "using its finite candidate.",
                        flush=True,
                    )
            candidate = np.asarray(joint, dtype=np.float64)
            if not np.isfinite(candidate).all():
                candidate = seed_joint.astype(np.float64)
            solved[env_index] = np.clip(
                candidate, self._joint_limits[:, 0], self._joint_limits[:, 1]
            )
        return solved

    def _solve_rmpflow(
        self,
        tcp_positions: np.ndarray,
        tcp_rotations: np.ndarray,
        seed_joints: np.ndarray,
    ) -> np.ndarray:
        if self._rmpflow_type is None or self._rmpflow_paths is None or self._joint_limits is None:
            raise RuntimeError("RmpFlow was not initialized")
        num_envs = seed_joints.shape[0]
        if self._rmpflows is None or len(self._rmpflows) != num_envs:
            self._rmpflows = [
                self._rmpflow_type(
                    robot_description_path=self._rmpflow_paths["robot_description_path"],
                    urdf_path=self._rmpflow_paths["urdf_path"],
                    rmpflow_config_path=self._rmpflow_paths["rmpflow_config_path"],
                    end_effector_frame_name=self._rmpflow_paths["end_effector_frame_name"],
                    maximum_substep_size=self._rmpflow_paths["maximum_substep_size"],
                    ignore_robot_state_updates=False,
                )
                for _ in range(num_envs)
            ]
            self._rmpflow_prev_joints = None

        target_positions, target_rotations = self._tcp_to_gripper_targets(
            tcp_positions, tcp_rotations
        )
        target_quaternions = _matrix_to_quat_wxyz(target_rotations)
        if self._rmpflow_prev_joints is None:
            velocities = np.zeros_like(seed_joints, dtype=np.float64)
        else:
            velocities = (
                seed_joints - self._rmpflow_prev_joints
            ) / self.rmpflow_frame_duration
        solved = np.empty_like(seed_joints, dtype=np.float64)
        empty = np.empty(0, dtype=np.float64)
        for env_index, rmpflow in enumerate(self._rmpflows):
            rmpflow.set_end_effector_target(
                target_positions[env_index], target_quaternions[env_index]
            )
            joint_target, _ = rmpflow.compute_joint_targets(
                seed_joints[env_index].astype(np.float64).copy(),
                velocities[env_index].astype(np.float64).copy(),
                empty,
                empty,
                self.rmpflow_frame_duration,
            )
            candidate = np.asarray(joint_target, dtype=np.float64)
            if not np.isfinite(candidate).all():
                candidate = seed_joints[env_index].astype(np.float64)
            solved[env_index] = np.clip(
                candidate, self._joint_limits[:, 0], self._joint_limits[:, 1]
            )
        self._rmpflow_prev_joints = seed_joints.copy()
        return solved

    def _solve_ee_to_joint(
        self,
        tcp_positions: np.ndarray,
        tcp_rotations: np.ndarray,
        seed_joints: np.ndarray,
    ) -> np.ndarray:
        if self.ee_to_joint_solver == "local_lula_dls":
            return self._solve_local_lula_ik(tcp_positions, tcp_rotations, seed_joints)
        if self.ee_to_joint_solver == "isaaclab_diffik_dls":
            return self._solve_isaaclab_diffik(tcp_positions, tcp_rotations, seed_joints)
        if self.ee_to_joint_solver == "lula_kinematics":
            return self._solve_lula_kinematics(tcp_positions, tcp_rotations, seed_joints)
        if self.ee_to_joint_solver == "rmpflow":
            return self._solve_rmpflow(tcp_positions, tcp_rotations, seed_joints)
        raise RuntimeError(f"Unsupported ee_to_joint_solver={self.ee_to_joint_solver!r}")

    def _trace_insight_action(
        self,
        response: dict,
        state: np.ndarray,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        joint_targets: np.ndarray,
    ) -> None:
        if self._trace_path is None:
            return

        from scipy.spatial.transform import Rotation

        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        actual_joints = state[:, 7:14]
        records = []
        for env_index in range(state.shape[0]):
            target_gripper_rot = target_rotation[env_index] @ _LULA_GRIPPER_TO_TCP_ROT.T
            target_gripper_pos = (
                target_position[env_index]
                - target_gripper_rot @ _LULA_GRIPPER_TO_TCP_POS
            )
            solved_pose = self._kinematics.pose(
                joint_targets[env_index].astype(np.float64)[:, None], self._kinematics_frame
            )
            solved_position = np.asarray(solved_pose.translation, dtype=np.float64)
            solved_rotation = np.asarray(solved_pose.rotation.matrix(), dtype=np.float64)
            ik_position_error = float(np.linalg.norm(target_gripper_pos - solved_position))
            ik_rotation_error = float(
                np.linalg.norm(
                    Rotation.from_matrix(target_gripper_rot @ solved_rotation.T).as_rotvec()
                )
            )

            position_step = None
            rotation_step = None
            joint_target_step = None
            actual_joint_step = None
            previous_target_tracking_error = None
            if self._trace_prev_position is not None:
                position_step = float(
                    np.linalg.norm(
                        target_position[env_index] - self._trace_prev_position[env_index]
                    )
                )
                rotation_step = float(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            target_rotation[env_index]
                            @ self._trace_prev_rotation[env_index].T
                        ).as_rotvec()
                    )
                )
                joint_target_step = float(
                    np.max(
                        np.abs(
                            joint_targets[env_index]
                            - self._trace_prev_joint_target[env_index]
                        )
                    )
                )
                actual_joint_step = float(
                    np.max(
                        np.abs(
                            actual_joints[env_index]
                            - self._trace_prev_actual_joint[env_index]
                        )
                    )
                )
                previous_target_tracking_error = float(
                    np.max(
                        np.abs(
                            self._trace_prev_joint_target[env_index]
                            - actual_joints[env_index]
                        )
                    )
                )

            records.append(
                {
                    "step": self._trace_step,
                    "env": env_index,
                    "chunk_index": int(response.get("chunk_index", -1)),
                    "chunk_actions": int(response.get("chunk_actions", -1)),
                    "chunk_shape": response.get("chunk_shape"),
                    "inference_ran": bool(response.get("inference_ran", False)),
                    "request_kind": response.get("_client_request_kind"),
                    "server_elapsed_ms": response.get("elapsed_ms"),
                    "client_payload_ms": response.get("_client_payload_ms"),
                    "client_http_ms": response.get("_client_http_ms"),
                    "client_cache_probe_ms": response.get("_client_cache_probe_ms"),
                    "client_ik_ms": response.get("_client_ik_ms"),
                    "target_smoothing_alpha": self.target_smoothing_alpha,
                    "ik_converged": bool(
                        ik_position_error <= self.ik_position_tolerance_m
                        and ik_rotation_error <= self.ik_rotation_tolerance_rad
                    ),
                    "ik_position_error_m": ik_position_error,
                    "ik_rotation_error_rad": ik_rotation_error,
                    "ee_command_distance_m": float(
                        np.linalg.norm(target_position[env_index] - current_position[env_index])
                    ),
                    "ee_command_rotation_rad": float(
                        np.linalg.norm(
                            Rotation.from_matrix(
                                target_rotation[env_index] @ current_rotation[env_index].T
                            ).as_rotvec()
                        )
                    ),
                    "ee_target_step_m": position_step,
                    "ee_target_step_rotation_rad": rotation_step,
                    "joint_target_step_max_rad": joint_target_step,
                    "actual_joint_step_max_rad": actual_joint_step,
                    "previous_target_tracking_error_max_rad": previous_target_tracking_error,
                    "command_joint_error_max_rad": float(
                        np.max(np.abs(joint_targets[env_index] - actual_joints[env_index]))
                    ),
                    "ee_target_position": target_position[env_index].tolist(),
                    "joint_target": joint_targets[env_index].tolist(),
                    "actual_joint": actual_joints[env_index].tolist(),
                    "observation_state": state[env_index].tolist(),
                }
            )

        with self._trace_path.open("a", encoding="utf-8") as trace_file:
            for record in records:
                trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._trace_step += 1
        self._trace_prev_position = target_position.copy()
        self._trace_prev_rotation = target_rotation.copy()
        self._trace_prev_joint_target = joint_targets.copy()
        self._trace_prev_actual_joint = actual_joints.copy()

    def trace_after_env_step(self, env) -> None:
        """Record bounded PhysX DiffIK targets after the action term has run."""
        context = self._pending_physx_trace
        self._pending_physx_trace = None
        if context is None or self._trace_path is None:
            return

        from scipy.spatial.transform import Rotation

        arm_term = env.action_manager._terms["arm_action"]
        required = (
            "last_joint_position",
            "last_unbounded_joint_target",
            "last_joint_target",
            "last_joint_step_scale",
        )
        if any(getattr(arm_term, name, None) is None for name in required):
            raise RuntimeError("Bounded PhysX DiffIK action term did not expose diagnostics")

        command_seed = arm_term.last_joint_position.detach().cpu().numpy()
        unbounded_target = arm_term.last_unbounded_joint_target.detach().cpu().numpy()
        joint_targets = arm_term.last_joint_target.detach().cpu().numpy()
        joint_scales = arm_term.last_joint_step_scale.detach().cpu().numpy()
        post_actual_joints = env.scene["robot"].data.joint_pos[:, :7].detach().cpu().numpy()

        response = context["response"]
        state = context["state"]
        current_position = context["current_position"]
        current_rotation = context["current_rotation"]
        target_position = context["target_position"]
        target_rotation = context["target_rotation"]
        pre_actual_joints = state[:, 7:14]

        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        records = []
        for env_index in range(state.shape[0]):
            position_step = None
            rotation_step = None
            joint_target_step = None
            previous_target_tracking_error = None
            if self._trace_prev_position is not None:
                position_step = float(
                    np.linalg.norm(
                        target_position[env_index] - self._trace_prev_position[env_index]
                    )
                )
                rotation_step = float(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            target_rotation[env_index]
                            @ self._trace_prev_rotation[env_index].T
                        ).as_rotvec()
                    )
                )
                joint_target_step = float(
                    np.max(
                        np.abs(
                            joint_targets[env_index]
                            - self._trace_prev_joint_target[env_index]
                        )
                    )
                )
                previous_target_tracking_error = float(
                    np.max(
                        np.abs(
                            self._trace_prev_joint_target[env_index]
                            - pre_actual_joints[env_index]
                        )
                    )
                )

            records.append(
                {
                    "step": self._trace_step,
                    "env": env_index,
                    "chunk_index": int(response.get("chunk_index", -1)),
                    "chunk_actions": int(response.get("chunk_actions", -1)),
                    "chunk_shape": response.get("chunk_shape"),
                    "inference_ran": bool(response.get("inference_ran", False)),
                    "request_kind": response.get("_client_request_kind"),
                    "server_elapsed_ms": response.get("elapsed_ms"),
                    "client_payload_ms": response.get("_client_payload_ms"),
                    "client_http_ms": response.get("_client_http_ms"),
                    "client_cache_probe_ms": response.get("_client_cache_probe_ms"),
                    "controller_backend": "isaaclab_diffik_physx_bounded",
                    "physx_jacobian": True,
                    "target_smoothing_alpha": self.target_smoothing_alpha,
                    "ee_command_distance_m": float(
                        np.linalg.norm(
                            target_position[env_index] - current_position[env_index]
                        )
                    ),
                    "ee_command_rotation_rad": float(
                        np.linalg.norm(
                            Rotation.from_matrix(
                                target_rotation[env_index]
                                @ current_rotation[env_index].T
                            ).as_rotvec()
                        )
                    ),
                    "ee_target_step_m": position_step,
                    "ee_target_step_rotation_rad": rotation_step,
                    "joint_target_step_max_rad": joint_target_step,
                    "actual_joint_step_max_rad": float(
                        np.max(
                            np.abs(
                                post_actual_joints[env_index]
                                - pre_actual_joints[env_index]
                            )
                        )
                    ),
                    "previous_target_tracking_error_max_rad": previous_target_tracking_error,
                    "command_joint_error_max_rad": float(
                        np.max(
                            np.abs(
                                joint_targets[env_index] - command_seed[env_index]
                            )
                        )
                    ),
                    "post_step_tracking_error_max_rad": float(
                        np.max(
                            np.abs(
                                joint_targets[env_index] - post_actual_joints[env_index]
                            )
                        )
                    ),
                    "unbounded_joint_step_max_rad": float(
                        np.max(
                            np.abs(
                                unbounded_target[env_index] - command_seed[env_index]
                            )
                        )
                    ),
                    "joint_step_scale": float(joint_scales[env_index, 0]),
                    "joint_step_saturated": bool(joint_scales[env_index, 0] < 0.999999),
                    "ee_target_position": target_position[env_index].tolist(),
                    "joint_target": joint_targets[env_index].tolist(),
                    "actual_joint": pre_actual_joints[env_index].tolist(),
                    "post_actual_joint": post_actual_joints[env_index].tolist(),
                    "observation_state": state[env_index].tolist(),
                }
            )

        with self._trace_path.open("a", encoding="utf-8") as trace_file:
            for record in records:
                trace_file.write(json.dumps(record, separators=(",", ":")) + "\n")

        self._trace_step += 1
        self._trace_prev_position = target_position.copy()
        self._trace_prev_rotation = target_rotation.copy()
        self._trace_prev_joint_target = joint_targets.copy()
        self._trace_prev_actual_joint = post_actual_joints.copy()

    def _request_action_response(self, obs_state, obs_imgs, task_prompts) -> dict:
        def full_query(cache_probe_ms: float = 0.0) -> dict:
            payload_started = time.perf_counter()
            missing_images = [
                source for source in self.image_mapping.values() if source not in obs_imgs
            ]
            if missing_images:
                raise KeyError(f"Need images {missing_images}; got {list(obs_imgs)}")

            def to_uint8(tensor: torch.Tensor) -> np.ndarray:
                image = tensor.detach().clamp(0, 1).mul(255).to(torch.uint8).cpu().numpy()
                return np.ascontiguousarray(image)

            payload = {
                "state": _encode_array(obs_state.detach().to(torch.float32).cpu().numpy()),
                "images": {
                    target: _encode_array(to_uint8(obs_imgs[source]))
                    for target, source in self.image_mapping.items()
                },
                "task": list(task_prompts),
            }
            payload_ms = (time.perf_counter() - payload_started) * 1000.0
            request_started = time.perf_counter()
            response = _request_json("POST", self.endpoint, payload)
            response["_client_request_kind"] = "full"
            response["_client_payload_ms"] = payload_ms
            response["_client_http_ms"] = (
                time.perf_counter() - request_started
            ) * 1000.0
            response["_client_cache_probe_ms"] = cache_probe_ms
            if response.get("ok"):
                self._needs_full_query = False
            return response

        if not self._supports_cached_only or self._needs_full_query:
            return full_query()

        request_started = time.perf_counter()
        response = _request_json("POST", self.endpoint, {"cached_only": True})
        cache_probe_ms = (time.perf_counter() - request_started) * 1000.0
        if response.get("needs_query"):
            return full_query(cache_probe_ms=cache_probe_ms)
        response["_client_request_kind"] = "cached"
        response["_client_payload_ms"] = 0.0
        response["_client_http_ms"] = cache_probe_ms
        response["_client_cache_probe_ms"] = 0.0
        return response

    def select_action(self, obs_state, obs_imgs, task_prompts):
        if obs_state.ndim != 2 or obs_state.shape[1] != int(self.cfg.state_dim):
            raise ValueError(f"Expected state [B, {self.cfg.state_dim}], got {tuple(obs_state.shape)}")
        if self.raw_action_mode == "replay":
            action = self._replay_raw_action()
            response = {
                "ok": True,
                "chunk_index": -1,
                "chunk_actions": -1,
                "chunk_shape": list(action.shape),
                "inference_ran": False,
                "_client_request_kind": "raw_replay",
                "_client_payload_ms": 0.0,
                "_client_http_ms": 0.0,
                "_client_cache_probe_ms": 0.0,
            }
        else:
            response = self._request_action_response(obs_state, obs_imgs, task_prompts)
            if not response.get("ok"):
                raise RuntimeError(f"GR00T 0.6 inference failed: {response}")
            action = _decode_array(response["action"]).astype(np.float32, copy=False)
        if action.ndim != 2 or action.shape[1] != int(self.cfg.action_dim):
            raise ValueError(f"Expected action [B, {self.cfg.action_dim}], got {action.shape}")
        if action.shape[0] != obs_state.shape[0]:
            raise ValueError(
                f"Expected action batch {obs_state.shape[0]}, got {action.shape[0]}"
            )
        if not np.isfinite(action).all():
            raise ValueError("GR00T 0.6 returned non-finite action values")
        if self.raw_action_mode == "record":
            self._record_raw_action(action)
        self._raw_action_step += 1

        if self.checkpoint_kind == "insight_finetuned":
            state = obs_state.detach().to(torch.float32).cpu().numpy()
            if self.action_space == "joint_abs":
                joint_targets = np.clip(action[:, 0:7], _PANDA_LOWER, _PANDA_UPPER)
                target_gripper = np.clip(action[:, 7:8], 0.0, 0.04)
                env_action = np.concatenate(
                    [joint_targets, target_gripper], axis=1
                ).astype(np.float32, copy=False)
                return torch.from_numpy(env_action).to(self.device)

            current_position = state[:, 0:3]
            current_rotation = _quat_wxyz_to_matrix(state[:, 3:7])
            target_position = action[:, 0:3]
            target_rotation = _rotation_6d_to_matrix(action[:, 3:9])
            target_position, target_rotation = self._smooth_targets(
                target_position, target_rotation
            )
            delta_position = _clip_vector_norm(target_position - current_position, self.position_delta_limit)
            delta_rotation_matrix = target_rotation @ np.swapaxes(current_rotation, -1, -2)
            delta_rotation = _clip_vector_norm(
                _matrix_to_axis_angle(delta_rotation_matrix), self.rotation_delta_limit
            )
            current_gripper = state[:, 14:16].mean(axis=1, keepdims=True)
            delta_gripper = action[:, 9:10] - current_gripper
            if self.gripper_delta_limit > 0:
                delta_gripper = np.clip(
                    delta_gripper, -self.gripper_delta_limit, self.gripper_delta_limit
                )
            target_gripper = np.clip(current_gripper + delta_gripper, 0.0, 0.04)

            limited_position = current_position + delta_position
            limited_rotation = _axis_angle_to_matrix(delta_rotation) @ current_rotation
            if self.ee_to_joint_solver == "isaaclab_diffik_physx_bounded":
                target_quat = _matrix_to_quat_wxyz(limited_rotation)
                self._pending_physx_trace = {
                    "response": response,
                    "state": state.copy(),
                    "current_position": current_position.copy(),
                    "current_rotation": current_rotation.copy(),
                    "target_position": limited_position.copy(),
                    "target_rotation": limited_rotation.copy(),
                }
                env_action = np.concatenate(
                    [limited_position, target_quat, target_gripper], axis=1
                ).astype(np.float32, copy=False)
            else:
                ik_started = time.perf_counter()
                joint_targets = self._solve_ee_to_joint(
                    limited_position,
                    limited_rotation,
                    state[:, 7:14],
                )
                response["_client_ik_ms"] = (
                    time.perf_counter() - ik_started
                ) * 1000.0
                self._trace_insight_action(
                    response,
                    state,
                    current_position,
                    current_rotation,
                    limited_position,
                    limited_rotation,
                    joint_targets,
                )
                env_action = np.concatenate(
                    [joint_targets, target_gripper], axis=1
                ).astype(np.float32, copy=False)
        else:
            # oxe_droid_relative_eef_relative_joint: eef_9d (0:9), gripper (9), joints (10:17).
            joints = np.clip(action[:, 10:17], _PANDA_LOWER, _PANDA_UPPER)
            gripper = np.clip(action[:, 9:10], 0.0, 1.0) * 0.04
            env_action = np.concatenate([joints, gripper], axis=1).astype(np.float32, copy=False)
        return torch.from_numpy(env_action).to(self.device)
