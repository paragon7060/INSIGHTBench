from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from insightbench.utils.obs import build_obs_images, build_obs_state


def _fake_obs_batch(batch_size: int = 2) -> dict:
    height, width = 4, 5
    image = torch.arange(batch_size * height * width * 3, dtype=torch.uint8).reshape(
        batch_size, height, width, 3
    )

    return {
        "policy": {
            "joint_pos": torch.arange(batch_size * 10, dtype=torch.float32).reshape(batch_size, 10),
            "joint_vel": torch.arange(batch_size * 10, dtype=torch.float32).reshape(batch_size, 10) + 100,
            "ee_position": torch.arange(batch_size * 3, dtype=torch.float32).reshape(batch_size, 3) + 200,
            "ee_quat": torch.arange(batch_size * 4, dtype=torch.float32).reshape(batch_size, 4) + 300,
            "actions": torch.arange(batch_size * 8, dtype=torch.float32).reshape(batch_size, 8) + 400,
            "wrist": image,
            "left_shoulder": image + 1,
            "right_shoulder": image + 2,
            "guide": image + 3,
        }
    }


@pytest.mark.parametrize("policy_type", ["pi0", "diffusion", "instruction_gpt"])
def test_build_obs_state_uses_first_eight_joint_dims(policy_type: str) -> None:
    obs_batch = _fake_obs_batch()

    state = build_obs_state(obs_batch, policy_type)

    assert torch.equal(state, obs_batch["policy"]["joint_pos"][:, :8])
    assert state.shape == (2, 8)


def test_build_obs_state_smolvla_concatenates_ee_and_joints() -> None:
    obs_batch = _fake_obs_batch()
    p = obs_batch["policy"]

    state = build_obs_state(obs_batch, "smolvla")

    expected = torch.cat([p["ee_position"], p["ee_quat"], p["joint_pos"]], dim=1)
    assert torch.equal(state, expected)
    assert state.shape == (2, 17)


def test_build_obs_state_groot_concatenates_and_drops_last_two_dims() -> None:
    obs_batch = _fake_obs_batch()
    p = obs_batch["policy"]

    state = build_obs_state(obs_batch, "groot")

    expected = torch.cat(
        [p["ee_position"], p["ee_quat"], p["joint_pos"], p["joint_vel"], p["actions"]],
        dim=1,
    )[:, :-2]
    assert torch.equal(state, expected)
    assert state.shape == (2, 33)


def _assert_image_tensor(tensor: torch.Tensor) -> None:
    assert tensor.shape == (2, 3, 4, 5)
    assert tensor.dtype == torch.float32
    assert torch.all(tensor >= 0.0)
    assert torch.all(tensor <= 1.0)


def test_build_obs_images_uses_guide_camera_by_default() -> None:
    images = build_obs_images(_fake_obs_batch(), infer_type="guide", guide_cam=True)

    assert set(images) == {
        "observation.images.wrist",
        "observation.images.right_shoulder",
        "observation.images.guide",
    }
    for tensor in images.values():
        _assert_image_tensor(tensor)


def test_build_obs_images_can_disable_guide_camera() -> None:
    images = build_obs_images(_fake_obs_batch(), infer_type="guide", guide_cam=False)

    assert set(images) == {
        "observation.images.wrist",
        "observation.images.right_shoulder",
        "observation.images.left_shoulder",
    }
    for tensor in images.values():
        _assert_image_tensor(tensor)


def test_build_obs_images_semantic_mode_uses_semantic_wrist_key() -> None:
    images = build_obs_images(_fake_obs_batch(), infer_type="sem", guide_cam=True)

    assert set(images) == {
        "observation.images.wrist",
        "observation.images.wrist_semantic",
        "observation.images.right_shoulder",
    }
    for tensor in images.values():
        _assert_image_tensor(tensor)
