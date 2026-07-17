from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from insightbench.utils.eval_artifacts import (
    env_video_dir,
    eval_job_name,
    resolve_asset_split,
    video_job_dir,
)
from insightbench.utils.video import VideoRecorder


def _write_config(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "cabinet.yaml").write_text(
        yaml.safe_dump(
            {
                "object": "cabinet",
                "seen_assets": ["46130"],
                "unseen_assets": ["19179"],
            }
        )
    )


def test_resolve_asset_split_from_task_config(tmp_path) -> None:
    _write_config(tmp_path)
    assert resolve_asset_split("cabinet", "46130", tmp_path) == "seen"
    assert resolve_asset_split("cabinet", "19179", tmp_path) == "unseen"


def test_resolve_asset_split_rejects_unlisted_asset(tmp_path) -> None:
    _write_config(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        resolve_asset_split("cabinet", "missing", tmp_path)


def test_video_layout_v2_path() -> None:
    job_dir = video_job_dir(
        "/eval/videos", "1ext", "seen", "cabinet", "46130", 0
    )
    assert job_dir == Path(
        "/eval/videos/1ext/seen/jobs/cabinet_46130_task0"
    )
    assert env_video_dir(job_dir, 7) == job_dir / "env7"
    assert eval_job_name("cabinet", "46130", 0) == "cabinet_46130_task0"


def test_video_recorder_writes_per_env_view_files(tmp_path) -> None:
    recorder = VideoRecorder(
        str(tmp_path), ["wrist", "right_shoulder"], num_envs=2, fps=10
    )
    recorder.open(
        (32, 32, 3),
        object_name="cabinet",
        asset_name="46130",
        scene_key="1ext",
        split="seen",
        task_idx=0,
    )
    frame = torch.zeros((2, 32, 32, 3), dtype=torch.uint8)
    recorder.write_frame(
        {"policy": {"wrist": frame, "right_shoulder": frame}}
    )
    recorder.close_and_rename([True, False])

    job_dir = tmp_path / "1ext/seen/jobs/cabinet_46130_task0"
    assert (job_dir / "env0/wrist_success.mp4").is_file()
    assert (job_dir / "env0/right_shoulder_success.mp4").is_file()
    assert (job_dir / "env1/wrist_failure.mp4").is_file()
    assert (job_dir / "env1/right_shoulder_failure.mp4").is_file()
    assert not list(job_dir.rglob("*_temp.mp4"))
