"""Video recording utility for parallel evaluation environments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from insightbench.utils.eval_artifacts import env_video_dir, video_job_dir


class VideoRecorder:
    """Record one mp4 per environment and camera view using artifact layout v2.

    Layout::

        videos/<scene>/<split>/jobs/<object>_<asset>_task<idx>/env<idx>/
            wrist_<success|failure>.mp4
            right_shoulder_<success|failure>.mp4
    """

    def __init__(
        self,
        video_dir: str,
        views: Sequence[str],
        num_envs: int,
        fps: int = 10,
        codec: str = "mp4v",
    ):
        self.video_dir = video_dir
        self.views = list(views)
        self.num_envs = num_envs
        self.fps = fps
        self.codec = codec
        self._writers: dict[str, list[cv2.VideoWriter]] = {}
        self._temp_paths: dict[str, list[Path]] = {}
        self._job_dir: Path | None = None

    def open(
        self,
        frame_shape: tuple,
        *,
        object_name: str,
        asset_name: str,
        scene_key: str,
        split: str,
        task_idx: int,
    ) -> None:
        """Create VideoWriter objects. Call once before the eval loop."""
        if self._writers:
            raise RuntimeError("VideoRecorder is already open")

        self._job_dir = video_job_dir(
            self.video_dir,
            scene_key,
            split,
            object_name,
            asset_name,
            task_idx,
        )
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        for view in self.views:
            self._writers[view] = []
            self._temp_paths[view] = []
            for env_idx in range(self.num_envs):
                env_dir = env_video_dir(self._job_dir, env_idx)
                env_dir.mkdir(parents=True, exist_ok=True)
                temp_path = env_dir / f"{view}_temp.mp4"
                for stale_name in (
                    f"{view}_temp.mp4",
                    f"{view}_success.mp4",
                    f"{view}_failure.mp4",
                ):
                    stale_path = env_dir / stale_name
                    if stale_path.exists():
                        stale_path.unlink()
                self._temp_paths[view].append(temp_path)
                writer = cv2.VideoWriter(str(temp_path), fourcc, self.fps, (w, h))
                if not writer.isOpened():
                    print(f"[VideoRecorder] WARNING: could not open writer for {temp_path}")
                self._writers[view].append(writer)

    def write_frame(self, obs_batch: dict) -> None:
        """Append one frame from each env to each view's video."""
        for view in self.views:
            frames: torch.Tensor = obs_batch["policy"][view]  # (B, H, W, 3) uint8 RGB
            for env_idx in range(self.num_envs):
                frame_bgr = cv2.cvtColor(
                    frames[env_idx].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                self._writers[view][env_idx].write(frame_bgr)

    def close_and_rename(self, success_mask: Sequence[bool]) -> None:
        """Release writers and rename temporary files with success/failure suffixes."""
        if len(success_mask) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} success values, received {len(success_mask)}"
            )

        for view in self.views:
            for writer in self._writers[view]:
                writer.release()

        for view in self.views:
            for env_idx, is_success in enumerate(success_mask):
                temp_path = self._temp_paths[view][env_idx]
                status = "success" if is_success else "failure"
                final_path = temp_path.with_name(f"{view}_{status}.mp4")
                if temp_path.exists():
                    temp_path.rename(final_path)
                else:
                    print(f"[VideoRecorder] WARNING: temp file not found: {temp_path}")

        self._writers.clear()
        self._temp_paths.clear()
        self._job_dir = None
