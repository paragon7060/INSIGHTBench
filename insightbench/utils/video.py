"""Video recording utility for parallel evaluation environments."""

from __future__ import annotations

import os
from typing import Sequence

import cv2
import numpy as np
import torch


class VideoRecorder:
    """Records evaluation episodes as mp4 videos, one file per env per camera view.

    Usage::

        recorder = VideoRecorder("./outputs/videos", ["wrist", "right_shoulder"], num_envs=8, fps=10)
        recorder.open(frame_shape=(224, 224, 3), asset_name="door_01", task_name="3a", task_idx=0)
        for step in range(eval_steps):
            recorder.write_frame(obs_batch)
        recorder.close_and_rename(success_mask=[True, False, ...], asset_name="door_01", task_name="3a", task_idx=0)
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
        self._temp_paths: dict[str, list[str]] = {}

    def open(
        self, frame_shape: tuple, asset_name: str, task_name: str, task_idx: int
    ) -> None:
        """Create VideoWriter objects. Call once before the eval loop."""
        task_dir = os.path.join(
            self.video_dir, asset_name, f"task{task_name}", f"task{task_idx}"
        )
        os.makedirs(task_dir, exist_ok=True)
        h, w = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        for view in self.views:
            self._writers[view] = []
            self._temp_paths[view] = []
            for i in range(self.num_envs):
                tmp = os.path.join(task_dir, f"{view}_env{i}_temp.mp4")
                self._temp_paths[view].append(tmp)
                writer = cv2.VideoWriter(tmp, fourcc, self.fps, (w, h))
                if not writer.isOpened():
                    print(f"[VideoRecorder] WARNING: could not open writer for {tmp}")
                self._writers[view].append(writer)

    def write_frame(self, obs_batch: dict) -> None:
        """Append one frame from each env to each view's video."""
        for view in self.views:
            frames: torch.Tensor = obs_batch["policy"][view]  # (B, H, W, 3) uint8 RGB
            for i in range(self.num_envs):
                frame_bgr = cv2.cvtColor(
                    frames[i].cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                self._writers[view][i].write(frame_bgr)

    def close_and_rename(
        self,
        success_mask: Sequence[bool],
        asset_name: str,
        task_name: str,
        task_idx: int,
    ) -> None:
        """Release writers and rename temp files with success/failure suffix."""
        for view in self.views:
            for w in self._writers[view]:
                w.release()

        task_dir = os.path.join(
            self.video_dir, asset_name, f"task{task_name}", f"task{task_idx}"
        )
        for view in self.views:
            for i, is_success in enumerate(success_mask):
                temp = self._temp_paths[view][i]
                status = "success" if is_success else "failure"
                final = os.path.join(
                    task_dir,
                    f"{view}_env{i}_{status}.mp4",
                )
                if os.path.exists(temp):
                    os.rename(temp, final)
                else:
                    print(f"[VideoRecorder] WARNING: temp file not found: {temp}")

        self._writers.clear()
        self._temp_paths.clear()
