"""LeRobot version compatibility shims.

InsightBench supports:
  - lerobot fork v0.3.4 (custom, includes GuideVLA)
  - lerobot official v0.4.x (latest PyPI)

Key API differences handled here:
  - add_frame(): task/timestamp as separate args (v0.3.x) vs inside frame dict (v0.4.x)
  - LeRobotDatasetMetadata: root=None triggers Hub download in both versions
"""

from __future__ import annotations

from packaging.version import Version

import lerobot

_LEROBOT_VERSION = Version(lerobot.__version__)
_IS_V04 = _LEROBOT_VERSION >= Version("0.4.0")


def lerobot_version() -> str:
    return str(_LEROBOT_VERSION)


def add_frame(
    dataset,
    frame: dict,
    task: str,
    timestamp: float | None = None,
) -> None:
    """Version-safe wrapper around LeRobotDataset.add_frame().

    v0.3.x: add_frame(frame, task, timestamp)
    v0.4.x: add_frame({**frame, "task": task, "timestamp": timestamp})
    """
    if _IS_V04:
        payload = dict(frame)
        payload["task"] = task
        if timestamp is not None:
            payload["timestamp"] = timestamp
        dataset.add_frame(payload)
    else:
        dataset.add_frame(frame, task=task, timestamp=timestamp)


def create_dataset(
    repo_id: str,
    fps: int,
    features: dict,
    root: str | None = None,
    image_writer_threads: int = 4,
):
    """Version-safe LeRobotDataset.create() wrapper."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    common = dict(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        image_writer_threads=image_writer_threads,
    )
    if _IS_V04:
        # v0.4.x added vcodec, streaming_encoding, etc. — use defaults
        return LeRobotDataset.create(**common)
    else:
        return LeRobotDataset.create(**common)


def load_dataset(repo_id: str, root: str | None = None):
    """Version-safe LeRobotDataset load."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(repo_id, root=root)


def load_policy_with_stats(policy_cls, checkpoint: str, stats: dict):
    """Load a policy with dataset_stats, handling version differences.

    Both v0.3.x and v0.4.x accept dataset_stats via **kwargs → __init__.
    This wrapper makes the intent explicit and provides a single call site.
    """
    return policy_cls.from_pretrained(checkpoint, dataset_stats=stats)


DATASET_FORMAT_VERSION = "v3.0" if _IS_V04 else "v2.1"
