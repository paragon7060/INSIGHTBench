"""Canonical paths and metadata helpers for evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml


EvalSplit = Literal["seen", "unseen"]


def _task_config_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "task"


def resolve_asset_split(
    object_name: str,
    asset_name: str,
    task_config_dir: str | Path | None = None,
) -> EvalSplit:
    """Return the configured seen/unseen split for one evaluation asset."""
    config_dir = Path(task_config_dir) if task_config_dir is not None else _task_config_dir()
    config_path = config_dir / f"{object_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Task config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}

    asset_name = str(asset_name)
    seen_assets = {str(asset) for asset in config.get("seen_assets") or []}
    unseen_assets = {str(asset) for asset in config.get("unseen_assets") or []}
    in_seen = asset_name in seen_assets
    in_unseen = asset_name in unseen_assets
    if in_seen == in_unseen:
        raise ValueError(
            f"Asset {asset_name!r} for {object_name!r} must appear in exactly one "
            f"of seen_assets or unseen_assets in {config_path}"
        )
    return "seen" if in_seen else "unseen"


def eval_job_name(object_name: str, asset_name: str, task_idx: int) -> str:
    """Return the stable folder name for one asset-task evaluation job."""
    return f"{object_name}_{asset_name}_task{int(task_idx)}"


def video_job_dir(
    video_dir: str | Path,
    scene_key: str,
    split: EvalSplit | str,
    object_name: str,
    asset_name: str,
    task_idx: int,
) -> Path:
    """Return videos/<scene>/<split>/jobs/<job> for artifact layout v2."""
    if split not in {"seen", "unseen"}:
        raise ValueError(f"Unknown eval split: {split!r}")
    return (
        Path(video_dir)
        / str(scene_key)
        / str(split)
        / "jobs"
        / eval_job_name(object_name, asset_name, task_idx)
    )


def env_video_dir(job_dir: str | Path, env_idx: int) -> Path:
    """Return the folder for one parallel evaluation environment."""
    env_idx = int(env_idx)
    if env_idx < 0:
        raise ValueError("env_idx must be non-negative")
    return Path(job_dir) / f"env{env_idx}"
