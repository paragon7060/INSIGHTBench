#!/usr/bin/env python3
"""Migrate completed eval videos from the legacy tree to artifact layout v2."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from insightbench.utils.eval_artifacts import resolve_asset_split, video_job_dir


LAYOUT_TEMPLATE = (
    "{scene_key}/{split}/jobs/{object}_{asset}_task{task_idx}/"
    "env{env_idx}/{view}_{status}.mp4"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def _write_json_atomic(path: Path, payload: dict) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temp_path, path)


def _single_video(path: Path, view: str, env_idx: int, legacy: bool) -> Path:
    if legacy:
        candidates = list(path.glob(f"{view}_env{env_idx}_success.mp4"))
        candidates += list(path.glob(f"{view}_env{env_idx}_failure.mp4"))
    else:
        env_dir = path / f"env{env_idx}"
        candidates = list(env_dir.glob(f"{view}_success.mp4"))
        candidates += list(env_dir.glob(f"{view}_failure.mp4"))
    if len(candidates) != 1 or candidates[0].stat().st_size < 1000:
        raise RuntimeError(
            f"Expected one nonempty {view} video for env{env_idx} under {path}; "
            f"found {candidates}"
        )
    return candidates[0]


def migrate_condition(condition_dir: Path, apply: bool) -> dict:
    condition_dir = condition_dir.resolve()
    video_root = condition_dir / "videos"
    results_root = condition_dir / "results"
    manifest_path = condition_dir / "run_manifest.json"
    if not video_root.is_dir() or not results_root.is_dir() or not manifest_path.is_file():
        raise RuntimeError(f"Not an eval condition directory: {condition_dir}")

    manifest = json.loads(manifest_path.read_text())
    num_envs = int(manifest.get("num_envs", 8))
    views = list(manifest.get("video_views") or ["wrist", "right_shoulder"])
    result_paths = sorted(results_root.glob("*.json"))
    moved = 0
    already_v2 = 0
    cleanup_paths = {
        *video_root.rglob("*_temp.mp4"),
        *video_root.rglob("._*"),
        *video_root.rglob(".DS_Store"),
    }

    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        required = {"object", "asset", "task_idx", "scene_key", "attempts"}
        if not required.issubset(result):
            continue
        if int(result["attempts"]) != num_envs:
            raise RuntimeError(f"Unexpected attempts in {result_path}: {result['attempts']}")

        split = result.get("split") or resolve_asset_split(
            result["object"], result["asset"]
        )
        old_job = (
            video_root
            / str(result["asset"])
            / f"task{result['scene_key']}"
            / f"task{result['task_idx']}"
        )
        new_job = video_job_dir(
            video_root,
            result["scene_key"],
            split,
            result["object"],
            result["asset"],
            result["task_idx"],
        )

        for env_idx in range(num_envs):
            for view in views:
                try:
                    source = _single_video(old_job, view, env_idx, legacy=True)
                except RuntimeError:
                    target = _single_video(new_job, view, env_idx, legacy=False)
                    already_v2 += 1
                    continue

                status = "success" if source.name.endswith("_success.mp4") else "failure"
                target = new_job / f"env{env_idx}" / f"{view}_{status}.mp4"
                if target.exists():
                    raise RuntimeError(f"Migration target already exists: {target}")
                if apply:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, target)
                moved += 1

        if apply and result.get("split") != split:
            result["split"] = split
            _write_json_atomic(result_path, result)

    expected = len(result_paths) * num_envs * len(views)
    if moved + already_v2 != expected:
        raise RuntimeError(
            f"Video count mismatch for {condition_dir}: "
            f"planned={moved} existing_v2={already_v2} expected={expected}"
        )

    if apply:
        for cleanup_path in sorted(cleanup_paths):
            if cleanup_path.is_file() or cleanup_path.is_symlink():
                cleanup_path.unlink()
        for directory in sorted(video_root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        manifest["artifact_layout_version"] = 2
        manifest["video_layout"] = LAYOUT_TEMPLATE
        _write_json_atomic(manifest_path, manifest)

    return {
        "condition": str(condition_dir),
        "jobs": len(result_paths),
        "num_envs": num_envs,
        "views": views,
        "moved": moved,
        "already_v2": already_v2,
        "cleanup_files": len(cleanup_paths),
        "expected_videos": expected,
        "applied": apply,
    }


def main() -> int:
    args = parse_args()
    for condition_dir in args.condition:
        print(json.dumps(migrate_condition(condition_dir, args.apply), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
