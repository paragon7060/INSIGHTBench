#!/usr/bin/env python3
"""Validate one completed evaluation job using artifact layout v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--num-envs", required=True, type=int)
    parser.add_argument("--view", action="append", dest="views", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.result.is_file() or not args.trace.is_file() or args.trace.stat().st_size == 0:
        return 1

    try:
        result = json.loads(args.result.read_text())
    except Exception:
        return 1

    if int(result.get("attempts", -1)) != args.num_envs:
        return 1

    split = result.get("split")
    if split not in {"seen", "unseen"}:
        return 1
    job_name = (
        f"{result['object']}_{result['asset']}_task{int(result['task_idx'])}"
    )
    job_dir = (
        args.video_dir
        / str(result["scene_key"])
        / split
        / "jobs"
        / job_name
    )

    for env_idx in range(args.num_envs):
        env_dir = job_dir / f"env{env_idx}"
        for view in args.views:
            matches = list(env_dir.glob(f"{view}_success.mp4"))
            matches += list(env_dir.glob(f"{view}_failure.mp4"))
            if len(matches) != 1 or matches[0].stat().st_size < 1000:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
