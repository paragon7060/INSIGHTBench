"""Result serialization utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path


def save_result(
    results_dir: str,
    obj_name: str,
    asset_path: str,
    task_idx: int,
    scene_key: str,
    successes: int,
    attempts: int,
) -> str:
    """Write a single JSON result file and return its path."""
    os.makedirs(results_dir, exist_ok=True)

    result = {
        "object":    obj_name,
        "asset":     asset_path,
        "task_idx":  task_idx,
        "scene_key": scene_key,
        "attempts":  attempts,
        "successes": successes,
        "rate":      round(successes / attempts, 4) if attempts > 0 else 0.0,
    }

    filename = f"{obj_name}_{asset_path}_task{task_idx}.json"
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w") as f:
        json.dump(result, f, indent=4)

    print(f"[Results] {successes}/{attempts} — saved to {filepath}")
    return filepath


def aggregate_results(results_dir: str) -> dict:
    """Read all JSON result files in a directory and compute overall stats."""
    files = list(Path(results_dir).glob("*.json"))
    if not files:
        print(f"[Results] No result files found in {results_dir}")
        return {}

    total_attempts = 0
    total_successes = 0
    per_object: dict[str, dict] = {}

    for f in sorted(files):
        with open(f) as fp:
            r = json.load(fp)
        obj = r["object"]
        if obj not in per_object:
            per_object[obj] = {"successes": 0, "attempts": 0}
        per_object[obj]["successes"] += r["successes"]
        per_object[obj]["attempts"]  += r["attempts"]
        total_successes += r["successes"]
        total_attempts  += r["attempts"]

    summary = {
        "total": {
            "successes": total_successes,
            "attempts":  total_attempts,
            "rate":      round(total_successes / total_attempts, 4) if total_attempts else 0.0,
        },
        "per_object": {
            obj: {**v, "rate": round(v["successes"] / v["attempts"], 4) if v["attempts"] else 0.0}
            for obj, v in per_object.items()
        },
    }

    print(f"[Results] Overall: {total_successes}/{total_attempts} "
          f"({summary['total']['rate']*100:.1f}%)")
    for obj, s in summary["per_object"].items():
        print(f"  {obj}: {s['successes']}/{s['attempts']} ({s['rate']*100:.1f}%)")

    return summary
