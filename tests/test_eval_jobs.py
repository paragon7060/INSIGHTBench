from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from insightbench.eval_jobs import generate_eval_jobs, summarize_jobs


TASK_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "task"


def test_all_benchmark_jobs_match_task_configs() -> None:
    jobs = generate_eval_jobs(TASK_CONFIG_DIR)

    assert summarize_jobs(jobs) == {
        "total": 139,
        "by_category": {"cabinet": 23, "door": 36, "bottle": 80},
        "by_split": {"seen": 73, "unseen": 66},
    }
    assert jobs[0].worker_tsv() == "cabinet\t46130\t0"
    assert jobs[-1].worker_tsv() == "bottle\tb4\t7"


def test_category_and_split_filters_are_composable() -> None:
    jobs = generate_eval_jobs(
        TASK_CONFIG_DIR,
        categories="door,bottle",
        splits="unseen",
    )

    assert summarize_jobs(jobs) == {
        "total": 56,
        "by_category": {"door": 16, "bottle": 40},
        "by_split": {"unseen": 56},
    }
    assert {job.category for job in jobs} == {"door", "bottle"}
    assert {job.split for job in jobs} == {"unseen"}


def test_asset_filter_validates_against_selected_scope() -> None:
    jobs = generate_eval_jobs(
        TASK_CONFIG_DIR,
        categories="bottle",
        splits="seen",
        assets="14b,b17",
    )

    assert len(jobs) == 16
    assert {job.asset for job in jobs} == {"14b", "b17"}

    with pytest.raises(ValueError, match="absent from the chosen"):
        generate_eval_jobs(
            TASK_CONFIG_DIR,
            categories="bottle",
            splits="seen",
            assets="5b",
        )


def test_deployment_launcher_dry_run_writes_filtered_plan(tmp_path: Path) -> None:
    run_dir = tmp_path / "door_seen"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts/eval_batch_persistent.py"),
        "--config",
        "configs/eval/groot.yaml",
        "--checkpoint",
        "/tmp/test-checkpoint",
        "--categories",
        "door",
        "--splits",
        "seen",
        "--assets",
        "99660039960014l",
        "--gpus",
        "0,1",
        "--num-envs",
        "2",
        "--run-dir",
        str(run_dir),
        "--dry-run",
    ]

    result = subprocess.run(command, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["selection"] == {
        "total": 4,
        "by_category": {"door": 4},
        "by_split": {"seen": 4},
    }
    assert manifest["num_envs"] == 2
    assert len((run_dir / "assignments/gpu_0.tsv").read_text().splitlines()) == 2
    assert len((run_dir / "assignments/gpu_1.tsv").read_text().splitlines()) == 2
