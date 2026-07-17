#!/usr/bin/env python3
"""Deployment-oriented multi-GPU launcher for persistent INSIGHTBench eval."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from insightbench.eval_jobs import (
    CATEGORY_ORDER,
    SPLIT_ORDER,
    EvalJob,
    generate_eval_jobs,
    parse_assets,
    parse_selection,
    summarize_jobs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one persistent Isaac/policy worker per GPU with filtered "
            "benchmark categories."
        )
    )
    parser.add_argument("--config", default="configs/eval/pi0.yaml")
    parser.add_argument("--checkpoint", help="Shortcut for policy.checkpoint=<path-or-repo>.")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=8)
    parser.add_argument("--gpu", "--gpus", dest="gpus", default="0")
    parser.add_argument(
        "--categories",
        "--objects",
        dest="categories",
        default="all",
        help="cabinet,door,bottle or comma-separated combination (default: all).",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="seen, unseen, or seen,unseen (default: all).",
    )
    parser.add_argument("--assets", default="all", help="Optional comma-separated asset ids.")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--run-dir", "--run_dir", type=Path, help="Structured run output root."
    )
    output_group.add_argument(
        "--log-dir",
        "--log_dir",
        type=Path,
        help="Backward-compatible log path; its parent becomes the run root.",
    )
    parser.add_argument(
        "--task-config-dir",
        "--task_config_dir",
        type=Path,
        default=REPO_ROOT / "configs/task",
    )
    parser.add_argument("--max-jobs", "--max_jobs", type=int, default=0)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--pos-rand", "--pos_rand", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs already marked ok in this run dir.",
    )
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="KEY=VALUE",
        help="Direct OmegaConf overrides, matching scripts/evaluate.py.",
    )
    return parser.parse_args()


def _csv_tokens(value: str, label: str) -> tuple[str, ...]:
    tokens = tuple(item.strip() for item in value.split(",") if item.strip())
    if not tokens or len(tokens) != len(set(tokens)):
        raise ValueError(f"{label} entries must be non-empty and unique: {value!r}")
    return tokens


def _default_run_dir(config: str, categories: Sequence[str]) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    category_token = "-".join(categories)
    return REPO_ROOT / "outputs/eval_runs" / f"{Path(config).stem}_{category_token}_{stamp}"


def _resolve_config_path(value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, REPO_ROOT / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Evaluation config not found: {value}")


def _has_override(overrides: Sequence[str], key: str) -> bool:
    return any(item.partition("=")[0] == key for item in overrides)


def _override_value(overrides: Sequence[str], key: str) -> str | None:
    for item in reversed(overrides):
        override_key, separator, value = item.partition("=")
        if separator and override_key == key:
            return value
    return None


def _read_ok_jobs(status_dir: Path) -> set[tuple[str, str, int]]:
    completed = set()
    if not status_dir.is_dir():
        return completed
    for path in status_dir.glob("gpu_*.tsv"):
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            fields = raw_line.split("\t")
            if len(fields) == 4 and fields[3] == "ok":
                completed.add((fields[0], fields[1], int(fields[2])))
    return completed


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() or None


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_job_plan(path: Path, jobs: Sequence[EvalJob]) -> None:
    path.write_text(_job_plan_text(jobs), encoding="utf-8")


def _job_plan_text(jobs: Sequence[EvalJob]) -> str:
    return (
        "category\tsplit\tasset\ttask_idx\n"
        + "".join(f"{job.plan_tsv()}\n" for job in jobs)
    )


def _worker_command(
    *,
    args: argparse.Namespace,
    job_file: Path,
    log_dir: Path,
    status_file: Path,
    overrides: Sequence[str],
    append_status: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/evaluate_persistent.py"),
        "--config",
        args.config,
        "--job-file",
        str(job_file),
        "--log-dir",
        str(log_dir),
        "--status-file",
        str(status_file),
        "--num_envs",
        str(args.num_envs),
        "--enable_cameras",
        "--headless",
    ]
    if args.pos_rand:
        command.append("--pos_rand")
    if append_status:
        command.append("--append-status")
    command.extend(overrides)
    return command


def _retry_command(
    args: argparse.Namespace, job: EvalJob, gpu: str, overrides: Sequence[str]
) -> str:
    command = [
        "env",
        "PYTHONUNBUFFERED=1",
        f"CUDA_VISIBLE_DEVICES={gpu}",
        sys.executable,
        "scripts/evaluate.py",
        "--config",
        args.config,
        "--object",
        job.category,
        "--asset_path",
        job.asset,
        "--task_idx",
        str(job.task_idx),
        "--num_envs",
        str(args.num_envs),
        "--enable_cameras",
        "--headless",
    ]
    if args.pos_rand:
        command.append("--pos_rand")
    command.extend(overrides)
    return shlex.join(command)


def main() -> int:
    args = parse_args()
    if args.num_envs <= 0 or args.max_jobs < 0:
        raise SystemExit("--num-envs must be positive and --max-jobs must be non-negative")

    try:
        args.config = str(_resolve_config_path(args.config))
        categories = parse_selection(args.categories, allowed=CATEGORY_ORDER, label="categories")
        splits = parse_selection(args.splits, allowed=SPLIT_ORDER, label="splits")
        assets = parse_assets(args.assets)
        gpus = _csv_tokens(args.gpus, "GPU")
        jobs = generate_eval_jobs(
            args.task_config_dir.expanduser().resolve(),
            categories=categories,
            splits=splits,
            assets=assets,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    if args.max_jobs:
        jobs = jobs[: args.max_jobs]
    if not jobs:
        raise SystemExit("The category/split/asset selection produced no evaluation jobs")

    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
        log_dir = run_dir / "logs"
    elif args.log_dir:
        log_dir = args.log_dir.expanduser().resolve()
        run_dir = log_dir.parent
    else:
        run_dir = _default_run_dir(args.config, categories)
        log_dir = run_dir / "logs"
    status_dir = run_dir / "status"
    assignment_dir = run_dir / "assignments"
    result_dir = run_dir / "results"
    video_dir = run_dir / "videos"
    for path in (run_dir, log_dir, status_dir, assignment_dir, result_dir, video_dir):
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists() and not args.resume:
        raise SystemExit(f"Run directory already contains a manifest; use --resume: {run_dir}")
    if args.resume and not manifest_path.is_file():
        raise SystemExit(f"Cannot resume: run manifest not found: {manifest_path}")

    overrides = [*args.override, *args.overrides]
    if args.checkpoint:
        if _has_override(overrides, "policy.checkpoint"):
            raise SystemExit(
                "Use either --checkpoint or --override policy.checkpoint=..., not both"
            )
        overrides.append(f"policy.checkpoint={args.checkpoint}")
    if args.no_video and _has_override(overrides, "eval.save_video"):
        raise SystemExit("Use either --no-video or an eval.save_video override, not both")
    if args.no_video:
        overrides.append("eval.save_video=false")
    if not _has_override(overrides, "eval.results_dir"):
        overrides.append(f"eval.results_dir={result_dir}")
    if not _has_override(overrides, "eval.video_dir"):
        overrides.append(f"eval.video_dir={video_dir}")

    prior_manifest = None
    if args.resume:
        prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs_path = run_dir / "jobs.tsv"
        if not jobs_path.is_file() or jobs_path.read_text(encoding="utf-8") != _job_plan_text(jobs):
            raise SystemExit(
                "Resume selection differs from jobs.tsv; rerun with the original "
                "--categories/--splits/--assets/--max-jobs values"
            )
        expected_resume_values = {
            "config": args.config,
            "checkpoint": _override_value(overrides, "policy.checkpoint"),
            "num_envs": args.num_envs,
            "pos_rand": args.pos_rand,
            "overrides": overrides,
        }
        mismatches = [
            key
            for key, value in expected_resume_values.items()
            if prior_manifest.get(key) != value
        ]
        if mismatches:
            raise SystemExit(
                "Resume configuration differs from run_manifest.json for: "
                + ", ".join(mismatches)
            )

    completed_before = _read_ok_jobs(status_dir) if args.resume else set()
    pending_jobs = [job for job in jobs if job.key not in completed_before]
    assignments = {gpu: [] for gpu in gpus}
    for index, job in enumerate(pending_jobs):
        assignments[gpus[index % len(gpus)]].append(job)

    _write_job_plan(run_dir / "jobs.tsv", jobs)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "created_at": prior_manifest["created_at"] if prior_manifest else now,
        "command": [sys.executable, *sys.argv],
        "repo_root": str(REPO_ROOT),
        "git_commit": _git_commit(),
        "config": str(Path(args.config).expanduser()),
        "checkpoint": _override_value(overrides, "policy.checkpoint"),
        "categories": list(categories),
        "splits": list(splits),
        "assets": list(assets) if assets else "all",
        "gpus": list(gpus),
        "num_envs": args.num_envs,
        "pos_rand": args.pos_rand,
        "overrides": overrides,
        "selection": summarize_jobs(jobs),
        "pending_jobs": len(pending_jobs),
        "skipped_completed_jobs": len(jobs) - len(pending_jobs),
        "run_dir": str(run_dir),
        "dry_run": args.dry_run,
        "resume": args.resume,
        "status": "planned" if args.dry_run else "running",
    }
    if prior_manifest:
        manifest["resumed_at"] = now
    _write_json(manifest_path, manifest)

    print(f"Run directory : {run_dir}")
    print(f"Categories    : {', '.join(categories)}")
    print(f"Splits        : {', '.join(splits)}")
    print(f"GPUs          : {', '.join(gpus)}")
    print(f"Jobs          : {len(jobs)} total, {len(pending_jobs)} pending")
    summary = summarize_jobs(jobs)
    print(f"By category   : {summary['by_category']}")
    print(f"By split      : {summary['by_split']}")

    worker_specs = []
    for gpu, gpu_jobs in assignments.items():
        assignment_path = assignment_dir / f"gpu_{gpu}.tsv"
        assignment_path.write_text(
            "".join(f"{job.worker_tsv()}\n" for job in gpu_jobs), encoding="utf-8"
        )
        status_path = status_dir / f"gpu_{gpu}.tsv"
        command = _worker_command(
            args=args,
            job_file=assignment_path,
            log_dir=log_dir,
            status_file=status_path,
            overrides=overrides,
            append_status=args.resume,
        )
        worker_specs.append((gpu, gpu_jobs, status_path, command))
        print(f"GPU {gpu:<4}: {len(gpu_jobs)} jobs")

    if args.dry_run or not pending_jobs:
        manifest["status"] = "dry_run" if args.dry_run else "complete"
        if not pending_jobs:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            manifest["completed_jobs"] = len(jobs)
            manifest["failed_or_uncompleted_jobs"] = 0
        _write_json(manifest_path, manifest)
        message = (
            "No Isaac process started."
            if args.dry_run
            else "All selected jobs were already complete."
        )
        print(message)
        return 0

    processes = []
    try:
        for gpu, gpu_jobs, status_path, command in worker_specs:
            if not gpu_jobs:
                continue
            worker_log_path = log_dir / f"worker_gpu{gpu}.log"
            log_mode = "a" if args.resume else "w"
            worker_log = worker_log_path.open(log_mode, encoding="utf-8", buffering=1)
            env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": gpu}
            process = subprocess.Popen(
                command, cwd=REPO_ROOT, env=env, stdout=worker_log, stderr=subprocess.STDOUT
            )
            processes.append((gpu, process, worker_log))
            print(f"GPU {gpu}: worker started pid={process.pid}")
        worker_failures = 0
        for gpu, process, worker_log in processes:
            return_code = process.wait()
            worker_log.close()
            if return_code:
                worker_failures += 1
                print(f"GPU {gpu}: worker failed rc={return_code}")
            else:
                print(f"GPU {gpu}: worker complete")
    except KeyboardInterrupt:
        for _, process, _ in processes:
            process.terminate()
        raise

    completed_after = _read_ok_jobs(status_dir)
    failed_jobs = [job for job in jobs if job.key not in completed_after]
    retry_path = run_dir / "retry_failed.sh"
    assignment_gpu = {
        job.key: gpu for gpu, gpu_jobs, _, _ in worker_specs for job in gpu_jobs
    }
    retry_lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"cd {shlex.quote(str(REPO_ROOT))}"]
    for job in failed_jobs:
        gpu = assignment_gpu.get(job.key, gpus[0])
        retry_lines.extend([
            f"# {job.category}/{job.split} {job.asset} task{job.task_idx}",
            _retry_command(args, job, gpu, overrides),
        ])
    retry_path.write_text("\n".join(retry_lines) + "\n", encoding="utf-8")
    retry_path.chmod(0o755)

    manifest.update(
        {
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "complete" if not failed_jobs else "failed",
            "completed_jobs": len(jobs) - len(failed_jobs),
            "failed_or_uncompleted_jobs": len(failed_jobs),
            "worker_failures": worker_failures,
        }
    )
    _write_json(manifest_path, manifest)
    print(
        f"Done: completed={len(jobs) - len(failed_jobs)}/{len(jobs)} "
        f"failed={len(failed_jobs)} worker_failures={worker_failures}"
    )
    return 1 if failed_jobs else 0


if __name__ == "__main__":
    raise SystemExit(main())
