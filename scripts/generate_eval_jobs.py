#!/usr/bin/env python3
"""Generate filtered eval batch jobs from task-category configs.

Output format:
    object<TAB>asset<TAB>task_idx
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from insightbench.eval_jobs import generate_eval_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print eval jobs derived from configs/task/*.yaml as TSV."
    )
    parser.add_argument(
        "--task-config-dir",
        default=Path(__file__).resolve().parents[1] / "configs" / "task",
        type=Path,
        help="Directory containing task YAML configs.",
    )
    parser.add_argument(
        "--categories",
        "--objects",
        default="all",
        help="Comma-separated cabinet,door,bottle categories (default: all).",
    )
    parser.add_argument(
        "--splits",
        default="all",
        help="Comma-separated seen,unseen splits (default: all).",
    )
    parser.add_argument(
        "--assets",
        default="all",
        help="Optional comma-separated asset ids within the selected categories/splits.",
    )
    parser.add_argument(
        "--include-split",
        action="store_true",
        help="Print category, split, asset, task_idx instead of the legacy three columns.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        jobs = generate_eval_jobs(
            args.task_config_dir.resolve(),
            categories=args.categories,
            splits=args.splits,
            assets=args.assets,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Job selection error: {exc}", file=sys.stderr)
        return 2
    for job in jobs:
        print(job.plan_tsv() if args.include_split else job.worker_tsv())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
