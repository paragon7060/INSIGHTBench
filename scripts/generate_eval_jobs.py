#!/usr/bin/env python3
"""Generate eval batch jobs from task configs.

Output format:
    object<TAB>asset<TAB>task_idx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:
    OmegaConf = None


DEFAULT_TASK_ORDER = ("cabinet.yaml", "door.yaml", "bottle.yaml")


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
    return parser.parse_args()


def iter_task_configs(task_config_dir: Path) -> Iterable[Path]:
    configs = {path.name: path for path in task_config_dir.glob("*.yaml")}

    for name in DEFAULT_TASK_ORDER:
        path = configs.pop(name, None)
        if path is not None:
            yield path

    for path in sorted(configs.values(), key=lambda item: item.name):
        yield path


def load_task_config(config_path: Path) -> dict:
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)

    if OmegaConf is not None:
        return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    raise RuntimeError("Install PyYAML or OmegaConf to read task YAML configs.")


def asset_order(config: dict) -> list[str]:
    return [str(asset) for asset in (config.get("seen_assets") or [])] + [
        str(asset) for asset in (config.get("unseen_assets") or [])
    ]


def task_count_for_asset(config: dict, asset: str) -> int:
    object_name = config["object"]
    if object_name == "cabinet":
        asset_task_counts = config.get("asset_task_counts")
        if asset_task_counts is None or asset not in asset_task_counts:
            raise ValueError(f"cabinet asset {asset!r} is missing asset_task_counts")
        return int(asset_task_counts[asset])

    task_lib = config.get("task_lib")
    if task_lib is None:
        raise ValueError(f"{object_name!r} config must define task_lib")
    return len(task_lib)


def generate_jobs(task_config_dir: Path) -> Iterable[tuple[str, str, int]]:
    for config_path in iter_task_configs(task_config_dir):
        config = load_task_config(config_path)
        object_name = str(config["object"])

        for asset in asset_order(config):
            for task_idx in range(task_count_for_asset(config, asset)):
                yield object_name, asset, task_idx


def main() -> int:
    args = parse_args()
    task_config_dir = args.task_config_dir.resolve()
    if not task_config_dir.is_dir():
        print(f"Task config directory not found: {task_config_dir}", file=sys.stderr)
        return 1

    for object_name, asset, task_idx in generate_jobs(task_config_dir):
        print(f"{object_name}\t{asset}\t{task_idx}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
