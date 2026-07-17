"""Evaluation job planning from INSIGHTBench task-category configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - OmegaConf fallback covers deployed envs.
    yaml = None

try:
    from omegaconf import OmegaConf
except ModuleNotFoundError:  # pragma: no cover - PyYAML is the normal lightweight path.
    OmegaConf = None


CATEGORY_ORDER = ("cabinet", "door", "bottle")
SPLIT_ORDER = ("seen", "unseen")
DEFAULT_TASK_CONFIG_ORDER = tuple(f"{category}.yaml" for category in CATEGORY_ORDER)


@dataclass(frozen=True, slots=True)
class EvalJob:
    """One benchmark job with explicit category and generalization split."""

    category: str
    split: str
    asset: str
    task_idx: int

    @property
    def key(self) -> tuple[str, str, int]:
        return self.category, self.asset, self.task_idx

    def worker_tsv(self) -> str:
        """Three-column contract consumed by evaluate_persistent.py."""
        return f"{self.category}\t{self.asset}\t{self.task_idx}"

    def plan_tsv(self) -> str:
        return f"{self.category}\t{self.split}\t{self.asset}\t{self.task_idx}"

    def to_dict(self) -> dict[str, str | int]:
        return asdict(self)


def parse_selection(
    value: str | Sequence[str] | None,
    *,
    allowed: Sequence[str],
    label: str,
) -> tuple[str, ...]:
    """Normalize comma/list selectors while retaining canonical order."""
    if value is None:
        return tuple(allowed)
    raw = value.split(",") if isinstance(value, str) else list(value)
    tokens = {str(item).strip().lower() for item in raw if str(item).strip()}
    if not tokens or tokens == {"all"}:
        return tuple(allowed)
    if "all" in tokens:
        raise ValueError(f"{label}: 'all' cannot be combined with explicit values")
    unknown = tokens.difference(allowed)
    if unknown:
        raise ValueError(
            f"Unknown {label}: {', '.join(sorted(unknown))}; "
            f"choose from {', '.join(allowed)} or all"
        )
    return tuple(item for item in allowed if item in tokens)


def parse_assets(value: str | Sequence[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    raw = value.split(",") if isinstance(value, str) else list(value)
    assets = tuple(dict.fromkeys(str(item).strip() for item in raw if str(item).strip()))
    if not assets or assets == ("all",):
        return None
    if "all" in assets:
        raise ValueError("assets: 'all' cannot be combined with explicit asset ids")
    return assets


def iter_task_configs(task_config_dir: Path) -> Iterable[Path]:
    configs = {path.name: path for path in task_config_dir.glob("*.yaml")}
    for name in DEFAULT_TASK_CONFIG_ORDER:
        path = configs.pop(name, None)
        if path is not None:
            yield path
    yield from sorted(configs.values(), key=lambda item: item.name)


def load_task_config(config_path: Path) -> dict:
    if yaml is not None:
        with config_path.open("r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
    if OmegaConf is not None:
        return OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    raise RuntimeError("Install PyYAML or OmegaConf to read task YAML configs.")


def task_count_for_asset(config: dict, asset: str) -> int:
    category = str(config["object"])
    if category == "cabinet":
        asset_task_counts = config.get("asset_task_counts")
        if asset_task_counts is None or asset not in asset_task_counts:
            raise ValueError(f"cabinet asset {asset!r} is missing asset_task_counts")
        return int(asset_task_counts[asset])
    task_lib = config.get("task_lib")
    if task_lib is None:
        raise ValueError(f"{category!r} config must define task_lib")
    return len(task_lib)


def generate_eval_jobs(
    task_config_dir: Path,
    *,
    categories: str | Sequence[str] | None = None,
    splits: str | Sequence[str] | None = None,
    assets: str | Sequence[str] | None = None,
) -> list[EvalJob]:
    """Generate a deterministic category→split→asset→task job plan."""
    if not task_config_dir.is_dir():
        raise FileNotFoundError(f"Task config directory not found: {task_config_dir}")
    selected_categories = parse_selection(
        categories, allowed=CATEGORY_ORDER, label="categories"
    )
    selected_splits = parse_selection(splits, allowed=SPLIT_ORDER, label="splits")
    selected_assets = parse_assets(assets)
    asset_filter = set(selected_assets or ())
    found_assets: set[str] = set()
    jobs: list[EvalJob] = []

    configs_by_category = {}
    for config_path in iter_task_configs(task_config_dir):
        config = load_task_config(config_path)
        category = str(config["object"])
        if category in configs_by_category:
            raise ValueError(f"Duplicate task category config: {category}")
        configs_by_category[category] = config

    missing_categories = set(selected_categories).difference(configs_by_category)
    if missing_categories:
        raise ValueError(f"Missing task category configs: {', '.join(sorted(missing_categories))}")

    for category in selected_categories:
        config = configs_by_category[category]
        for split in selected_splits:
            for raw_asset in config.get(f"{split}_assets") or []:
                asset = str(raw_asset)
                if asset_filter and asset not in asset_filter:
                    continue
                found_assets.add(asset)
                for task_idx in range(task_count_for_asset(config, asset)):
                    jobs.append(EvalJob(category, split, asset, task_idx))

    if selected_assets:
        unknown_assets = set(selected_assets).difference(found_assets)
        if unknown_assets:
            raise ValueError(
                "Selected assets are absent from the chosen categories/splits: "
                + ", ".join(sorted(unknown_assets))
            )
    return jobs


def summarize_jobs(jobs: Sequence[EvalJob]) -> dict[str, dict[str, int] | int]:
    by_category = {category: 0 for category in CATEGORY_ORDER}
    by_split = {split: 0 for split in SPLIT_ORDER}
    for job in jobs:
        by_category[job.category] += 1
        by_split[job.split] += 1
    return {
        "total": len(jobs),
        "by_category": {key: value for key, value in by_category.items() if value},
        "by_split": {key: value for key, value in by_split.items() if value},
    }
