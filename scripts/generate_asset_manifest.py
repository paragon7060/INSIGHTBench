#!/usr/bin/env python3
"""Generate the public InsightBench asset-bundle manifest.

This script is intentionally read-only: it inspects the current source asset
folders and writes a JSON manifest only when --output is provided. It does not
copy, move, delete, or rewrite assets.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shlex
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

PUBLIC_TEST_EXCLUSIONS: dict[str, dict[str, str]] = {
    "cabinet": {
        "45194": "confirmed problematic; excluded from public benchmark config",
    },
    "door": {
        "99655089960011l": "confirmed problematic; excluded from public benchmark config",
    },
    "bottle": {},
}

REQUIRED_FILES: dict[str, list[str]] = {
    "cabinet": [
        "mobility_new.usd",
        "link_handle_relationship.json",
    ],
    "door": [
        "mobility_push_cw.usd",
        "mobility_push_cw.urdf",
        "mobility_push_ccw.usd",
        "mobility_push_ccw.urdf",
        "mobility_pull_cw.usd",
        "mobility_pull_cw.urdf",
        "mobility_pull_ccw.usd",
        "mobility_pull_ccw.urdf",
    ],
    "bottle": [
        "mobility.usd",
        "mobility_reversed.usd",
        "bounding_box.json",
    ],
}

BUNDLE_SPECS: list[dict[str, Any]] = [
    {
        "suite": "TestSuite",
        "object": "cabinet",
        "source_rel": "TestSuite/cabinet_suite",
        "target_rel": "TestSuite/cabinet_suite",
        "source_dataset": "PartManip",
        "license_key": "partmanip_cc_by_nc_4_0",
        "exclude": PUBLIC_TEST_EXCLUSIONS["cabinet"],
    },
    {
        "suite": "TestSuite",
        "object": "door",
        "source_rel": "TestSuite/door_suite",
        "target_rel": "TestSuite/door_suite",
        "source_dataset": "AdaManip",
        "license_key": "adamanip_verify_upstream",
        "exclude": PUBLIC_TEST_EXCLUSIONS["door"],
    },
    {
        "suite": "TestSuite",
        "object": "bottle",
        "source_rel": "TestSuite/bottle_suite",
        "target_rel": "TestSuite/bottle_suite",
        "source_dataset": "AdaManip",
        "license_key": "adamanip_verify_upstream",
        "exclude": PUBLIC_TEST_EXCLUSIONS["bottle"],
    },
    {
        "suite": "TrainSuite",
        "object": "cabinet",
        "source_rel": "PartManip/drawer/train",
        "target_rel": "TrainSuite/cabinet_suite",
        "source_dataset": "PartManip",
        "license_key": "partmanip_cc_by_nc_4_0",
        "exclude": {},
    },
    {
        "suite": "TrainSuite",
        "object": "door",
        "source_rel": "AdaManip/door",
        "target_rel": "TrainSuite/door_suite",
        "source_dataset": "AdaManip",
        "license_key": "adamanip_verify_upstream",
        "exclude": {},
    },
    {
        "suite": "TrainSuite",
        "object": "bottle",
        "source_rel": "AdaManip/bottle",
        "target_rel": "TrainSuite/bottle_suite",
        "source_dataset": "AdaManip",
        "license_key": "adamanip_verify_upstream",
        "exclude": {},
    },
]

SHARED_SPECS: list[dict[str, str]] = [
    {
        "name": "guides",
        "source_rel": "guides",
        "target_rel": "guides",
        "license_key": "insightbench_generated_guides",
    },
    {
        "name": "FrankaEmika",
        "source_rel": "FrankaEmika",
        "target_rel": "FrankaEmika",
        "license_key": "franka_assets_verify_upstream",
    },
    {
        "name": "franka_description",
        "source_rel": "franka_description",
        "target_rel": "franka_description",
        "license_key": "franka_description_verify_upstream",
    },
]


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _posix(path: Path) -> str:
    return path.as_posix()


def _asset_dirs(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _dir_stats(root: Path) -> dict[str, int]:
    file_count = 0
    size_bytes = 0
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                file_count += 1
                size_bytes += path.stat().st_size
    return {"file_count": file_count, "size_bytes": size_bytes}


def _missing_required(asset_dir: Path, object_name: str) -> list[str]:
    return [
        relpath
        for relpath in REQUIRED_FILES[object_name]
        if not (asset_dir / relpath).is_file()
    ]


def _entry_for_asset(assets_root: Path, spec: dict[str, Any], asset_dir: Path) -> dict[str, Any]:
    asset_id = asset_dir.name
    stats = _dir_stats(asset_dir)
    missing = _missing_required(asset_dir, spec["object"])
    return {
        "suite": spec["suite"],
        "object": spec["object"],
        "asset_id": asset_id,
        "source_dataset": spec["source_dataset"],
        "license_key": spec["license_key"],
        "source_rel": _posix(asset_dir.relative_to(assets_root)),
        "target_rel": _posix(Path(spec["target_rel"]) / asset_id),
        "status": "include",
        "required_files": REQUIRED_FILES[spec["object"]],
        "missing_required_files": missing,
        "file_count": stats["file_count"],
        "size_bytes": stats["size_bytes"],
        "notes": [],
    }


def build_manifest(assets_root: Path) -> dict[str, Any]:
    assets_root = assets_root.resolve()
    entries: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    missing_sources: list[str] = []

    for spec in BUNDLE_SPECS:
        source_dir = assets_root / spec["source_rel"]
        if not source_dir.is_dir():
            missing_sources.append(spec["source_rel"])
            continue

        excluded = spec["exclude"]
        for asset_dir in _asset_dirs(source_dir):
            asset_id = asset_dir.name
            if asset_id in excluded:
                exclusions.append(
                    {
                        "suite": spec["suite"],
                        "object": spec["object"],
                        "asset_id": asset_id,
                        "source_rel": _posix(asset_dir.relative_to(assets_root)),
                        "target_rel": _posix(Path(spec["target_rel"]) / asset_id),
                        "reason": excluded[asset_id],
                    }
                )
                continue
            entries.append(_entry_for_asset(assets_root, spec, asset_dir))

    shared_assets: list[dict[str, Any]] = []
    for spec in SHARED_SPECS:
        source_dir = assets_root / spec["source_rel"]
        stats = _dir_stats(source_dir)
        shared_assets.append(
            {
                "name": spec["name"],
                "source_rel": spec["source_rel"],
                "target_rel": spec["target_rel"],
                "license_key": spec["license_key"],
                "status": "include" if source_dir.is_dir() else "missing_source",
                "file_count": stats["file_count"],
                "size_bytes": stats["size_bytes"],
            }
        )
        if not source_dir.is_dir():
            missing_sources.append(spec["source_rel"])

    summary = _build_summary(entries, shared_assets, exclusions, missing_sources)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "assets_root": _posix(assets_root),
        "bundle_layout": {
            "hf_repo_root": [
                "TestSuite/",
                "TrainSuite/",
                "guides/",
                "FrankaEmika/",
                "franka_description/",
                "MANIFEST.json",
                "ASSET_LICENSES.md",
                "README.md",
            ],
            "local_download_note": "Download repo contents into ./Assets so local paths become Assets/TestSuite, Assets/TrainSuite, etc.",
        },
        "decisions": {
            "test_excluded_assets_removed_from_public_bundle": True,
            "cabinet_train_asset_ids_preserve_full_directory_names": True,
            "distribution": "physical copy archive for Google Drive; no symlinks in public bundle",
            "original_assets_are_not_modified": True,
        },
        "problematic_assets": {
            "known_test_exclusions": PUBLIC_TEST_EXCLUSIONS,
            "known_train_exclusions": {},
            "notes": [
                "No TrainSuite problematic assets were identified by inventory alone.",
                "Cabinet collection still requires smoke validation because current collect scene-key handling may need a code fix.",
            ],
        },
        "summary": summary,
        "exclusions": exclusions,
        "entries": entries,
        "shared_assets": shared_assets,
    }


def _build_summary(
    entries: list[dict[str, Any]],
    shared_assets: list[dict[str, Any]],
    exclusions: list[dict[str, Any]],
    missing_sources: list[str],
) -> dict[str, Any]:
    by_suite_object: dict[str, int] = {}
    missing_required_count = 0
    size_bytes = 0
    file_count = 0

    for entry in entries:
        key = f"{entry['suite']}/{entry['object']}"
        by_suite_object[key] = by_suite_object.get(key, 0) + 1
        missing_required_count += len(entry["missing_required_files"])
        size_bytes += entry["size_bytes"]
        file_count += entry["file_count"]

    shared_size = sum(item["size_bytes"] for item in shared_assets)
    shared_files = sum(item["file_count"] for item in shared_assets)
    return {
        "asset_count": len(entries),
        "asset_count_by_suite_object": dict(sorted(by_suite_object.items())),
        "excluded_asset_count": len(exclusions),
        "missing_required_file_count": missing_required_count,
        "missing_source_count": len(missing_sources),
        "missing_sources": sorted(set(missing_sources)),
        "asset_file_count": file_count,
        "asset_size_bytes": size_bytes,
        "shared_file_count": shared_files,
        "shared_size_bytes": shared_size,
        "total_file_count": file_count + shared_files,
        "total_size_bytes": size_bytes + shared_size,
    }


def print_summary(manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    print("InsightBench asset manifest summary")
    print(f"assets_root: {manifest['assets_root']}")
    print(f"asset_count: {summary['asset_count']}")
    for key, count in summary["asset_count_by_suite_object"].items():
        print(f"  {key}: {count}")
    print(f"excluded_asset_count: {summary['excluded_asset_count']}")
    print(f"missing_required_file_count: {summary['missing_required_file_count']}")
    print(f"missing_source_count: {summary['missing_source_count']}")
    if summary["missing_sources"]:
        for source in summary["missing_sources"]:
            print(f"  missing source: {source}")
    print(f"total_size_bytes: {summary['total_size_bytes']}")


def print_rsync_plan(assets_root: Path, staging_root: Path) -> None:
    def quote(path: Path | str) -> str:
        return shlex.quote(str(path))

    print("# Review with --dry-run first. These commands do not run automatically.")
    print(f"mkdir -p {quote(staging_root)}")

    for spec in BUNDLE_SPECS:
        source = assets_root / spec["source_rel"]
        target = staging_root / spec["target_rel"]
        print(f"mkdir -p {quote(target)}")
        command = [
            "rsync",
            "-a",
            "--dry-run",
        ]
        for asset_id in sorted(spec["exclude"]):
            command.extend(["--exclude", quote(asset_id + "/")])
        command.extend(
            [
                quote(source) + "/",
                quote(target) + "/",
            ]
        )
        print(" ".join(command))

    for spec in SHARED_SPECS:
        source = assets_root / spec["source_rel"]
        target = staging_root / spec["target_rel"]
        print(f"mkdir -p {quote(target)}")
        print(f"rsync -a --dry-run {quote(source)}/ {quote(target)}/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, default=Path("Assets"))
    parser.add_argument("--output", type=Path, default=None, help="Write manifest JSON to this path.")
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=Path("InsightBench-Assets-Staging"),
        help="Root used when printing an rsync staging plan.",
    )
    parser.add_argument(
        "--print-rsync-plan",
        action="store_true",
        help="Print dry-run rsync commands for the decided public bundle layout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when source folders or required files are missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_manifest(args.assets_root)
    print_summary(manifest)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"wrote manifest: {args.output}")

    if args.print_rsync_plan:
        print()
        print_rsync_plan(args.assets_root, args.staging_root)

    summary = manifest["summary"]
    if args.strict and (
        summary["missing_source_count"] > 0 or summary["missing_required_file_count"] > 0
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
