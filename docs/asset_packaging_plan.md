# InsightBench Asset Packaging Plan

This document defines the public asset bundle layout for InsightBench. It is a
staging plan only: original local asset folders must not be moved, deleted, or
rewritten while preparing the public bundle.

## Decisions

- Public `TestSuite` excludes known problematic evaluation assets.
- Excluded evaluation assets are removed only from the staging/public bundle,
  not from the original local `Assets/` tree.
- Train cabinet assets keep their full directory names, such as
  `StorageFurniture-46130-link_0-handle_0-joint_0-handlejoint_0`.
- Public distribution uses physical copies, not symlinks.
- The public archive will be distributed separately through Google Drive.
- `MANIFEST.json` is required.
- `ASSET_LICENSES.md` is required.
- Door, bottle, and cabinet collection each need one smoke test before release.
- Cabinet collection remains in scope and must be fixed if the current
  `scene_key=1ext` path is incompatible with collection metadata.

## Known Problematic Assets

Only evaluation-suite problematic assets are currently identified:

- `TestSuite/cabinet_suite/45194`
- `TestSuite/door_suite/99655089960011l`

No TrainSuite problematic assets were identified by inventory alone. TrainSuite
problems should be discovered by collection smoke tests and then recorded in the
manifest or a future exclude list.

## Source To Target Mapping

The public Google Drive archive should contain the bundle root directly. Users
will unpack or download it into local `Assets/`, so paths become
`Assets/TestSuite`, `Assets/TrainSuite`, and so on.

| Source | Public target |
| --- | --- |
| `Assets/TestSuite/cabinet_suite` | `TestSuite/cabinet_suite` |
| `Assets/TestSuite/door_suite` | `TestSuite/door_suite` |
| `Assets/TestSuite/bottle_suite` | `TestSuite/bottle_suite` |
| `Assets/PartManip/drawer/train` | `TrainSuite/cabinet_suite` |
| `Assets/AdaManip/door` | `TrainSuite/door_suite` |
| `Assets/AdaManip/bottle` | `TrainSuite/bottle_suite` |
| `Assets/guides` | `guides` |
| `Assets/FrankaEmika` | `FrankaEmika` |
| `Assets/franka_description` | `franka_description` |

## Current Inventory

Observed source counts:

- Cabinet train source: 249 assets under `Assets/PartManip/drawer/train`
- Door train/source assets: 47 assets under `Assets/AdaManip/door`
- Bottle train/source assets: 32 assets under `Assets/AdaManip/bottle`
- Test cabinet source: 7 assets before public exclusion, 6 after exclusion
- Test door source: 10 assets before public exclusion, 9 after exclusion
- Test bottle source: 10 assets

Expected public counts:

- `TestSuite/cabinet_suite`: 6
- `TestSuite/door_suite`: 9
- `TestSuite/bottle_suite`: 10
- `TrainSuite/cabinet_suite`: 249
- `TrainSuite/door_suite`: 47
- `TrainSuite/bottle_suite`: 32

## Required Files

`scripts/collect_demo.py` expects the default TrainSuite roots:

- `./Assets/TrainSuite/cabinet_suite`
- `./Assets/TrainSuite/door_suite`
- `./Assets/TrainSuite/bottle_suite`

`cfg.helper.get_info_collect()` requires:

Cabinet:

- `mobility_new.usd`
- `link_handle_relationship.json`

Door:

- `mobility_push_cw.usd`
- `mobility_push_cw.urdf`
- `mobility_push_ccw.usd`
- `mobility_push_ccw.urdf`
- `mobility_pull_cw.usd`
- `mobility_pull_cw.urdf`
- `mobility_pull_ccw.usd`
- `mobility_pull_ccw.urdf`

Bottle:

- `mobility.usd`
- `mobility_reversed.usd`
- `bounding_box.json`

The public bundle should copy whole asset directories, not only required files,
because USD and URDF files may reference textures, materials, DAE files, and
additional metadata.

## Manifest Generation

Generate a read-only manifest from the current local assets:

```bash
python3 scripts/generate_asset_manifest.py \
  --assets-root Assets \
  --output MANIFEST.json \
  --strict
```

Print a dry-run staging plan:

```bash
python3 scripts/generate_asset_manifest.py \
  --assets-root Assets \
  --staging-root InsightBench-Assets-Staging \
  --print-rsync-plan
```

The generated manifest includes:

- included asset entries
- excluded public TestSuite assets and reasons
- source and target relative paths
- source dataset and license key
- required-file check results
- file counts and byte sizes
- shared asset folders

## Staging Workflow

1. Generate `MANIFEST.json` from the current local source assets.
2. Review manifest counts and missing-file results.
3. Generate the dry-run rsync plan.
4. Run dry-run rsync commands and inspect output.
5. Run the same rsync commands without `--dry-run` only after review.
6. Copy `MANIFEST.json`, `ASSET_LICENSES.md`, and a short asset README into the
   staging root.
7. Create a compressed archive for Google Drive distribution.
8. Download/unpack the archive in a clean location and verify that local paths
   resolve as `Assets/TestSuite`, `Assets/TrainSuite`, `Assets/guides`, etc.
9. Run door, bottle, and cabinet collection smoke tests.

Recommended staging archive name:

```text
InsightBench-Assets-v0.1.tar.gz
```

## Validation Before Release

Required checks:

- Manifest strict mode passes.
- `TestSuite` asset count matches benchmark config.
- `TrainSuite` asset count matches source inventory.
- Excluded TestSuite assets are not present in staging.
- `guides`, `FrankaEmika`, and `franka_description` are present.
- `scripts/collect_demo.py` can run one smoke collection for door, bottle, and
  cabinet using default TrainSuite layout.

Known code risk to resolve before declaring cabinet collection supported:

- `scripts/collect_batch.sh` uses cabinet `scene_key=1ext`.
- `cfg.helper.get_info_collect()` currently derives a cabinet handle index from
  `scene_key[-1]`, which is unsafe for `1ext`.

