---
license: other
pretty_name: InsightBench Assets v0.1
tags:
  - robotics
  - simulation
  - isaac-sim
  - manipulation
  - benchmark
---

# InsightBench Assets v0.1

This dataset repository hosts the public asset archive for InsightBench, a
robot manipulation benchmark with visual guide-conditioned tasks.

Primary archive:

- `InsightBench-Assets-v0.1.tar.gz`
- `InsightBench-Assets-v0.1.tar.gz.sha256`

Google Drive mirror:

https://drive.google.com/drive/folders/1ut90XWm8BaQhdlbxyp03wzSVSYNtpVmc

## Download

```bash
huggingface-cli download paragon7060/InsightBench-Assets-v0.1 \
  --repo-type dataset \
  --include "InsightBench-Assets-v0.1.tar.gz" "InsightBench-Assets-v0.1.tar.gz.sha256" \
  --local-dir .
```

Verify and extract from the InsightBench repository root:

```bash
sha256sum -c InsightBench-Assets-v0.1.tar.gz.sha256
mkdir -p Assets
tar -xzf InsightBench-Assets-v0.1.tar.gz -C Assets
```

Expected archive SHA256:

```text
7bbd3ef59131a0322e8041be776e9c1356426b720946d77212c8f9ff396823e1
```

## Extracted Layout

The archive is extracted into the existing `Assets/` directory; the runtime
consumer root is therefore `Assets/`.

```text
Assets/
  TestSuite/
    cabinet_suite/
    door_suite/
    bottle_suite/
  TrainSuite/
    cabinet_suite/
    door_suite/
    bottle_suite/
  guides/
  FrankaEmika/
  franka_description/
  MANIFEST.json
  ASSET_LICENSES.md
  README.md
```

## Asset Counts

- `TestSuite/cabinet_suite`: 6 assets
- `TestSuite/door_suite`: 9 assets
- `TestSuite/bottle_suite`: 10 assets
- `TrainSuite/cabinet_suite`: 249 assets
- `TrainSuite/door_suite`: 47 assets
- `TrainSuite/bottle_suite`: 32 assets

The public bundle excludes these known problematic evaluation assets:

- `TestSuite/cabinet_suite/45194`
- `TestSuite/door_suite/99655089960011l`

## Metadata

- `MANIFEST.json` records source-to-target mapping, required-file checks,
  counts, exclusions, and bundle decisions.
- `ASSET_LICENSES.md` records source dataset attribution and license notes.

## Asset Patch (2026-07-18)

- Normalized `TestSuite/cabinet_suite/34178/mobility_new.usd` so the fully
  closed pose is joint zero (`joint_2` range approximately `0` to
  `1.64619 rad`).
- Added `guides/arrows/guide_arrow_visual.usd` for guide arrows parented under
  moving cabinet links without nested rigid-body schemas.

## License And Attribution

InsightBench code is licensed separately in the code repository. Simulation
assets are subject to the licenses and redistribution terms of their respective
sources, including PartManip, AdaManip, Franka assets, and generated guide
assets. See `ASSET_LICENSES.md` inside the archive for details.
