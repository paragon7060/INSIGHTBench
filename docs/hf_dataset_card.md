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
718dbfec6a8402148087885090f48f8d9ce6fd412e18eaa88972af8c1c4af9ed
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

## License And Attribution

InsightBench code is licensed separately in the code repository. Simulation
assets are subject to the licenses and redistribution terms of their respective
sources, including PartManip, AdaManip, Franka assets, and generated guide
assets. See `ASSET_LICENSES.md` inside the archive for details.
