# InsightBench Asset Download

InsightBench assets are distributed separately from the code repository.

Primary Hugging Face dataset:

https://huggingface.co/datasets/paragon7060/InsightBench-Assets-v0.1

Google Drive mirror:

https://drive.google.com/drive/folders/1ut90XWm8BaQhdlbxyp03wzSVSYNtpVmc

## Recommended Download

Download the compressed archive from Hugging Face:

```bash
huggingface-cli download paragon7060/InsightBench-Assets-v0.1 \
  --repo-type dataset \
  --include "InsightBench-Assets-v0.1.tar.gz" "InsightBench-Assets-v0.1.tar.gz.sha256" \
  --local-dir .
```

## Google Drive Mirror

The Google Drive mirror is split into parts. Download every file below only if
you are using the mirror instead of Hugging Face:

- `InsightBench-Assets-v0.1.tar.gz.part-000`
- `InsightBench-Assets-v0.1.tar.gz.part-001`
- `InsightBench-Assets-v0.1.tar.gz.part-002`
- `InsightBench-Assets-v0.1.tar.gz.part-003`
- `InsightBench-Assets-v0.1.tar.gz.part-004`
- `InsightBench-Assets-v0.1.tar.gz.part-005`
- `InsightBench-Assets-v0.1.tar.gz.part-006`
- `InsightBench-Assets-v0.1.tar.gz.part-007`
- `InsightBench-Assets-v0.1.tar.gz.part-008`
- `InsightBench-Assets-v0.1.tar.gz.part-009`
- `InsightBench-Assets-v0.1.tar.gz.part-010`
- `InsightBench-Assets-v0.1.tar.gz.part-011`
- `InsightBench-Assets-v0.1.tar.gz.part-012`
- `InsightBench-Assets-v0.1.tar.gz.part-013`
- `InsightBench-Assets-v0.1.tar.gz.sha256`
- `InsightBench-Assets-v0.1.parts.sha256`
- `README.txt`

Reconstruct the archive from the Google Drive parts:

From the directory containing the downloaded parts:

```bash
cat InsightBench-Assets-v0.1.tar.gz.part-* > InsightBench-Assets-v0.1.tar.gz
sha256sum -c InsightBench-Assets-v0.1.tar.gz.sha256
```

Expected archive SHA256:

```text
718dbfec6a8402148087885090f48f8d9ce6fd412e18eaa88972af8c1c4af9ed
```

Optional per-part verification:

```bash
sha256sum -c InsightBench-Assets-v0.1.parts.sha256
```

## Install

Extract the archive from the InsightBench repository root:

```bash
mkdir -p Assets
tar -xzf InsightBench-Assets-v0.1.tar.gz -C Assets
```

Expected local layout:

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

## Counts

- `TestSuite/cabinet_suite`: 6 assets
- `TestSuite/door_suite`: 9 assets
- `TestSuite/bottle_suite`: 10 assets
- `TrainSuite/cabinet_suite`: 249 assets
- `TrainSuite/door_suite`: 47 assets
- `TrainSuite/bottle_suite`: 32 assets

The public bundle excludes these known problematic evaluation assets:

- `TestSuite/cabinet_suite/45194`
- `TestSuite/door_suite/99655089960011l`

## Notes

- Evaluation requires `TestSuite`, `guides`, `FrankaEmika`, and
  `franka_description`.
- Data collection requires `TrainSuite` as well.
- Asset metadata and attribution are included inside the archive as
  `MANIFEST.json` and `ASSET_LICENSES.md`.
