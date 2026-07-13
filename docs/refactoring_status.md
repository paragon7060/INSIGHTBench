# InsightBench Refactoring Status

Last updated: 2026-07-13

This document summarizes the current multi-thread refactoring status, known
blockers, validation results, and remaining TODOs.

## Operating Model

- Headquarters thread: owns decisions, priorities, task decomposition, and
  final status synthesis.
- Code Thread: changes only the assigned TODO.
- Code Review Thread: reviews Code Thread output.
- Run/Eval Thread: runs validation only. It must not modify code or assets.
- Asset Packaging Thread: stages and validates the asset bundle.

Thread outputs should be pasted back into Headquarters before the next
decision.

## Completed Work

### Refactor Foundation

- `REFACTORING_PLAN.md` is the source-of-truth plan.
- Eval runtime and its door/cabinet/bottle config import path no longer depend
  on CuRobo. `CuroboInteractionAction` loads planner dependencies only when a
  collection action is constructed; `tests/test_eval_without_curobo.py` guards
  this eval-only boundary.
- Pi0, Diffusion, SmolVLA, and InstructionGPT now import observation/action
  constants from LeRobot 0.4.1's `lerobot.utils.constants` module.
- Policy stats trimming was fixed for Pi0/Diffusion/SmolVLA/InstructionGPT.
- Public eval configs were cleaned up around checkpoint/stat overrides.
- Public eval configs now leave `policy.checkpoint` empty by default, and
  `scripts/evaluate.py` fails before Isaac launch when the override is missing.
- Public installers infer repo-adjacent `../IsaacLab`, `../lerobot`, and
  full-install-only `../curobo` instead of maintainer workstation paths.
- Before the first IsaacLab editable install, both installers bootstrap
  `numpy==1.26.4`, `pillow==11.2.1`, `setuptools<70`, and `toml`; this prevents
  editable metadata resolution from starting with NumPy 2. Immediately before
  LeRobot extras, they upgrade to `setuptools==80.9.0`, satisfying LeRobot
  0.4.1's `>=71,<81` requirement.
- Lightweight tests were added and previously passed.
- Cabinet collect metadata parsing was fixed for full TrainSuite directory
  names and `1ext` scene keys.

### Eval Smoke

Earlier Pi0 smokes for representative door, cabinet, and bottle tasks completed
environment build, policy load, observation/action steps, result JSON save, and
video save. Those historical results are retained below as pipeline evidence;
they are not the final unified-environment task score.

Generated results included:

- `outputs/results/pi0/door_99660039960014l_task0.json`
- `outputs/results/pi0/cabinet_46130_task0.json`
- `outputs/results/pi0/bottle_14b_task0.json`
- `outputs/results/pi0/summary.csv`

Current scores were `0/1` for each smoke run. That is not treated as a smoke
failure; the smoke objective is runtime pipeline validation.

### Unified Eval Runtime

Fresh unified-environment policy status is tracked separately from task success:

| Policy | Pipeline health | Task result | Classification |
|--------|-----------------|-------------|----------------|
| Pi0 | PASS: app, environment, cameras, legacy-config migration, state-dict load, first action, rollout, and result writing | Door `1/1` | Runtime and task-success PASS; the `918 vs 867` cache mismatch is resolved. |
| SmolVLA | PASS: full evaluation pipeline | Door `1/1` | Runtime and task-success PASS. |
| GR00T | PASS: app, environment, cameras, loading, inference, rollout, and result writing | Door `0/1` | Pipeline PASS; task did not succeed in this one-episode smoke. |
| Diffusion | Not run | N/A | Deferred because no checkpoint was supplied; not classified as a failure. |

Latest LeRobot dataset-writer migration remains outside the Phase 3 scope.

### Asset Staging

Physical asset copy completed under:

- `<asset-staging-root>/`

Validation summary from Asset Packaging Thread:

```text
manifest_target_validation: PASS
entry_count: 353
TestSuite/bottle: 10
TestSuite/cabinet: 6
TestSuite/door: 9
TrainSuite/bottle: 32
TrainSuite/cabinet: 249
TrainSuite/door: 47
missing_targets: 0
missing_required_files: 0
excluded_present: 0
missing_shared_dirs: 0
symlink_count: 0
staging_size: 3.0G
```

Hugging Face archive publication is complete:

- Primary dataset repo:
  https://huggingface.co/datasets/paragon7060/InsightBench-Assets-v0.1
- Uploaded files:
  - `InsightBench-Assets-v0.1.tar.gz`
  - `InsightBench-Assets-v0.1.tar.gz.sha256`
  - `MANIFEST.json`
  - `ASSET_LICENSES.md`
  - `README.md`
- Remote repo file count: 6
- Remote archive size: 1,367,195,801 bytes

Google Drive archive mirror is complete:

- Mirrored archive: `InsightBench-Assets-v0.1.tar.gz`
- Archive SHA256:
  `718dbfec6a8402148087885090f48f8d9ce6fd412e18eaa88972af8c1c4af9ed`
- Drive release folder:
  https://drive.google.com/drive/folders/1ut90XWm8BaQhdlbxyp03wzSVSYNtpVmc
- Drive upload format: 14 split parts plus archive checksum, per-part
  checksum, and `README.txt`.

GitHub-facing asset download instructions are documented in
`README.md` and `docs/assets.md`; Hugging Face is the primary download path and
Google Drive is the mirror.

### Collect Demo Controls

`scripts/collect_demo.py` now supports:

- `--pos_rand`
- `--no_pos_rand` / `--fixed_pos`
- `--debug_collect`
- `--progress_interval`
- `--skill_timeout_s`
- `--no_frame_write` / `--dry_run_frames`
- `--collect_decimation`
- `--collect_render_interval`
- `--smoke_action_steps`
- `--smoke_decimation`
- `--smoke_step_timeout_s`
- `--smoke_save_episode`

Current default collect render behavior is:

```text
decimation=300
render_interval=300
```

This avoids rendering RTX camera frames on every physics substep.

`interact/motion_generator.py` now exposes `last_plan_debug` for collect debug
logs without changing the `command()` return interface.

## Current Blockers

### B1. Collect Smoke Needs Run/Eval Validation Of The Bounded Path

Before the render fix, door collect smoke reached:

```text
env build: PASS
CuRobo planning: PASS
max_T=86 to 125
action execution: timeout
```

Debug logs showed `env.step()` taking about 13 seconds per trajectory waypoint,
while frame writing was about 0.05 seconds. The relevant execution shape was:

```text
one collector action -> 300 synchronous physics substeps -> RTX camera render
```

`--collect_decimation` was already propagated to `env_cfg.decimation`, so the
collector itself can select a lower cadence. P3 now adds an explicit smoke-only
path: `--smoke_action_steps N` forces `decimation=10`, `render_interval=10`, and
stops after `N` planned trajectory actions. It also records `EnvStepBegin` and
`EnvStepEnd` around every settle/action call. A smoke deadline is checked at
physics-loop, render, and observation boundaries; normal collection has no
deadline and keeps `decimation=300`.

`--no_frame_write` continues to skip LeRobot image/frame writes and episode save.
For a separate writer check, omit that flag and use `--smoke_save_episode`; this
saves a partial smoke episode without applying the production success threshold.
Run/Eval has not yet validated the new bounded path on GPU.

### B2. GR00T Requires The Unified Policy Environment

The original `env_insightbench` used:

```text
<legacy-isaaclab-root>/lerobot
LeRobot 0.3.4 editable install
```

That checkout does not include:

```text
lerobot.policies.groot
```

A GROOT-capable checkout was validated locally at:

```text
<public-workspace>/lerobot
```

Public provenance is confirmed for the same source baseline:

```text
URL:     https://github.com/huggingface/lerobot.git
branch:  main
commit:  f6b16f6d97155e3ce34ab2a1ec145e9413588197
version: 0.4.1
license: Apache-2.0
policies present: pi0, diffusion, smolvla, groot
extras default: pi, smolvla, groot
```

The exact commit was fetched successfully from the public GitHub repository.

The `groot_lab` environment uses that checkout and has now passed the relevant
import and smoke checks:

```text
Python 3.11.14
LeRobot 0.4.1 from <public-workspace>/lerobot
transformers 4.57.1
torch 2.7.0+cu128
torchvision 0.22.0+cu128
numpy 1.26.4
Pillow 11.2.1
opencv-python-headless 4.11.0.86
Pi0 import OK
Diffusion import OK
SmolVLA import OK
GR00T import OK
IsaacLab import OK
GR00T eval smoke exit 0
```

The project direction is to make this the default pattern: one unified
evaluation environment for all supported policies, installed with a
GROOT-capable LeRobot checkout pinned to the public baseline above. The install
scripts fail fast when `lerobot.policies.groot` is missing because older
IsaacLab-local LeRobot checkouts cannot run GR00T evaluation.

The installers explicitly pin `transformers==4.57.1` after LeRobot extras.
LeRobot 0.4.1 allows `transformers>=4.53,<5`, and the fresh resolver selected
4.53.3; that version reaches GR00T model loading but fails in Eagle processor
inference. The 4.57.1 pin matches the validated runtime. Before Pi0 model
construction, InsightBench installs a version-checked SigLIP shim matching the
two OpenPI semantics Pi0 requires: a `siglip.check` installation check that is
true only when the forward patch is present, and a bf16 cast of vision embeddings
before a bf16 SigLIP encoder. This retains one 4.57.1 runtime for Pi0 and GR00T.

Pi0's Transformer bridge now imports `PaliGemmaWithExpertModel` from the
baseline's `lerobot.policies.pi0.modeling_pi0`; the older fork-only
`paligemma_with_expert` module is not present in the documented checkout.
Legacy Pi0 checkpoint configs from LeRobot 0.3.4 are normalized in memory
before 0.4.1 decoding. The supported checkpoint contract is exact: 224x224
padding maps to `image_resolution`, `num_steps` maps to
`num_inference_steps`, and `proj_width=1024` maps to the 0.4.1
`gemma_300m` action expert. Disabled Aloha transforms, eager attention, and
enabled caching are accepted; alternate values and unclassified fields fail
fast. Training-only freeze/train flags are ignored for evaluation only. The
source `config.json` is never modified. Pi0 and SmolVLA now pass stats through
their official 0.4.1 pre/post-processor factories rather than model
constructors. Focused tests cover the processor contracts. The unified env7
Pi0 and SmolVLA door smokes have completed full action rollouts with `1/1`
task success; GR00T has completed its full pipeline smoke with `0/1` task
success.
Transformers 4.57's mutable `DynamicCache` also appends Pi0's 51-token
state/action suffix even when the model receives `use_cache=False`. The legacy
Pi0 path instead used `fill_kv_cache=False`: its 816-token image/language prefix
must remain unchanged across every denoising step. The Pi0 adapter now restores
that prefix after each suffix forward and fails fast if the cache/prefix contract
is already invalid. A CPU regression test constructs the migrated 816+51
attention/position contract with Gemma's real cache API. The env7 Pi0 door
smoke subsequently passed first action, full rollout, result writing, and `1/1`
task success, closing this compatibility blocker.
SmolVLA's public config no longer defaults to an unverified dataset stats repo.
It fails before Isaac launch unless the user supplies either a verified
`policy.dataset_stats_repo` or `policy.dataset_stats_root`.

OpenCV fresh-install blocker is resolved in the release scripts: install
metadata now follows LeRobot 0.4.1's declared `opencv-python-headless`
dependency and no longer uninstalls the headless package.

Dependency hygiene status: the following are non-blocking upstream metadata
residuals for the validated evaluation runtime.

| `pip check` item | Owner | Status |
|------------------|-------|--------|
| `rerun-sdk 0.26.2` requires `numpy>=2` | LeRobot 0.4.1 visualization/telemetry metadata | Runtime-safe release exception. All LeRobot-compatible `rerun-sdk>=0.24,<0.27` wheels inspected declare `numpy>=2`; policy imports for Pi0, Diffusion, SmolVLA, and GR00T do not import `rerun`. |
| `nvidia-srl-*` missing `usd-core` or requiring `lxml<5` | Isaac Sim URDF exporter prebundle | Optional exporter tooling, not used by InsightBench eval runtime. |
| `plotly`, `selenium`, `azure-identity`, `msal-extensions` missing optional deps | Isaac Sim prebundled core/cloud tooling | Optional tooling, not used by InsightBench eval runtime. |

Fresh install smoke should verify that no additional InsightBench-owned
dependency conflicts appear beyond the release exceptions above.

### B3. Latest LeRobot 0.6.x Cannot Share The Current IsaacLab Env

Latest LeRobot migration is not part of Phase 3. It remains a Phase 4 research
track and is not required for the unified Pi0/Diffusion/SmolVLA/GR00T release
baseline.

Known package constraints:

```text
IsaacLab latest: Python ==3.11.*, numpy <2
LeRobot 0.6.x: Python >=3.12, numpy >=2
```

Therefore a single environment containing current IsaacLab and upstream
LeRobot 0.6.x is not the Phase 3 target. The likely future architecture is:

- IsaacLab env: simulation/runtime, Python 3.11, NumPy 1.x
- latest LeRobot env: policy inference, Python 3.12, NumPy 2.x
- bridge: RPC/IPC between simulation and policy process

## Usage Notes

### Pi0 Eval Smoke

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n env_insightbench \
python scripts/evaluate.py \
  --config configs/eval/pi0.yaml \
  --object door \
  --asset_path 99660039960014l \
  --task_idx 0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  policy.checkpoint=LOCAL_PI0_PRETRAINED_MODEL_DIR
```

### Bounded Collect Smoke: Dry Run

Use a staging or extracted TrainSuite path.

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n env_insightbench \
python scripts/collect_demo.py \
  --object door \
  --asset_id 99660039960014l \
  --scene_key 3a \
  --dataset_name local/insightbench-smoke-door-debug \
  --asset_dir Assets/TrainSuite/door_suite \
  --num_envs 1 \
  --target_episodes 1 \
  --max_loops 1 \
  --no_pos_rand \
  --debug_collect \
  --progress_interval 1 \
  --smoke_action_steps 2 \
  --no_frame_write \
  --headless \
  --enable_cameras
```

Expected completion markers are `EnvStepBegin`, `EnvStepEnd`, and
`[CollectSmoke] completed action_steps=2; no_frame_write=True`.

### Bounded Collect Smoke: Frame Write And Save

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n env_insightbench \
python scripts/collect_demo.py \
  --object door \
  --asset_id 99660039960014l \
  --scene_key 3a \
  --dataset_name local/insightbench-smoke-door-frames \
  --asset_dir Assets/TrainSuite/door_suite \
  --num_envs 1 \
  --target_episodes 1 \
  --max_loops 1 \
  --no_pos_rand \
  --debug_collect \
  --progress_interval 1 \
  --smoke_action_steps 2 \
  --smoke_save_episode \
  --headless \
  --enable_cameras
```

Expected completion marker is `[CollectSmoke] completed action_steps=2; saved 1
partial frame-write episode(s)`.

### SmolVLA Door Smoke

```bash
CUDA_VISIBLE_DEVICES=3 timeout 20m conda run --no-capture-output -n env_insightbench \
python scripts/evaluate.py \
  --config configs/eval/smolvla.yaml \
  --object door \
  --asset_path 99660039960014l \
  --task_idx 0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  policy.checkpoint=LOCAL_SMOLVLA_PRETRAINED_MODEL_DIR \
  policy.dataset_stats_repo=YOUR_HF_DATASET_REPO
```

### GR00T Smoke

```bash
CUDA_VISIBLE_DEVICES=3 timeout 20m conda run --no-capture-output -n groot_lab \
python scripts/evaluate.py \
  --config configs/eval/groot.yaml \
  --object door \
  --asset_path 99660039960014l \
  --task_idx 0 \
  --num_envs 1 \
  --headless \
  --enable_cameras \
  policy.checkpoint=LOCAL_GROOT_PRETRAINED_MODEL_DIR
```

Result:

```text
[1/3] Env ready
[2/3] Policy loaded  (groot)
ep0: 0/1 success
RESULT: door/99660039960014l/task0
Score: 0/1 (0.0%)
```

This is a runtime pass for the evaluation pipeline. The one-episode smoke did
not succeed at the task, but env creation, camera observations, GR00T loading,
inference, rollout, and result writing completed.

## Remaining TODOs

### Immediate TODOs

1. Run/Eval: run the bounded no-frame-write door collect smoke, then the separate
   frame-write/save smoke.
2. Run/Eval: provide a verified Diffusion checkpoint before its evaluation;
   Diffusion is deferred, not failed.
3. Docs: keep README/install scripts aligned with the unified policy env.

### Asset Release TODOs

1. Run/Eval: run door/cabinet/bottle collect smoke using
   `<asset-staging-root>`.
2. Asset Packaging Thread: create archive and checksum only after collect smoke
   passes or the blocker is explicitly waived.
3. Docs: update asset download instructions once final hosting location is
   chosen.

### Phase 4 Latest LeRobot TODOs

1. Decide whether latest LeRobot support is implemented through a separate
   policy server/process.
2. Spike a Python 3.12 + LeRobot 0.6.x policy env.
3. Define observation/action IPC schema between IsaacLab and policy process.
4. Port Pi0 inference first.
5. Port SmolVLA and Diffusion next.
6. Add GR00T support using a latest compatible checkpoint and processor files.
7. Migrate dataset collection to LeRobot v3 writer API only if the target env
   can support it.

### GitHub Release TODOs

1. Restore or initialize a valid local git repository.
2. Add a GitHub remote.
3. Commit current refactor state and docs.
4. Push to a branch.
5. Open a draft PR with this status summary and validation results.

Current blocker: `<insightbench-root>` is not recognized as a
git repository by `git status`.
