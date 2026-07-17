# InsightBench

**InsightBench** is a simulation benchmark for evaluating robot manipulation policies in scenarios where a *visual guide* provides task-specific information at inference time — such as an arrow indicating which drawer to open, a rotation direction for a door, or a grip point for a bottle cap.

> Built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (Isaac Sim 5.0) and [LeRobot](https://github.com/huggingface/lerobot).

---

## Benchmark Overview

InsightBench tests three manipulation categories across **diverse real-world asset models**:

| Category | # Assets (test) | # Task variants | Guide type |
|----------|----------------|-----------------|------------|
| **Cabinet** (drawer) | 6 | 2 – 6 per asset | Arrow overlay on image |
| **Door** | 9 | 4 | Rotation + push/pull arrow |
| **Bottle** | 10 | 8 (open & close) | Squeeze grip indicator |

Assets are split equally into **seen** and **unseen** sets (3/3 cabinet, 5/4 door, 5/5 bottle).

Each task requires the policy to interpret a visual guide image and execute a precise manipulation sequence using the Franka Panda arm.

---

## Benchmark Results

Success rate (%) over 8 parallel environments × 1 episode per asset-task pair (139 runs total).

| Policy | Cabinet | Door | Bottle | **Overall** |
|--------|:-------:|:----:|:------:|:-----------:|
| Diffusion Policy | — | — | — | — |
| Pi0 (no guide) | — | — | — | — |
| Pi0 (guide) | — | — | — | — |
| SmolVLA | — | — | — | — |
| GR00T | —% | —% | — | — |
| Instruction-GPT + Pi0 | — | — | — | — |

*Evaluated on Isaac Sim 5.0. Results will be updated as evaluations complete.*

---

## Installation

### Prerequisites

- NVIDIA GPU with CUDA 12.4 (tested on A100 / RTX 4090)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) cloned alongside this repo (Isaac Sim 5.0 standalone binary)
- Conda (Miniconda / Anaconda)
- Python 3.11 runtime. Python 3.10 and latest LeRobot 0.6.x are not validated
  for this IsaacLab release path.

InsightBench uses a **single policy evaluation environment** for Pi0, Diffusion,
SmolVLA, GR00T, and Instruction-GPT + Pi0. The environment must install an
LeRobot checkout that contains all policy modules, including
`lerobot.policies.groot`. The tested source is the upstream Hugging Face
checkout below; an alternative compatible fork must provide the same policy
module set.

The reproducible LeRobot baseline for this release is:

```bash
git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout f6b16f6d97155e3ce34ab2a1ec145e9413588197
```

This checkout reports `lerobot` version `0.4.1` and is Apache-2.0 licensed.
Latest upstream LeRobot `0.6.x` support is intentionally out of scope for this
release and remains a Phase 4 research item.

InsightBench provides **two install scripts** depending on your use case:

| Script | Use case | CuRobo | Install time |
|--------|----------|--------|-------------|
| `install_eval.sh` | Run benchmark evaluations for all supported policies | ✗ Not needed | ~5–15 min |
| `install.sh` | Data collection + interact scenes | ✓ Required | ~30 min (CUDA compile) |

> **Why two environments?** [CuRobo](https://github.com/NVlabs/curobo) is only needed for motion-planning-based data collection (`scripts/collect_demo.py`) and interactive scene testing (`scene_interact/`). Evaluation scripts and their door/cabinet/bottle config imports run without it; the collection action loads CuRobo only when it is constructed.

---

### Option A — Evaluation Only (recommended for most users)

```bash
# 1. Use the sibling repository layout expected by the installer defaults.
#    Install Isaac Lab / Isaac Sim as ./IsaacLab following the Isaac Lab docs.
mkdir -p insightbench-workspace
cd insightbench-workspace

git clone https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout f6b16f6d97155e3ce34ab2a1ec145e9413588197

git clone https://github.com/your-github-org-or-user/InsightBench.git
cd InsightBench

# 2. Run the eval-only installer.
# Defaults infer ../IsaacLab and ../lerobot from this repo location.
bash install_eval.sh
# or specify a custom env name and paths:
ISAACLAB_ROOT=/custom/path/to/IsaacLab \
LEROBOT_ROOT=/custom/path/to/lerobot \
MINICONDA_ROOT=/custom/path/to/miniconda3 \
bash install_eval.sh my_env

# 3. Activate
conda activate env_insightbench_eval

# 4. (Optional) For InstructionGPT policy:
export OPENAI_API_KEY="sk-..."            # set in shell profile, never hardcode

# 5. Log in to HuggingFace (first time only — for checkpoint download)
huggingface-cli login
```

### Option B — Full Install (data collection + interact scenes)

```bash
# Same sibling layout as above, plus CuRobo as ../curobo.
git clone https://github.com/NVlabs/curobo.git ../curobo
bash install.sh
# or specify paths when IsaacLab/LeRobot/conda/CuRobo are elsewhere:
ISAACLAB_ROOT=/custom/path/to/IsaacLab \
LEROBOT_ROOT=/custom/path/to/lerobot \
MINICONDA_ROOT=/custom/path/to/miniconda3 \
CUROBO_ROOT=/custom/path/to/curobo \
bash install.sh my_env
conda activate env_insightbench
```

> **LeRobot baseline note**: Both scripts install LeRobot from `LEROBOT_ROOT`
> (default: repo-adjacent `../lerobot`). For a unified all-policy evaluation
> environment, this checkout must include `pi0`, `diffusion`, `smolvla`, and
> `groot` under `src/lerobot/policies/`. The install scripts fail fast when
> `groot` is missing because older LeRobot checkouts do not support GR00T
> evaluation. Install with `LEROBOT_EXTRAS=pi,smolvla,groot`
> (the default). The `pi` extra is included because LeRobot 0.4.1 declares it
> as the Pi0/OpenPI dependency set; omitting it would make a fresh Pi0 runtime
> under-specified. The scripts pin `transformers==4.57.1` after installing
> LeRobot extras: its broad allowed range otherwise resolves to 4.53.3 in a
> fresh environment, while the GR00T Eagle processor is validated with 4.57.1.
> InsightBench installs a version-checked Pi0 SigLIP shim immediately before Pi0
> model construction. It recreates the two OpenPI changes required by the
> LeRobot 0.4.1 Pi0 path: the SigLIP installation check and bf16 vision-embedding
> casting before a bf16 encoder. This keeps one `transformers==4.57.1` runtime
> for Pi0 and GR00T. A fresh all-policy runtime smoke remains required before
> release certification.
> The official PyPI
> `lerobot` package and latest upstream LeRobot `0.6.x` are **not supported**
> for this IsaacLab runtime.

> **Installer path defaults**: `install_eval.sh` infers `../IsaacLab` and
> `../lerobot` relative to the InsightBench repo. `install.sh` also infers
> `../curobo`. If a candidate is missing, the installer fails fast and asks for
> `ISAACLAB_ROOT`, `LEROBOT_ROOT`, or `CUROBO_ROOT`.

### Dependency Hygiene Notes

The validated runtime is Python 3.11, IsaacLab 0.45.7, NumPy 1.26.4, and
LeRobot 0.4.1 at commit `f6b16f6d97155e3ce34ab2a1ec145e9413588197`.
Fresh installs intentionally pin `opencv-python-headless==4.11.0.86`, matching
LeRobot's declared OpenCV dependency and avoiding GUI OpenCV packages.

`pip check` may still report metadata warnings from upstream optional tooling:

| Package warning | Owner | Runtime impact |
|-----------------|-------|----------------|
| `rerun-sdk` requires `numpy>=2` | LeRobot visualization/telemetry dependency | Policy evaluation imports do not load `rerun`; every LeRobot-compatible `rerun-sdk>=0.24,<0.27` release declares `numpy>=2`, while IsaacLab requires NumPy 1.x. |
| `nvidia-srl-*` missing `usd-core` or requiring `lxml<5` | Isaac Sim URDF exporter prebundle | Optional exporter tooling; not used by InsightBench evaluation. |
| `plotly`, `selenium`, `azure-identity`, `msal-extensions` missing optional dependencies | Isaac Sim prebundled core/cloud tooling | Not used by InsightBench policy evaluation. |

There should be no additional InsightBench-owned dependency conflicts beyond
these known third-party metadata/tooling warnings.

### Download Assets

All simulation assets are distributed through Hugging Face as a compressed
dataset archive:

- [paragon7060/InsightBench-Assets-v0.1](https://huggingface.co/datasets/paragon7060/InsightBench-Assets-v0.1)

Download, verify, and extract:

```bash
huggingface-cli download paragon7060/InsightBench-Assets-v0.1 \
    --repo-type dataset \
    --include "InsightBench-Assets-v0.1.tar.gz" "InsightBench-Assets-v0.1.tar.gz.sha256" \
    --local-dir .

sha256sum -c InsightBench-Assets-v0.1.tar.gz.sha256
mkdir -p Assets
tar -xzf InsightBench-Assets-v0.1.tar.gz -C Assets
```

Expected archive SHA256:

```text
718dbfec6a8402148087885090f48f8d9ce6fd412e18eaa88972af8c1c4af9ed
```

See [`docs/assets.md`](docs/assets.md) for the full asset download,
verification, layout instructions, and Google Drive mirror.

The archive is extracted into the existing `Assets/` directory; the runtime
consumer root is therefore `Assets/`.

The extracted assets have the following structure:

```
Assets/
├── TestSuite/          # Evaluation assets (cabinet, door, bottle) — needed for eval
│   ├── cabinet_suite/
│   ├── door_suite/
│   └── bottle_suite/
├── TrainSuite/         # Training assets for data collection — needed for collect_demo.py
│   ├── cabinet_suite/  # 249 cabinet assets (PartManip, CC BY-NC 4.0)
│   ├── door_suite/     # door assets (AdaManip)
│   └── bottle_suite/   # bottle assets (AdaManip)
├── guides/             # Guide USD overlays (arrows, indicators)
├── FrankaEmika/        # Franka Panda robot USD
└── franka_description/ # Franka URDF/meshes
```

> **Note**: Policy checkpoints can be loaded **directly from HuggingFace Hub** at runtime when
> `policy.checkpoint` is set to a valid Hub repo ID. Evaluation configs do not use placeholder
> checkpoint IDs as defaults; policies without a verified public checkpoint must be overridden.

---

## Quick Start: Evaluate a Policy

> Use `env_insightbench_eval` for benchmark evaluation, or `env_insightbench`
> if you also need data collection tools. Both environments are intended to be
> **unified policy environments** when installed with the tested upstream
> LeRobot checkout or a compatible checkout with the same policy modules.

All policies share a single evaluation entry point. Select a policy via `--config`:

```bash
# Activate the unified evaluation environment:
conda activate env_insightbench_eval

# Log in to HuggingFace (first time only; needed for private/gated repos):
huggingface-cli login

# Optional sanity check: all supported policy modules should import.
python - <<'PY'
from isaaclab.app import AppLauncher
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.groot.modeling_groot import GrootPolicy
print("InsightBench evaluation environment OK")
PY

# Evaluate Pi0 on a door task.
# Override policy.checkpoint with either a Hub repo ID or a local pretrained_model path.
python scripts/evaluate.py \
    --config configs/eval/pi0.yaml \
    --object door \
    --asset_path 99660039960014l \
    --task_idx 0 \
    policy.checkpoint=your-hf-user-or-org/pi0-repo \
    policy.checkpoint_subfolder=pi0_guide_v7/checkpoints/050000/pretrained_model \
    --num_envs 8 \
    --headless --enable_cameras

# Evaluate GR00T on a bottle task.
# GR00T also requires a verified Hub repo ID or local checkpoint path.
python scripts/evaluate.py \
    --config configs/eval/groot.yaml \
    --object bottle \
    --asset_path 14b \
    --task_idx 3 \
    policy.checkpoint=your-hf-user-or-org/groot-repo \
    --num_envs 8 \
    --headless --enable_cameras

# Override any config field from the command line:
python scripts/evaluate.py \
    --config configs/eval/pi0.yaml \
    --object cabinet --asset_path 31249 --task_idx 1 \
    policy.checkpoint=local/pretrained_model \
    policy.checkpoint_subfolder= \
    policy.dataset_stats_root=data/paragon7060/INSIGHTfixpos_v3 \
    eval.save_video=true \
    --num_envs 8 --headless --enable_cameras
```

Results are saved per-run as JSON files under `eval.results_dir` (see config).

`policy.dataset_stats_root` is optional. Leave it unset to load stats from
`policy.dataset_stats_repo`, or set it to a local LeRobot dataset directory containing
`meta/stats.json`.

For SmolVLA, one of these two stats sources is required. Its public config
intentionally has no default Hub dataset repo; provide a verified override such as
`policy.dataset_stats_repo=your-hf-user-or-org/dataset-repo` or
`policy.dataset_stats_root=path/to/lerobot-dataset`.

### Full Benchmark Evaluation (All Assets × All Tasks)

```bash
# Runs all (object, asset, task) combinations sequentially on GPU 0
# Use a local config copy with policy.checkpoint set to your verified checkpoint.
chmod +x scripts/eval_batch.sh
./scripts/eval_batch.sh --config configs/eval/local_pi0.yaml --num_envs 8 --gpu 0

# Aggregate results into a summary table
python scripts/aggregate_results.py --results_dir outputs/results/pi0 --save_csv
```

Failed runs are automatically collected in `outputs/retry_failed.sh`.

### Persistent Category Evaluation

For repeated evaluation, the persistent launcher starts one worker per GPU and loads the
policy once per worker. Categories can be selected independently from the seen/unseen split.

```bash
# Preview the exact job plan without starting Isaac Sim.
./scripts/eval_batch_persistent.sh \
    --config configs/eval/groot.yaml \
    --checkpoint /path/to/checkpoint \
    --categories door,bottle \
    --splits seen,unseen \
    --gpus 0,1 \
    --num-envs 2 \
    --run-dir outputs/eval_runs/groot_door_bottle \
    --dry-run

# Run the previewed plan: remove --dry-run and add --resume.
# After interruption, repeat that run command with --resume again.
```

Valid categories are `cabinet`, `door`, and `bottle`. Use `--splits seen` or
`--splits unseen` to evaluate one benchmark split, and `--assets id1,id2` for an
optional asset-level subset. Repeat `--override key=value` for additional config values.

Match the evaluation contract to the checkpoint before launching a full run. In particular,
the visual keys in the checkpoint's `config.json`, its training prompt set, and the evaluation
horizon must agree with the runtime settings. For an instruction-trained GR00T checkpoint that
was evaluated with a 300-step horizon, add:

```bash
--override policy.infer_type=instruction \
--override eval.eval_steps=300
```

A checkpoint whose `input_features` contains only `wrist` and `right_shoulder` was not trained
with the `guide` camera. Do not treat it as a three-camera guide checkpoint merely because the
dataset still contains a guide video. Check a small asset subset first with `--assets` before
starting the complete benchmark.

The launcher writes `run_manifest.json`, the complete `jobs.tsv`, and per-GPU assignments
before starting Isaac Sim. Workers add status files, per-job logs, results, and videos;
completed runs also receive `retry_failed.sh` for any unfinished jobs.
The original `scripts/evaluate.py` and `scripts/eval_batch.sh` entry points remain available.

---

## Supported Policies

| Config | Policy | Notes |
|--------|--------|-------|
| `configs/eval/pi0.yaml` | [Pi0](https://github.com/Physical-Intelligence/openpi) | Guide cam as third input |
| `configs/eval/diffusion.yaml` | [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | Images resized to 224×224 |
| `configs/eval/smolvla.yaml` | [SmolVLA](https://github.com/huggingface/lerobot) | 16-dim state (ee + joints) |
| `configs/eval/groot.yaml` | [GR00T](https://github.com/NVIDIA/Isaac-GR00T) | No dataset stats needed |
| `configs/eval/instruction_gpt.yaml` | Instruction-GPT + Pi0 | Requires `OPENAI_API_KEY` |

The default evaluation configs require `policy.checkpoint` override unless the config
explicitly names a verified public checkpoint. For Pi0 Hub checkpoints stored below a repo
subdirectory, also set `policy.checkpoint_subfolder`.

Pi0 checkpoints saved with the supported LeRobot 0.3.4 config schema are
decoded in memory for the pinned 0.4.1 runtime; the checkpoint's `config.json`
is not changed. The compatibility path accepts only the verified 224x224,
1024-width, eager-attention, non-Aloha configuration contract and rejects
unclassified or semantically different settings.

On the pinned LeRobot 0.4.1 baseline, Pi0 and SmolVLA load dataset statistics
through their official policy pre/post-processor factories, not their model
constructors.

GR00T uses the same `scripts/evaluate.py` entry point as the other policies, but
it requires the tested LeRobot checkout described above, or a compatible checkout
with `lerobot.policies.groot`. A default IsaacLab LeRobot checkout that lacks
`lerobot.policies.groot` is not sufficient.

### Adding a New Policy

1. Create a wrapper in `insightbench/policies/your_policy.py` inheriting from `PolicyBase`.
2. Register it in `insightbench/policies/__init__.py`.
3. Add a config in `configs/eval/your_policy.yaml`.

---

## Data Collection

> **Requires the full environment** (`install.sh`) — CuRobo is used for motion planning.
>
> **Assets required**: Download the full Hugging Face asset archive and extract
> it into `Assets/`; see [`docs/assets.md`](docs/assets.md).

### Single Asset

```bash
conda activate env_insightbench

# Collect door demonstrations (200 successful episodes, 8 parallel envs)
python scripts/collect_demo.py \
    --object door \
    --asset_id 99660039960014l \
    --scene_key 3a \
    --dataset_name your-hf-user-or-org/INSIGHT-demo \
    --num_envs 8 \
    --target_episodes 200 \
    --pos_rand \
    --headless --enable_cameras
```

Object base position randomization is enabled by default during data collection. Use
`--pos_rand` to make that choice explicit, or `--no_pos_rand`/`--fixed_pos` to
collect a fixed-base-position dataset or debug a difficult asset.

### Bounded Smoke Collection

Production collection keeps `decimation=300`. For a short execution smoke, use
`--smoke_action_steps`: it deliberately overrides production timing with
`decimation=10` and `render_interval=10`, executes only the requested number of
planned action steps, and applies a per-step deadline inside the physics loop.
The smoke logs `EnvStepBegin`, `EnvStepEnd`, and a final `CollectSmoke` result.

First verify planner-to-environment execution without LeRobot frame writes:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_demo.py \
    --object door \
    --asset_id 99660039960014l \
    --scene_key 3a \
    --dataset_name local/insightbench-smoke-door-dry \
    --asset_dir Assets/TrainSuite/door_suite \
    --num_envs 1 \
    --target_episodes 1 \
    --max_loops 1 \
    --no_pos_rand \
    --debug_collect --progress_interval 1 \
    --smoke_action_steps 2 \
    --no_frame_write \
    --headless --enable_cameras
```

Then validate image/frame writing and the LeRobot episode save path with a separate
local dataset. `--smoke_save_episode` saves a deliberately partial smoke episode;
it is a writer check, not a successful demonstration.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_demo.py \
    --object door \
    --asset_id 99660039960014l \
    --scene_key 3a \
    --dataset_name local/insightbench-smoke-door-frames \
    --asset_dir Assets/TrainSuite/door_suite \
    --num_envs 1 \
    --target_episodes 1 \
    --max_loops 1 \
    --no_pos_rand \
    --debug_collect --progress_interval 1 \
    --smoke_action_steps 2 \
    --smoke_save_episode \
    --headless --enable_cameras
```

Scene keys per object type:
- **cabinet**: `1ext` (one key per asset, drawer variant is asset-specific)
- **door**: `3a` `3b` `3c` `3d` (pull-left, pull-right, push-CCW, push-CW)
- **bottle**: `5a`–`5h` (open/close × grip orientation variants)

### Batch Collection (All Assets × All Scene Keys)

```bash
# Single GPU — iterates all (asset, scene_key) combinations under Assets/TrainSuite/
chmod +x scripts/collect_batch.sh
./scripts/collect_batch.sh \
    --dataset_name your-hf-user-or-org/INSIGHT-demo \
    --target_episodes 100 \
    --num_envs 8 \
    --pos_rand \
    --gpu 0

# Multi-GPU (4 GPUs, round-robin distribution):
./scripts/collect_batch.sh \
    --dataset_name your-hf-user-or-org/INSIGHT-demo \
    --target_episodes 100 \
    --num_envs 8 \
    --pos_rand \
    --gpus 0,1,2,3
```

Each (asset, scene_key) pair runs until `--target_episodes` successful episodes are collected, then moves to the next. Failed runs are saved to `outputs/collect_retry_failed.sh`.

Demonstrations use CuRobo motion planning with per-scene action sampling. Progress is logged to `outputs/collect_logs/collection_results.csv`.

---

## Interactive Scene Testing

> **Requires the full environment** (`install.sh`) — CuRobo is used for motion planning.

Keyboard-driven interactive scripts are provided for all scenes. Isaac Sim output is suppressed to `/tmp/isaacsim_interactive.log` so the TUI stays clean.

| Script | Scene | Notes |
|--------|-------|-------|
| `scene_interact/interact_scene1_interactive.py` | Cabinet (Top Drawer) | Pull direction: +Y |
| `scene_interact/interact_scene2_interactive.py` | Microwave | Pull direction: +Y |
| `scene_interact/interact_scene3_interactive.py` | Door | Pull direction: −Y |
| `scene_interact/interact_scene5_interactive.py` | Bottle | `--task squeeze\|open\|open_rev\|close\|close_rev` |

Basic (text-input) variants are also available without the `_interactive` suffix.

### Running with WebRTC Livestream (view from local PC)

```bash
conda activate env_insightbench
cd InsightBench

# Cabinet drawer:
python scene_interact/interact_scene1_interactive.py \
    --num_envs 1 --livestream 2 --enable_cameras

# Microwave:
python scene_interact/interact_scene2_interactive.py \
    --num_envs 1 --livestream 2 --enable_cameras

# Door:
python scene_interact/interact_scene3_interactive.py \
    --num_envs 1 --livestream 2 --enable_cameras

# Bottle (squeeze task):
python scene_interact/interact_scene5_interactive.py \
    --task squeeze --num_envs 1 --livestream 2 --enable_cameras
```

Then open **Isaac Sim WebRTC Streaming Client** on your local PC and connect to the server IP.

### Interactive Keyboard Controls

| Key | Action |
|-----|--------|
| `0`–`6` | Select skill (0=Approach IK, 1=Grasp, 2=Pull, 3=Rotate, 4=Push/Pull MG, 5=Gripper Open, 6=Gripper Close) |
| `W/S` | Move target Y+/− |
| `A/D` | Move target X+/− |
| `Q/E` | Move target Z+/− |
| `↑↓` | Pitch +/− |
| `←→` | Yaw +/− |
| `[/]` | Roll +/− |
| `A/D` *(gauge mode, skills 2–3)* | Decrease / increase pull distance or rotation angle |
| `+/-` | Scale position step size |
| `ENTER` or `SPACE` | Execute current action |
| `R` | Reset environment |
| `ESC` or `Q` | Quit |

A **frame marker** (XYZ axes) appears in the simulation at the target EE position for approach/grasp skills.
A **sphere marker** shows the target endpoint for pull and rotate skills.

### Required Environment Variable

```bash
# Set automatically by conda activate hook.
# If running without conda activate, set manually:
export ISAACLAB_PATH=/path/to/IsaacLab
```

---

## Project Structure

```
InsightBench/
├── insightbench/               # Core library
│   ├── envs/builder.py         # Unified environment factory
│   ├── policies/               # Policy wrappers (Pi0, GR00T, SmolVLA, ...)
│   └── utils/                  # obs, video, results utilities
├── scripts/
│   ├── evaluate.py             # Single evaluation entry point
│   ├── evaluate_persistent.py  # Persistent policy worker
│   ├── collect_demo.py         # Demonstration data collection
│   ├── eval_batch.sh           # Batch evaluation over all assets/tasks
│   ├── eval_batch_persistent.sh # Category-aware deployment launcher
│   └── aggregate_results.py    # Result aggregation + CSV export
├── configs/
│   ├── eval/                   # Per-policy YAML configs
│   ├── task/                   # Task definitions (asset dirs, task libs)
│   └── env/                    # Environment defaults
├── cfg/                        # Isaac Lab scene/reward/obs/event configs
├── custom_lab/                 # Extended Isaac Lab env classes
├── interact/                   # CuRobo motion planning integration
└── sim_config/                 # CuRobo collision world YAMLs
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'isaacsim'`
Isaac Sim paths are injected by the conda activate hook. Make sure you activated the environment with `conda activate <ENV_NAME>` — running `bin/python` directly skips the hook.

### `OSError: libtorch_global_deps.so: cannot open shared object file`
The libstdc++ preload is missing. The activate hook handles this automatically, but if you run scripts outside of the conda env, set:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### `ModuleNotFoundError: No module named 'pkg_resources'`
The installers use a temporary `setuptools<70` bootstrap only for the three
IsaacLab editable installs, together with validated `numpy==1.26.4`,
`pillow==11.2.1`, and `toml`; IsaacLab setup metadata imports `toml`. They then
upgrade to `setuptools==80.9.0` before LeRobot because LeRobot 0.4.1 requires
`setuptools>=71,<81`. `flatdict==4.0.1` was validated with that final version.
If a manual install fails during the IsaacLab metadata stage:
```bash
pip install "numpy==1.26.4" "pillow==11.2.1" "setuptools<70" toml
pip install <failing_package> --no-build-isolation
pip install "setuptools==80.9.0"
```

### CuRobo build: `CUDA version (11.7) mismatches PyTorch (12.8)`
The `/usr/local/cuda` symlink may point to an older CUDA version. `install.sh` overrides it automatically. If you run the install manually, prepend:
```bash
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
    pip install -e /path/to/curobo --no-build-isolation
```

### numpy / OpenCV version conflicts
The unified LeRobot `0.4.1` baseline and IsaacLab runtime are pinned to a Python
3.11 / NumPy 1.x environment. The install scripts resolve package conflicts by
pinning:
- `numpy==1.26.4` — required by isaaclab (`numpy<2`)
- `opencv-python-headless==4.11.0.86` — LeRobot's declared OpenCV dependency,
  pinned to the NumPy 1.x-compatible version used in `groot_lab`
- `gymnasium==1.2.0` — required by isaaclab

---

## Environment Details

- **Simulator**: Isaac Sim 5.0 (PhysX GPU)
- **Robot**: Franka Panda (8-DOF joint position control)
- **Observation cameras**: wrist (224×224), right-shoulder (224×224), guide (224×224 for eval; 672×672 for collect datasets)
- **Simulation rate**: 120 Hz, decimation = 12 (10 Hz action frequency)
- **Episode length**: 100 steps (~10 s per skill)
- **Parallelism**: Tested with 8 × NVIDIA A100 (80 GB)

---

## Citation

If you use InsightBench in your research, please cite:

```bibtex
@article{insightbench2025,
  title   = {InsightBench: A Benchmark for Robot Manipulation with Visual Guides},
  author  = {[Authors]},
  journal = {[Venue]},
  year    = {2025},
  url     = {https://github.com/your-github-org-or-user/InsightBench}
}
```

---

## Asset Attribution

InsightBench uses simulation assets derived from the following works. We thank the authors for making their assets publicly available.

### Cabinet (Drawer) Assets — PartManip
Assets are derived from the [PartManip](https://github.com/PKU-EPIC/PartManip) dataset (drawer subset), converted from URDF to USD format for Isaac Sim.

> W. Geng et al., "PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations," CVPR 2023.

```bibtex
@inproceedings{geng2023partmanip,
  title     = {PartManip: Learning Cross-Category Generalizable Part Manipulation Policy from Point Cloud Observations},
  author    = {Geng, Haoran and Xu, Helin and Zhao, Chengyang and Xu, Chao and Yi, Li and Huang, Siyuan and Wang, He},
  booktitle = {CVPR},
  year      = {2023}
}
```

Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial use only.

### Door & Bottle Assets — AdaManip
Assets are derived from the [AdaManip](https://github.com/yuanfei-Wang/AdaManip) dataset, with axis corrections and USD conversion applied for Isaac Sim compatibility.

> Y. Wang et al., "AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning," 2024.

```bibtex
@article{wang2024adamanip,
  title   = {AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning},
  author  = {Wang, Yuanfei and others},
  year    = {2024},
  url     = {https://github.com/yuanfei-Wang/AdaManip}
}
```

No explicit license is provided by the original authors. Assets are redistributed here for academic research purposes with full attribution. If you are the author and have concerns, please open an issue or contact us.

---

## License

InsightBench code is licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.

Simulation assets are subject to the licenses of their respective sources as noted above.
