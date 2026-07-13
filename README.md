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

InsightBench provides **two install scripts** depending on your use case:

| Script | Use case | CuRobo | Install time |
|--------|----------|--------|-------------|
| `install_eval.sh` | Run benchmark evaluations | ✗ Not needed | ~5 min |
| `install.sh` | Data collection + interact scenes | ✓ Required | ~30 min (CUDA compile) |

> **Why two environments?** [CuRobo](https://github.com/NVlabs/curobo) is only needed for motion-planning-based data collection (`scripts/collect_demo.py`) and interactive scene testing (`scene_interact/`). Evaluation scripts run without it.

---

### Option A — Evaluation Only (recommended for most users)

```bash
# 1. Clone this repo next to IsaacLab
git clone https://github.com/your-github-org-or-user/InsightBench.git
cd InsightBench

# 2. Run the eval-only installer (no CuRobo, ~5 min)
bash install_eval.sh                      # creates env_insightbench_eval
# or specify a custom name:
bash install_eval.sh my_env
# or override paths when IsaacLab/conda are elsewhere:
ISAACLAB_ROOT=/path/to/IsaacLab MINICONDA_ROOT=/path/to/miniconda3 bash install_eval.sh my_env

# 3. Activate
conda activate env_insightbench_eval

# 4. (Optional) For InstructionGPT policy:
export OPENAI_API_KEY="sk-..."            # set in shell profile, never hardcode

# 5. Log in to HuggingFace (first time only — for checkpoint download)
huggingface-cli login
```

### Option B — Full Install (data collection + interact scenes)

```bash
# Same clone step as above, then:
bash install.sh                           # creates env_insightbench (~30 min, CuRobo CUDA compile)
# Override paths when IsaacLab/conda/CuRobo are elsewhere:
ISAACLAB_ROOT=/path/to/IsaacLab MINICONDA_ROOT=/path/to/miniconda3 CUROBO_ROOT=/path/to/curobo bash install.sh my_env
conda activate env_insightbench
```

> **LeRobot fork note**: Both scripts install a customized LeRobot fork from
> `IsaacLab/lerobot/`. The official PyPI `lerobot>=0.4.0` is **not compatible** —
> it has breaking API changes in dataset_stats normalization and does not include GuideVLA.

### Download Assets

All simulation assets are hosted on HuggingFace at [`paragon7060/InsightBench-Assets`](https://huggingface.co/datasets/paragon7060/InsightBench-Assets).

The repo has the following structure:

```
InsightBench-Assets/
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

```bash
# Option A — Evaluation only (TestSuite + guides + robot, ~300MB):
huggingface-cli download paragon7060/InsightBench-Assets \
    --repo-type dataset \
    --include "TestSuite/**" "guides/**" "FrankaEmika/**" "franka_description/**" \
    --local-dir Assets/

# Option B — Full download including TrainSuite for data collection (~3GB):
huggingface-cli download paragon7060/InsightBench-Assets \
    --repo-type dataset \
    --local-dir Assets/
```

> **Note**: Policy checkpoints can be loaded **directly from HuggingFace Hub** at runtime when
> `policy.checkpoint` is set to a valid Hub repo ID. Evaluation configs do not use placeholder
> checkpoint IDs as defaults; policies without a verified public checkpoint must be overridden.

---

## Quick Start: Evaluate a Policy

> Works with both `env_insightbench_eval` (eval-only) and `env_insightbench` (full).

All policies share a single evaluation entry point. Select a policy via `--config`:

```bash
# Activate environment (eval-only is sufficient):
conda activate env_insightbench_eval

# Log in to HuggingFace (first time only; needed for private/gated repos):
huggingface-cli login

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
    policy.checkpoint=ckpt/pi0_v2/checkpoints/100000/pretrained_model \
    policy.checkpoint_subfolder= \
    policy.dataset_stats_root=data/paragon7060/INSIGHTfixpos_v3 \
    eval.save_video=true \
    --num_envs 8 --headless --enable_cameras
```

Results are saved per-run as JSON files under `eval.results_dir` (see config).

`policy.dataset_stats_root` is optional. Leave it unset to load stats from
`policy.dataset_stats_repo`, or set it to a local LeRobot dataset directory containing
`meta/stats.json`.

### Full Benchmark Evaluation (All Assets × All Tasks)

```bash
# Runs all (object, asset, task) combinations sequentially on GPU 0
chmod +x scripts/eval_batch.sh
./scripts/eval_batch.sh --config configs/eval/pi0.yaml --num_envs 8 --gpu 0

# Aggregate results into a summary table
python scripts/aggregate_results.py --results_dir outputs/results/pi0 --save_csv
```

Failed runs are automatically collected in `outputs/retry_failed.sh`.

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

### Adding a New Policy

1. Create a wrapper in `insightbench/policies/your_policy.py` inheriting from `PolicyBase`.
2. Register it in `insightbench/policies/__init__.py`.
3. Add a config in `configs/eval/your_policy.yaml`.

---

## Data Collection

> **Requires the full environment** (`install.sh`) — CuRobo is used for motion planning.
>
> **Assets required**: Download `TrainSuite` from [`paragon7060/InsightBench-Assets`](https://huggingface.co/datasets/paragon7060/InsightBench-Assets) (Option B above).

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
│   ├── collect_demo.py         # Demonstration data collection
│   ├── eval_batch.sh           # Batch evaluation over all assets/tasks
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
Occurs during install when a legacy dependency (`flatdict`) uses old-style `setup.py`. The install scripts handle this automatically by pinning `setuptools<70`. If you see this error during manual install:
```bash
pip install "setuptools<70"
pip install <failing_package> --no-build-isolation
```

### CuRobo build: `CUDA version (11.7) mismatches PyTorch (12.8)`
The `/usr/local/cuda` symlink may point to an older CUDA version. `install.sh` overrides it automatically. If you run the install manually, prepend:
```bash
CUDA_HOME=/usr/local/cuda-12.5 PATH=/usr/local/cuda-12.5/bin:$PATH \
    pip install -e /path/to/curobo --no-build-isolation
```

### numpy / opencv version conflicts
LeRobot (fork v0.3.4) and isaaclab have conflicting numpy/opencv constraints. The install scripts resolve this by pinning:
- `numpy==1.26.4` — required by isaaclab (`numpy<2`)
- `opencv-python==4.11.0.86` — last version compatible with numpy 1.x
- `gymnasium==1.2.0` — required by isaaclab (lerobot's `<1.0.0` constraint is unused at runtime)

---

## Environment Details

- **Simulator**: Isaac Sim 5.0 (PhysX GPU)
- **Robot**: Franka Panda (8-DOF joint position control)
- **Observation cameras**: wrist (224×224), right-shoulder (224×224), guide (672×672)
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
