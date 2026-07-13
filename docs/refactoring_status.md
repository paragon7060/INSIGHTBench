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
- Eval path no longer depends on CuRobo.
- Policy stats trimming was fixed for Pi0/Diffusion/SmolVLA/InstructionGPT.
- Public eval configs were cleaned up around checkpoint/stat overrides.
- Lightweight tests were added and previously passed.
- Cabinet collect metadata parsing was fixed for full TrainSuite directory
  names and `1ext` scene keys.

### Eval Smoke

Pi0 eval smoke has passed for representative door, cabinet, and bottle tasks.
The runs completed environment build, policy load, observation/action step,
result JSON save, and video save.

Generated results included:

- `outputs/results/pi0/door_99660039960014l_task0.json`
- `outputs/results/pi0/cabinet_46130_task0.json`
- `outputs/results/pi0/bottle_14b_task0.json`
- `outputs/results/pi0/summary.csv`

Current scores were `0/1` for each smoke run. That is not treated as a smoke
failure; the smoke objective is runtime pipeline validation.

### Asset Staging

Physical asset copy completed under:

- `InsightBench-Assets-Staging/`

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

Google Drive archive creation is intentionally paused until collect smoke
passes.

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

Current default collect render behavior is:

```text
decimation=300
render_interval=300
```

This avoids rendering RTX camera frames on every physics substep.

`interact/motion_generator.py` now exposes `last_plan_debug` for collect debug
logs without changing the `command()` return interface.

## Current Blockers

### B1. Collect Smoke Needs Rerun After Render Fix

Before the render fix, door collect smoke reached:

```text
env build: PASS
CuRobo planning: PASS
max_T=86 to 125
action execution: timeout
```

Debug logs showed `env.step()` taking about 13 seconds per trajectory waypoint,
while frame writing was about 0.05 seconds. The root cause was likely:

```text
decimation=300 + render_interval=1 + RTX cameras
```

Code has since been changed to set `render_interval=decimation` by default, but
Run/Eval has not yet rerun collect smoke after that change.

### B2. GR00T Blocked In Current Environment

Current `env_insightbench` uses:

```text
/home/seonho/workspace/IsaacLab/lerobot
LeRobot 0.3.4 editable install
```

That checkout does not include:

```text
lerobot.policies.groot
```

A GROOT-capable checkout exists at:

```text
/home/seonho/workspace/ws_groot/lerobot
```

but it is not installed in `env_insightbench`. A PYTHONPATH import attempt then
failed on missing dependencies such as `accelerate`, `peft`, and `timm`.

This is classified as an environment/dependency blocker, not an InsightBench
wrapper import bug.

### B3. Latest LeRobot 0.6.x Cannot Share The Current IsaacLab Env

Latest LeRobot migration is not a small shim update.

Known package constraints:

```text
IsaacLab latest: Python ==3.11.*, numpy <2
LeRobot 0.6.x: Python >=3.12, numpy >=2
```

Therefore a single environment containing current IsaacLab and upstream
LeRobot 0.6.x is officially incompatible. The likely release architecture is:

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
  --enable_cameras
```

### Collect Smoke With Debug

Use the staging TrainSuite path.

```bash
CUDA_VISIBLE_DEVICES=3 timeout 8m conda run --no-capture-output -n env_insightbench \
python scripts/collect_demo.py \
  --object door \
  --asset_id 99660039960014l \
  --scene_key 3a \
  --dataset_name local/insightbench-smoke-door-debug \
  --asset_dir ./InsightBench-Assets-Staging/TrainSuite/door_suite \
  --num_envs 1 \
  --target_episodes 1 \
  --max_loops 1 \
  --no_pos_rand \
  --debug_collect \
  --progress_interval 5 \
  --skill_timeout_s 180 \
  --headless \
  --enable_cameras
```

To isolate frame writing:

```bash
CUDA_VISIBLE_DEVICES=3 timeout 8m conda run --no-capture-output -n env_insightbench \
python scripts/collect_demo.py \
  --object door \
  --asset_id 99660039960014l \
  --scene_key 3a \
  --dataset_name local/insightbench-smoke-door-debug-noframes \
  --asset_dir ./InsightBench-Assets-Staging/TrainSuite/door_suite \
  --num_envs 1 \
  --target_episodes 1 \
  --max_loops 1 \
  --no_pos_rand \
  --debug_collect \
  --progress_interval 5 \
  --skill_timeout_s 180 \
  --no_frame_write \
  --headless \
  --enable_cameras
```

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
  policy.checkpoint=/mntvol1/INSIGHTBench/ckpt/SmolVLA_v5_full/checkpoints/030000/pretrained_model
```

### GR00T Status

Do not rerun GR00T smoke in `env_insightbench` until a GROOT-capable LeRobot
environment is prepared.

## Remaining TODOs

### Immediate TODOs

1. Run/Eval: rerun door collect smoke after the `render_interval=decimation`
   fix.
2. Run/Eval: run the no-frame-write collect smoke to isolate remaining
   bottlenecks.
3. Run/Eval: run SmolVLA door smoke.
4. Headquarters: classify SmolVLA result as pass, checkpoint issue, stats issue,
   or wrapper issue.
5. Headquarters: keep GR00T blocked until environment strategy is decided.

### Asset Release TODOs

1. Run/Eval: run door/cabinet/bottle collect smoke using
   `InsightBench-Assets-Staging`.
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

Current blocker: `/home/seonho/workspace/InsightBench` is not recognized as a
git repository by `git status`.
