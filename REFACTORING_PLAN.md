# InsightBench Refactoring Plan

This document is the central handoff file for coordinating InsightBench
refactoring across separate threads.

Use it as the source of truth when opening Code Threads or Run/Eval Threads.
Each thread should handle only the TODO it was assigned.

작업 환경은 conda env "env_insightbench" 에서 수행한다.

## Thread Roles

### Headquarters Thread

- Owns prioritization and final decisions.
- Splits failures into Code Thread tasks.
- Reviews Run/Eval reports.
- Decides whether an issue is code, environment, asset, checkpoint, or docs.

### Code Thread

- Mutates code only for the assigned TODO.
- Must not run broad cleanup unless assigned.
- Must not change unrelated files.
- Must report changed files, verification commands, and remaining risks.

### Run/Eval Thread

- Runs validation and collects logs.
- Must not modify code.
- Must not delete files or clean outputs.
- Must report commands, failures, stack traces, and Code Thread candidates.

## Standard Code Thread Prompt

```text
너는 /home/seonho/workspace/InsightBench 리팩터링 Code Thread다.

/home/seonho/workspace/InsightBench/REFACTORING_PLAN.md 를 읽고,
지정된 TODO 하나만 수행하라.

규칙:
- 관련 없는 리팩터링 금지.
- 관련 없는 파일 삭제/정리 금지.
- 실패하면 원인과 재현 명령을 보고하라.
- 완료 후 수정 파일, 검증 명령, 남은 위험을 보고하라.
```

## Standard Run/Eval Thread Prompt

```text
너는 /home/seonho/workspace/InsightBench Run/Eval 검증 Thread다.

/home/seonho/workspace/InsightBench/REFACTORING_PLAN.md 를 읽고,
지정된 P2 TODO만 수행하라.

규칙:
- 코드 수정 금지.
- 삭제/cleanup 금지.
- 전체 benchmark 실행 금지.
- smoke test는 num_envs=1, 최소 episode로 실행.
- 실패하면 command, 단계, stack trace, 관련 파일 후보를 보고하라.
```

## Current Status Summary

The repo has already moved toward a public benchmark structure:

- `insightbench/` policy/env/utils package exists.
- `scripts/evaluate.py` is the unified evaluation entry point.
- `scripts/collect_demo.py` is the unified data collection entry point.
- `configs/eval` and `configs/task` separate policy and task settings.
- `scripts/eval_batch.sh` and `scripts/aggregate_results.py` exist.
- Python syntax compilation previously passed with `python3 -m compileall`.

Known risks from initial inspection:

- Evaluation/documentation contract around CuRobo has been a blocker.
- Policy stats trimming needed correction.
- Configs had local paths and placeholders.
- Packaging may not include all runtime top-level modules.
- Install scripts contained local hardcoded paths.
- Legacy Isaac Lab imports and old paths may remain.
- Repo hygiene has been imperfect because generated outputs and cache files exist.

## Phase 1: Structure And Bug Stabilization

Phase 1 makes the repo internally consistent enough for real runtime validation.

### P1-TODO 1: Remove CuRobo Dependency From Evaluation

Code Thread: yes.

Goal:
Make `scripts/evaluate.py` run in eval-only environments without CuRobo.

Files:

- `scripts/evaluate.py`
- `README.md`
- `install_eval.sh`
- Reference only: `scripts/collect_demo.py`
- Reference only: `interact/motion_generator.py`

Instructions:

- Remove `TrajectoryGenerator` import and construction from `scripts/evaluate.py`.
- Implement warmup without CuRobo by holding current robot joint positions.
- Keep `env.step()` action shape as `(env.num_envs, 9)`.
- Remove `curobo_mg` from `run_episode()` signature and call sites.
- Remove or rename CuRobo evaluation log messages.
- Do not modify `collect_demo.py` CuRobo behavior.

Forbidden:

- Do not delete or rewrite `interact/motion_generator.py`.
- Do not modify policy wrappers or configs in this TODO.
- Do not perform broad formatting or cleanup.

Verification:

```bash
python3 -m compileall -q scripts insightbench
rg -n "TrajectoryGenerator|curobo_mg|CuRobo" scripts/evaluate.py
```

Completion report:

- Changed files.
- How warmup now works.
- Whether evaluation runtime still references CuRobo.
- Verification results.

### P1-TODO 2: Fix Policy Dataset Stats Trimming

Code Thread: yes.

Goal:
Ensure Diffusion, SmolVLA, and InstructionGPT trim `dataset_meta.stats`, not the
metadata object itself.

Files:

- `insightbench/policies/pi0.py`
- `insightbench/policies/diffusion.py`
- `insightbench/policies/smolvla.py`
- `insightbench/policies/instruction_gpt.py`
- Optional: `tests/`

Instructions:

- Change calls from `Pi0Wrapper._trim_stats(dataset_meta, cfg)` to
  `Pi0Wrapper._trim_stats(dataset_meta.stats, cfg)`.
- Clarify `_trim_stats()` docstring/type hint: it expects a stats mapping/dict.
- Clean unused imports only in touched policy files.
- If possible, add Isaac-Sim-free tests for fake stats slicing.

Forbidden:

- Do not rewrite checkpoint download logic.
- Do not modify `scripts/evaluate.py` or configs in this TODO.
- Do not add tests requiring Isaac Sim, LeRobot model checkpoints, or network.

Verification:

```bash
python3 -m compileall -q insightbench
python3 -m pytest tests
rg -n "_trim_stats\\(dataset_meta" insightbench/policies
```

Completion report:

- Changed files.
- Exact trim call changes.
- Test status.
- Any skipped validation due to missing dependencies.

### P1-TODO 3: Make Eval Config Defaults Public/Reproducible

Code Thread: yes.

Goal:
Remove local personal paths and ambiguous placeholders from default eval configs.

Files:

- `configs/eval/*.yaml`
- `README.md`

Instructions:

- Remove or comment out personal absolute paths such as `/home/seonho/...`.
- Prefer `dataset_stats_root` as an override example, not a default.
- Keep only known public checkpoint IDs as defaults.
- If no public checkpoint is confirmed, mark the field as requiring override.
- Update README CLI override examples.

Forbidden:

- Do not invent public HF repo IDs.
- Do not change policy loading code.
- Do not rewrite unrelated README sections.

Verification:

```bash
rg -n "/home/seonho|/home/bluepot|YOUR_HF_USERNAME|YOUR_ORG" configs README.md
python3 -m compileall -q insightbench scripts
```

Completion report:

- Configs changed.
- Which policy configs still need user-provided checkpoints.
- Remaining placeholder/path search results and whether they are examples.

### P1-TODO 4: Fix Packaging Coverage

Code Thread: yes.

Goal:
Make modules imported by runtime scripts installable with the package.

Files:

- `pyproject.toml`
- `cfg/`
- `custom_lab/`
- `interact/`
- `demo_gen/`
- `insight_task_suite/`

Instructions:

- Check missing `__init__.py` files.
- Add minimal `__init__.py` where package discovery requires it.
- Include runtime top-level packages in setuptools package discovery.
- Prefer minimal packaging changes over moving files under `insightbench/`.

Likely package include list:

- `insightbench*`
- `cfg*`
- `custom_lab*`
- `interact*`
- `demo_gen*`
- `insight_task_suite*`

Forbidden:

- Do not move directories under `insightbench/` in this TODO.
- Do not globally rewrite import paths.
- Do not edit Isaac Sim runtime behavior.

Verification:

```bash
python3 -m compileall -q cfg custom_lab interact demo_gen insight_task_suite insightbench scripts
python3 -c "import cfg; import custom_lab; import interact; import demo_gen; import insight_task_suite; import insightbench; print('OK')"
```

Completion report:

- `pyproject.toml` changes.
- Added `__init__.py` files.
- Import/compile verification results.

### P1-TODO 5: Generalize Install Script Paths

Code Thread: yes.

Goal:
Allow install scripts to work outside the original local machine.

Files:

- `install.sh`
- `install_eval.sh`
- `README.md`

Instructions:

- Make paths overrideable via environment variables:
  - `ISAACLAB_ROOT`
  - `MINICONDA_ROOT`
  - `CUROBO_ROOT` for full install only.
- Keep reasonable defaults, but validate paths early and print helpful errors.
- Ensure `install_eval.sh` does not validate or install CuRobo.
- Ensure `install.sh` keeps CuRobo installation and validation.
- Update README examples.

Forbidden:

- Do not change dependency versions unless required and reported.
- Do not replace conda install flow.
- Do not run the installation commands as part of this TODO.

Verification:

```bash
bash -n install.sh
bash -n install_eval.sh
```

Completion report:

- Supported environment variables.
- Path validation behavior.
- Bash syntax check results.

### P1-TODO 6: Clean Legacy Imports And Paths

Code Thread: yes.

Goal:
Reduce old Isaac Lab API imports and old `InsightManip` paths.

Files:

- `interact/action.py`
- `interact/action_grasp.py`
- `scene_interact/interact_scene4.py`
- `assets/asset_lab/`
- `assets/asset_lab/README.md`
- Possibly `README.md`

Instructions:

- Search for:
  - `omni.isaac.lab`
  - `source/InsightManip`
  - `/pkgs/IsaacLabCurobo`
- Migrate actively used files to `isaaclab.*` imports where straightforward.
- Update docs/scripts from `source/InsightManip/asset_lab` to `assets/asset_lab`.
- Replace hardcoded `/pkgs/IsaacLabCurobo/Assets` where a current `Assets/`
  or env-var based path is clear.
- If a file is legacy and unsafe to migrate, mark/report it instead of deleting.

Forbidden:

- Do not delete legacy files without explicit approval.
- Do not rewrite CuRobo motion logic.
- Do not invent asset paths.

Verification:

```bash
rg -n "omni\\.isaac\\.lab|source/InsightManip|/pkgs/IsaacLabCurobo" .
python3 -m compileall -q interact scene_interact assets
```

Completion report:

- Fixed legacy paths/imports.
- Remaining legacy hits and why they remain.
- Compile result.

### P1-TODO 7: Repo Hygiene Plan And Safe Cleanup

Code Thread: maybe, but no deletion without approval.

Goal:
Identify generated files, cache, broken git state, and cleanup needs.

Targets:

- `.git`
- `.gitignore`
- `${data}/`
- `outputs/`
- `__pycache__/`
- `*.pyc`
- `insightbench.egg-info/`
- `.pytest_cache/`

Instructions:

- Inspect current state.
- Report deletion candidates.
- Update `.gitignore` if patterns are missing.
- Do not delete generated files unless the user explicitly approves after seeing
  the candidate list.
- Do not delete `.git` or run `git init` without explicit approval.

Forbidden:

- No `rm -rf` without explicit approval.
- No `git reset`, `git checkout`, or destructive git commands.
- Do not mix cleanup with code refactoring.

Verification:

```bash
ls -la .git
git status --short
find . -type f \( -name "*.pyc" -o -path "*/__pycache__/*" -o -path "./insightbench.egg-info/*" -o -path "./outputs/*" \) | head
```

Completion report:

- Git state.
- Cleanup candidate summary.
- `.gitignore` changes, if any.
- Commands requiring approval.

### P1-TODO 8: Single Source Of Truth For Eval Jobs

Code Thread: yes.

Goal:
Avoid duplicate asset/task definitions in `configs/task/*.yaml` and
`scripts/eval_batch.sh`.

Files:

- `scripts/eval_batch.sh`
- `configs/task/cabinet.yaml`
- `configs/task/door.yaml`
- `configs/task/bottle.yaml`
- Optional new file: `scripts/generate_eval_jobs.py`

Instructions:

- Prefer adding `scripts/generate_eval_jobs.py`.
- Read `configs/task/*.yaml`.
- Emit jobs as TSV or JSONL.
- Preserve seen assets before unseen assets.
- Cabinet uses `asset_task_counts`.
- Door/bottle use `len(task_lib)`.
- Update `eval_batch.sh` to consume generated jobs while keeping its CLI.

Forbidden:

- Do not change benchmark asset lists without reason.
- Do not run the full benchmark.
- Do not rewrite `evaluate.py`.

Verification:

```bash
python3 scripts/generate_eval_jobs.py | head
python3 scripts/generate_eval_jobs.py | wc -l
bash -n scripts/eval_batch.sh
python3 -m compileall -q scripts
```

Expected:

- Total jobs: 139.

Completion report:

- Source of truth change.
- Generated job count by object.
- CLI compatibility.

### P1-TODO 9: Add Lightweight Smoke Tests

Code Thread: yes.

Goal:
Add tests that run without Isaac Sim, CuRobo, checkpoints, or network.

Files:

- `tests/`
- `insightbench/utils/obs.py`
- `insightbench/utils/results.py`
- `insightbench/policies/pi0.py`

Test targets:

- `build_obs_state()`
- `build_obs_images()`
- stats trimming helper
- `save_result()`
- `aggregate_results()`

Instructions:

- Use fake tensors and temporary directories.
- Avoid AppLauncher, Isaac Sim, CuRobo, network, or model checkpoint loading.
- If `pi0.py` imports LeRobot at module import time and blocks testing, use a
  minimal refactor to expose a dependency-light stats trim helper.

Forbidden:

- Do not add network or model download tests.
- Do not add Isaac Sim tests.
- Do not add mock-only logic to production runtime paths.

Verification:

```bash
python3 -m pytest tests
python3 -m compileall -q insightbench tests
```

Completion report:

- Test files added.
- What each test covers.
- Test results.

## Phase 2: Runtime Validation

Phase 2 should mostly be performed by Run/Eval Threads. Code Threads should be
opened only when a concrete failure has a code fix candidate.

### P2.5-TODO: Build Standard Asset Bundle

Asset Packaging Thread: yes.
Code Thread: only for tooling or collection-code issues.

Goal:
Create the public asset layout expected by README and runtime code without
mutating original local asset folders.

Decisions:

- Public `TestSuite` excludes known problematic assets:
  - cabinet `45194`
  - door `99655089960011l`
- Train cabinet asset IDs preserve full source directory names.
- Public distribution uses physical copies, not symlinks.
- Google Drive is the distribution channel for the packaged archive.
- `MANIFEST.json` is required.
- `ASSET_LICENSES.md` is required.
- Door, bottle, and cabinet collection each need one smoke test.
- Cabinet collection remains supported, so any `scene_key=1ext` issue must be
  resolved instead of excluding cabinet collection.

Source to target mapping:

- `Assets/PartManip/drawer/train` -> `TrainSuite/cabinet_suite`
- `Assets/AdaManip/door` -> `TrainSuite/door_suite`
- `Assets/AdaManip/bottle` -> `TrainSuite/bottle_suite`
- `Assets/TestSuite/cabinet_suite` -> `TestSuite/cabinet_suite`
- `Assets/TestSuite/door_suite` -> `TestSuite/door_suite`
- `Assets/TestSuite/bottle_suite` -> `TestSuite/bottle_suite`
- `Assets/guides` -> `guides`
- `Assets/FrankaEmika` -> `FrankaEmika`
- `Assets/franka_description` -> `franka_description`

Initial tooling:

- `docs/asset_packaging_plan.md`
- `scripts/generate_asset_manifest.py`
- `ASSET_LICENSES.md`

Verification:

```bash
python3 scripts/generate_asset_manifest.py --assets-root Assets --strict
python3 scripts/generate_asset_manifest.py --assets-root Assets --print-rsync-plan
```

Completion:

- Manifest count matches expected public layout.
- Excluded TestSuite assets appear only in manifest exclusions, not included
  entries.
- Staged bundle passes door, bottle, and cabinet collection smoke tests.

### P2-TODO 1: Eval Smoke Test

Code Thread: no, unless a failure needs fixing.
Run/Eval Thread: yes.

Goal:
Verify `scripts/evaluate.py` runs in Isaac Sim for minimal tasks.

Commands to try:

```bash
pwd
which python
python --version
python -c "from isaaclab.app import AppLauncher; print('IsaacLab OK')"
python -m compileall -q scripts insightbench cfg custom_lab interact
```

Minimal eval smoke:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --config configs/eval/pi0.yaml \
  --object door \
  --asset_path 99660039960014l \
  --task_idx 0 \
  --num_envs 1 \
  --headless --enable_cameras
```

If door succeeds, also try:

- cabinet: `--object cabinet --asset_path 46130 --task_idx 0`
- bottle: `--object bottle --asset_path 14b --task_idx 0`

Report:

- Command.
- Success/failure.
- Failure stage:
  - env launch
  - asset load
  - policy load
  - obs build
  - action step
  - result save
- Key stack trace.
- Related files.

### P2-TODO 2: Policy Checkpoint And Dataset Stats Compatibility

Code Thread: likely after failure evidence.
Run/Eval Thread: yes for inspection.

Goal:
Validate policy configs against checkpoint and stats shapes.

Configs:

- `configs/eval/pi0.yaml`
- `configs/eval/pi0_noguide.yaml`
- `configs/eval/diffusion.yaml`
- `configs/eval/smolvla.yaml`
- `configs/eval/groot.yaml`
- `configs/eval/instruction_gpt.yaml`

Check:

- Checkpoint/repo availability.
- `checkpoint_subfolder` validity.
- Dataset stats path/repo availability.
- `state_dim`.
- `action_dim`.
- Observation image keys expected by model vs produced by `build_obs_images()`.
- State shape expected by model vs produced by `build_obs_state()`.

Report:

- Per-policy status:
  - usable
  - missing checkpoint
  - missing stats
  - shape mismatch
  - unsupported dependency
- Reproduction command for failures.

### P2-TODO 3: Asset Path And Guide Path Integrity

Code Thread: no unless path generation is wrong.
Run/Eval Thread: yes.

Goal:
Validate downloaded/linked Assets match code assumptions.

Inspect:

```bash
find Assets -maxdepth 3 -type d | sort | head -100
find Assets/TestSuite -maxdepth 3 -type f | head -100
find Assets/guides -maxdepth 3 -type f | head -100
```

Relevant files:

- `insightbench/envs/builder.py`
- `cfg/helper.py`
- `cfg/scene1ExtCfg.py`
- `cfg/scene3ExtCfg.py`
- `cfg/Scene5ExtCfg.py`
- `franka.py`

Check:

- TestSuite cabinet/door/bottle assets exist.
- Required USD files exist.
- `bounding_box.json` or equivalent metadata exists when helper expects it.
- Guide USD files exist.
- `ISAACLAB_PATH` and repo-local `Assets/` expectations are understood.

Report:

- Number of asset/task pairs checked.
- Missing paths.
- Whether the issue is missing asset download, broken symlink, or code path bug.

### P2-TODO 4: Batch Job Generation Dry Run

Code Thread: only if mismatch.
Run/Eval Thread: yes.

Goal:
Ensure full benchmark job list is exactly 139 jobs.

Expected:

- Cabinet: sum of `asset_task_counts`.
- Door: `len(seen_assets + unseen_assets) * len(task_lib)`.
- Bottle: `len(seen_assets + unseen_assets) * len(task_lib)`.
- Total: 139.

If generator exists:

```bash
python scripts/generate_eval_jobs.py | head
python scripts/generate_eval_jobs.py | tail
python scripts/generate_eval_jobs.py | wc -l
```

If generator does not exist:

- Compare `configs/task/*.yaml` to hardcoded `scripts/eval_batch.sh`.
- Do not run full benchmark.

Report:

- Expected count.
- Actual/generated count.
- Object-level counts.
- Duplicate/missing jobs.
- Excluded assets accidentally included.

### P2-TODO 5: Result And Video Output Validation

Code Thread: yes if broken.
Run/Eval Thread: yes.

Goal:
Validate evaluation artifacts can be saved and aggregated.

Files:

- `insightbench/utils/results.py`
- `insightbench/utils/video.py`
- `scripts/aggregate_results.py`

Check:

- Result JSON filename and fields.
- Re-run overwrite behavior.
- `successes`, `attempts`, `rate`.
- `aggregate_results.py` object-level and total summaries.
- Video output naming includes enough context.
- `save_video=true` and `save_video=false` both work.

Report:

- Sample result JSON path.
- Aggregation output.
- Video paths, if any.
- Problems and reproduction command.

### P2-TODO 6: Data Collection Smoke Test

Code Thread: no unless a code failure is found.
Run/Eval Thread: yes, full env required.

Goal:
Verify `scripts/collect_demo.py` can collect one local demo episode.

Prerequisite checks:

```bash
python -c "import curobo; print('CuRobo OK')"
python -c "from isaaclab.app import AppLauncher; print('IsaacLab OK')"
python -c "import lerobot; print('LeRobot OK')"
```

Door smoke command:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/collect_demo.py \
  --object door \
  --asset_id 99660039960014l \
  --scene_key 3a \
  --dataset_name local/insightbench-smoke-door \
  --num_envs 1 \
  --target_episodes 1 \
  --max_loops 5 \
  --headless --enable_cameras
```

If door succeeds, try:

- cabinet: `asset_id 46130`, valid scene key from current config/code.
- bottle: `asset_id 14b`, `scene_key 5a`.

Rules:

- Do not push to hub.
- Start with `save_video` off.
- Use local dataset only.

Report:

- Command.
- Success/failure.
- Dataset path.
- Frame schema status.
- Failure stage:
  - CuRobo init
  - planner
  - env/reward
  - dataset write
  - resume

### P2-TODO 7: Failure Taxonomy

Code Thread: no.
Headquarters: yes.

Goal:
Classify runtime failures into actionable buckets.

Categories:

- install/env issue
- missing asset
- missing checkpoint
- stats shape mismatch
- obs key mismatch
- Isaac Sim API mismatch
- CuRobo planning failure
- reward/success detection issue
- video codec issue
- dataset write issue

Completion:

- List Code Thread tasks.
- List environment/setup tasks.
- List blocker vs non-blocker failures.

## Phase 3: Release And Public Repo Quality

Phase 3 turns a working internal benchmark into a reliable public artifact.

### P3-TODO 1: Clean-Room README Reproduction

Code Thread: maybe for docs fixes.
Run/Eval Thread: yes for validation.

Goal:
Verify a user can follow README from clone to smoke eval.

Flow:

1. Clone or fresh checkout.
2. Run `install_eval.sh`.
3. Download assets.
4. Log into HuggingFace if needed.
5. Run one eval smoke.
6. Aggregate results.

Check:

- Commands match actual file names.
- Env var overrides are documented.
- Asset download patterns match HF structure.
- Checkpoint overrides are clear.
- Optional OpenAI API requirements are clear.

Completion:

- README reproduces a minimum door eval.
- Any blocked step is documented and converted to a TODO.

### P3-TODO 2: Public Release Metadata And Path Cleanup

Code Thread: yes.

Goal:
Remove private paths, placeholders, and inconsistent metadata before release.

Search:

```bash
rg -n "/home/seonho|/home/bluepot|/pkgs|YOUR_ORG|YOUR_HF_USERNAME|OPENAI_API_KEY|sk-" .
```

Check:

- README.
- configs.
- install scripts.
- docs.
- pyproject metadata.
- license.
- citation.
- dataset/model/asset attribution.

Rules:

- Examples may keep placeholder values if clearly marked.
- Defaults must not contain private paths.
- Do not remove attribution/license information.

Completion:

- Remaining private/path hits are intentional examples or gone.
- README, LICENSE, pyproject are consistent.

### P3-TODO 3: Top-Level Module Integration Decision

Code Thread: no initially.
Headquarters: yes.

Goal:
Decide long-term module layout.

Options:

- A: Keep top-level `cfg`, `custom_lab`, `interact`, etc. and package them.
- B: Move them under `insightbench/`.
- C: Split public evaluation package and internal data-collection package.

Recommendation for v0.1:

- Keep option A after P1 packaging fixes.
- Defer namespace migration to a separate v2 refactor.

Completion:

- Decision documented.
- Any future namespace migration is a separate large TODO.

### P3-TODO 4: Legacy Feature Include/Exclude Inventory

Code Thread: maybe after analysis.
Headquarters: yes.

Goal:
Decide which original `InsightManip` functionality belongs in public repo.

Original areas to classify:

- old evaluation scripts
- `evaluation_dp/`
- `experiment/`
- training scripts
- `model/`
- visualization utilities
- previous backups
- asset conversion tools
- dataset utility scripts

Decision criteria:

- Required for benchmark eval?
- Required for data collection?
- Required for paper reproduction?
- Internal experiment only?
- Uses old Isaac Lab API?
- Public license OK?

Completion:

- Include/exclude inventory.
- Missing but required files become Code Thread TODOs.
- Excluded internal functions are mentioned as not included if needed.

### P3-TODO 5: Lightweight Validation Script Or CI

Code Thread: yes.

Goal:
Provide a local/CI validation path that does not require Isaac Sim.

Candidate command:

```bash
python3 -m compileall -q insightbench cfg custom_lab interact demo_gen scripts
python3 -m pytest tests
python3 scripts/generate_eval_jobs.py --count
rg -n "/home/seonho|/home/bluepot|/pkgs/IsaacLabCurobo" .
```

Implementation options:

- `scripts/validate_repo.py`
- `Makefile`
- lightweight GitHub Actions workflow

Rules:

- No Isaac Sim in CI.
- No GPU tests in CI.
- No network access required.
- Runtime eval remains manual.

Completion:

- One command validates syntax, tests, YAML/job count, and forbidden paths.
- README documents how to run it.

### P3-TODO 6: Benchmark Table Generation

Code Thread: yes if current aggregation is insufficient.

Goal:
Generate README/paper result tables from JSON results.

Files:

- `scripts/aggregate_results.py`
- Optional new `scripts/make_table.py`
- `README.md`

Features:

- Per-object success rate.
- Overall success rate.
- Policy-level rows.
- Missing jobs count.
- Failed jobs count.
- Optional seen/unseen split.
- CSV and Markdown export.

Completion:

- `outputs/results/<policy>` can produce a README-ready table row.
- Missing jobs are reported, not silently ignored.

### P3-TODO 7: Split Detailed Docs From README

Code Thread: yes.

Goal:
Keep README quick-start focused and move details into docs.

Suggested docs:

- `docs/assets.md`
- `docs/checkpoints.md`
- `docs/evaluation.md`
- `docs/data_collection.md`
- `docs/troubleshooting.md`

Content:

- HF Assets structure.
- TestSuite vs TrainSuite.
- Policy checkpoint requirements.
- Dataset stats requirements.
- Data collection schema.
- Common errors and fixes.

Completion:

- README is shorter and links to docs.
- Old path references are corrected.
- Existing details are preserved or intentionally removed.

### P3-TODO 8: Security, Privacy, And Artifact Scan

Code Thread: no unless cleanup is approved.
Headquarters/Run-Eval: yes.

Goal:
Confirm no private data or large artifacts are published.

Scan for:

- API keys.
- HF tokens.
- OpenAI keys.
- wandb tokens.
- private paths.
- logs with server/user info.
- mp4 outputs.
- checkpoints.
- dataset images.
- generated cache.

Completion:

- Report artifacts to remove.
- Remove only after explicit approval.
- Public release contains no private artifacts.

### P3-TODO 9: Release Candidate Review

Code Thread: no.
Headquarters: yes.

Goal:
Decide whether repo is ready for v0.1/public release.

Checklist:

- Phase 1 complete.
- Eval smoke passes.
- Collect smoke passes or known issue documented.
- README clean-room test passes.
- Lightweight validation passes.
- Asset/checkpoint docs clear.
- Aggregation works.
- Git state normal.
- No private artifacts.

Completion:

- Release blockers: zero, or explicitly waived.
- Known issues documented.
- Tag/release can proceed.

## Phase 4: Latest LeRobot Migration

Phase 4 is separate from the Phase 1-3 public refactor. It covers the decision
to support recent upstream LeRobot releases while preserving IsaacLab runtime
compatibility.

Current finding:

- Current IsaacLab runtime is constrained to Python 3.11 and NumPy 1.x.
- Latest LeRobot 0.6.x requires Python 3.12 and NumPy 2.x.
- A single environment is therefore not a safe default target.
- GR00T is blocked in the current env because the installed LeRobot 0.3.4 fork
  does not include `lerobot.policies.groot`.

Reference status document:

- `docs/refactoring_status.md`

### P4-TODO 1: Latest LeRobot Environment Spike

Env Thread: yes.
Code Thread: no.
Run/Eval Thread: no.

Goal:
Verify whether a latest LeRobot 0.6.x environment can import policy classes and
run policy-only inference.

Checks:

- Python 3.12 env creation.
- LeRobot 0.6.x install with policy extras.
- Imports:
  - `PI0Policy`
  - `SmolVLAPolicy`
  - `DiffusionPolicy`
  - `GrootPolicy`
  - `PolicyProcessorPipeline`
  - `LeRobotDataset`

Completion:

- Import smoke report is pasted back to Headquarters.
- Dependency conflicts are listed explicitly.

### P4-TODO 2: Policy Server Architecture Decision

Headquarters: yes.
Code Thread: no.

Goal:
Decide whether latest LeRobot support uses a separate policy process instead of
sharing the IsaacLab simulation environment.

Decision inputs:

- P4-TODO 1 results.
- IsaacLab Python/NumPy constraints.
- Latency and serialization risk for image observations.

Completion:

- One architecture is selected:
  - legacy single-env only
  - separate latest-LeRobot policy server
  - unsupported latest-LeRobot path

### P4-TODO 3: Observation/Action IPC Schema

Code Thread: yes, only after P4-TODO 2.

Goal:
Define a stable interface between IsaacLab simulation and an external policy
process.

Required fields:

- `observation.state`
- camera images
- task prompt
- reset/episode metadata
- returned action tensor

Completion:

- Schema documented.
- One dummy round-trip smoke test exists.

### P4-TODO 4: Policy Wrapper Migration

Code Thread: yes, only after P4-TODO 2.
Run/Eval Thread: yes for smoke.

Goal:
Support latest LeRobot policy loading without breaking the legacy runtime path.

Policies:

- Pi0 first.
- SmolVLA second.
- Diffusion third.
- GR00T last.

Completion:

- Door smoke passes for each migrated policy, or failure is classified.

### P4-TODO 5: Dataset Writer Migration Decision

Headquarters: yes.
Code Thread: maybe.

Goal:
Decide whether data collection should remain on the IsaacLab-compatible LeRobot
fork or migrate to LeRobot v3 writer APIs.

Known required changes for v3 writer:

- Remove `create_episode_buffer` public usage.
- Remove `_get_image_file_path` private API usage.
- Use `add_frame({...,"task": task})`.
- Use no-arg `save_episode()`.
- Call `finalize()` before upload/exit.

Completion:

- Dataset strategy is documented before code changes begin.

## Current GitHub Upload Blocker

`/home/seonho/workspace/InsightBench` currently fails:

```bash
git status --short --branch
```

with:

```text
fatal: not a git repository (or any of the parent directories): .git
```

Before pushing to GitHub:

1. Restore or initialize a valid git repository.
2. Add/confirm the GitHub remote.
3. Stage only intended refactor/docs changes.
4. Commit and push to a branch.
5. Open a draft PR.

## Recommended Execution Order

1. Finish Phase 1 TODOs.
2. Run P2-TODO 1, 3, 4, and 6 in a Run/Eval Thread.
3. Use P2 failure reports to create targeted Code Thread tasks.
4. Run P2-TODO 5 after eval smoke can save outputs.
5. Do P2-TODO 7 in Headquarters.
6. Start Phase 3 docs/release work after runtime smoke tests pass.
7. Start Phase 4 only after deciding whether latest LeRobot support is a release
   requirement for v0.1.

## Run/Eval Final Report Template

```text
1. Environment summary
- python path/version:
- conda env:
- IsaacLab import:
- CuRobo import:
- LeRobot import:
- GPU/CUDA:

2. P2-TODO 1 Eval Smoke
- commands:
- success/failure:
- failure stage:
- stack trace summary:

3. P2-TODO 3 Asset Integrity
- checked asset/task count:
- missing paths:
- code path issue vs missing download:

4. P2-TODO 4 Batch Dry Run
- expected jobs:
- actual jobs:
- object counts:
- mismatch:

5. P2-TODO 6 Collect Demo Smoke
- full env checks:
- command:
- success/failure:
- dataset path:
- failure stage:

6. Code Thread issues
- title:
- reproduction command:
- related files:
- key error:
- recommended fix:

7. Headquarters decisions needed
- checkpoints:
- assets:
- env setup:
- private/public access:
```
