from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = REPO_ROOT / "scripts" / "evaluate.py"
PERSISTENT = REPO_ROOT / "scripts" / "evaluate_persistent.py"
PERSISTENT_BATCH = REPO_ROOT / "scripts" / "eval_batch_persistent.sh"
PERSISTENT_BATCH_PY = REPO_ROOT / "scripts" / "eval_batch_persistent.py"
BUILDER = REPO_ROOT / "insightbench" / "envs" / "builder.py"


def _function_dump(path: Path, function_name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"{function_name} not found in {path}")


def test_persistent_episode_logic_matches_original_evaluator() -> None:
    for function_name in (
        "_get_task_prompt",
        "_resolve_policy_inference_options",
        "_noop_warmup",
        "run_episode",
    ):
        assert _function_dump(PERSISTENT, function_name) == _function_dump(ORIGINAL, function_name)


def test_persistent_path_is_additive_and_retries_with_original_evaluator() -> None:
    wrapper_source = PERSISTENT_BATCH.read_text(encoding="utf-8")
    batch_source = PERSISTENT_BATCH_PY.read_text(encoding="utf-8")
    persistent_source = PERSISTENT.read_text(encoding="utf-8")

    assert "eval_batch_persistent.py" in wrapper_source
    assert "scripts/evaluate.py" in batch_source
    assert "evaluate_persistent.py" in batch_source
    assert "--objects" in batch_source
    assert persistent_source.count("load_policy(policy_cfg, policy_device)") == 1
    assert "Reusing loaded policy" in persistent_source
    assert "stage_utils.create_new_stage()" in persistent_source
    assert "simulation_context.stop" in persistent_source
    assert "policy_ref[0] = policy" in persistent_source


def test_persistent_batch_supports_deployment_controls() -> None:
    source = PERSISTENT_BATCH_PY.read_text(encoding="utf-8")

    for option in ("--categories", "--splits", "--assets", "--resume", "--dry-run"):
        assert option in source
    for artifact in ("run_manifest.json", "jobs.tsv", "retry_failed.sh"):
        assert artifact in source


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return ast.get_source_segment(source, node) or ""
    raise AssertionError(f"{function_name} not found in {path}")


def test_reward_is_checked_after_every_policy_action() -> None:
    for path in (ORIGINAL, PERSISTENT):
        source = _function_source(path, "run_episode")
        assert source.count("step_rew =") == 1
        assert source.index("env.step(action)") < source.index("step_rew =")


def test_eval_reset_defaults_are_deterministic_and_symmetric() -> None:
    source = _function_source(BUILDER, "configure_eval_default_joint_state")
    assert "default_joint_pos[:, 7:9] = 0.04" in source
    assert "default_joint_vel[:, 7:9] = 0.0" in source
    for path in (ORIGINAL, PERSISTENT):
        script_source = path.read_text(encoding="utf-8")
        assert "configure_eval_default_joint_state(" in script_source
        assert "default_joint_vel[:, 8] = 0.04" not in script_source


def test_asset_sampling_is_seeded_before_asset_info_is_built() -> None:
    source = _function_source(BUILDER, "build_env")
    info_call = source.index("info = get_info_test")
    assert source.index("random.seed(seed)") < info_call
    assert source.index("np.random.seed(seed)") < info_call
