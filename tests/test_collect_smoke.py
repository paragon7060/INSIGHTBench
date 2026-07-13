from __future__ import annotations

import ast
from pathlib import Path

import pytest

from insightbench.utils.collect_smoke import SMOKE_MAX_DECIMATION, resolve_collect_timing


_ENV_SOURCE = Path(__file__).resolve().parents[1] / "custom_lab/envs/manager_based_rl_step_env.py"


def _deadline_stages_in_step(class_name: str) -> list[str]:
    """Read calls from the concrete AST without importing Isaac Sim modules."""
    tree = ast.parse(_ENV_SOURCE.read_text())
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    step_node = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "step"
    )
    stages = []
    for node in ast.walk(step_node):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "_check_collect_step_deadline" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            stages.append(node.args[0].value)
    return stages


def test_production_collect_timing_preserves_requested_decimation() -> None:
    timing = resolve_collect_timing(
        collect_decimation=300,
        collect_render_interval=0,
        smoke_action_steps=0,
        smoke_decimation=10,
        smoke_step_timeout_s=60,
    )

    assert timing.decimation == 300
    assert timing.render_interval == 300
    assert not timing.smoke_enabled
    assert timing.smoke_step_timeout_s is None


def test_smoke_timing_uses_low_decimation_and_one_render_per_action() -> None:
    timing = resolve_collect_timing(
        collect_decimation=300,
        collect_render_interval=300,
        smoke_action_steps=2,
        smoke_decimation=10,
        smoke_step_timeout_s=45,
    )

    assert timing.smoke_enabled
    assert timing.decimation == 10
    assert timing.render_interval == 10
    assert timing.smoke_action_steps == 2
    assert timing.smoke_step_timeout_s == 45


@pytest.mark.parametrize(
    ("smoke_decimation", "smoke_step_timeout_s", "match"),
    [
        (0, 60, "smoke_decimation"),
        (SMOKE_MAX_DECIMATION + 1, 60, "smoke_decimation"),
        (10, 0, "smoke_step_timeout_s"),
    ],
)
def test_smoke_timing_rejects_unbounded_or_invalid_settings(
    smoke_decimation: int,
    smoke_step_timeout_s: float,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        resolve_collect_timing(
            collect_decimation=300,
            collect_render_interval=0,
            smoke_action_steps=1,
            smoke_decimation=smoke_decimation,
            smoke_step_timeout_s=smoke_step_timeout_s,
        )


def test_both_env_step_implementations_check_collect_deadline() -> None:
    expected_stages = {
        "action processing",
        "physics stepping",
        "simulation",
        "rendering",
        "scene update",
        "post-physics checks",
        "termination computation",
        "reward computation",
        "environment reset",
        "command update",
        "interval events",
        "observation computation",
    }

    for class_name in ("ManagerBasedRLStepEnv", "ManagerBasedContinuousEnv"):
        stages = _deadline_stages_in_step(class_name)
        assert expected_stages.issubset(stages), class_name
