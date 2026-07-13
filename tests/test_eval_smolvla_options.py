"""Regression coverage for SmolVLA evaluation option resolution."""

from __future__ import annotations

import ast
from pathlib import Path
import types

from insightbench.utils.eval_config import load_eval_config


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EVALUATE_SOURCE = _REPO_ROOT / "scripts" / "evaluate.py"
_SMOLVLA_CONFIG = _REPO_ROOT / "configs" / "eval" / "smolvla.yaml"


def _load_inference_option_resolver():
    """Execute evaluate.py's dependency-light OmegaConf helper only.

    Importing the full script launches Isaac Sim, so this keeps the regression
    focused on the pre-launch config code that previously raised NameError.
    """
    tree = ast.parse(_EVALUATE_SOURCE.read_text())
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ImportFrom) and node.module == "omegaconf"
        )
        or (
            isinstance(node, ast.FunctionDef) and node.name == "_resolve_policy_inference_options"
        )
    ]
    module = types.ModuleType("evaluate_smolvla_options_under_test")
    source = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(source, str(_EVALUATE_SOURCE), "exec"), module.__dict__)
    return module._resolve_policy_inference_options


def test_smolvla_evaluation_options_use_omegaconf_before_isaac_launch() -> None:
    cfg = load_eval_config(
        str(_SMOLVLA_CONFIG),
        [
            "policy.checkpoint=your-hf-user-or-org/smolvla-policy",
            "policy.dataset_stats_repo=your-hf-user-or-org/dataset-repo",
        ],
    )

    resolve = _load_inference_option_resolver()

    assert resolve(cfg.policy) == ("guide", True)
