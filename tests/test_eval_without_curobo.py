"""Regression coverage for the evaluation-only CuRobo boundary."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ACTION_BATCH_SOURCE = _REPO_ROOT / "interact/action_batch.py"
_EVAL_CONFIG_SOURCES = (
    _REPO_ROOT / "cfg/BaseTaskCfg.py",
    _REPO_ROOT / "cfg/BaseCfg.py",
    _REPO_ROOT / "cfg/scene1Cfg.py",
    _REPO_ROOT / "cfg/scene3Cfg.py",
    _REPO_ROOT / "cfg/scene5Cfg.py",
)


def _has_direct_curobo_import(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "curobo" or alias.name.startswith("curobo.") for alias in node.names):
                return True
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "curobo" or node.module.startswith("curobo."):
                return True
    return False


def test_evaluation_config_defers_curobo_action_import() -> None:
    """Config sources only reference the lazily importable action class."""
    assert not any(_has_direct_curobo_import(path) for path in _EVAL_CONFIG_SOURCES)

    # Import the real module while stubbing only its Isaac-side base classes.
    # Fully importing BaseTaskCfg requires a launched Isaac Sim app because Omni
    # physics extensions are unavailable to a standalone Python interpreter.
    script = f"""
import builtins
import importlib
import importlib.util
import sys
import types

real_import = builtins.__import__

def reject_curobo(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "curobo" or name.startswith("curobo."):
        raise ModuleNotFoundError("CuRobo is intentionally unavailable in this test", name="curobo")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = reject_curobo

def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module

isaaclab = package("isaaclab")
isaaclab_utils = package("isaaclab.utils")
isaaclab_utils_math = types.ModuleType("isaaclab.utils.math")
isaaclab_assets = package("isaaclab.assets")
isaaclab_articulation = types.ModuleType("isaaclab.assets.articulation")
isaaclab_utils.math = isaaclab_utils_math
isaaclab.assets = isaaclab_assets
isaaclab_assets.articulation = isaaclab_articulation
isaaclab_articulation.Articulation = object
sys.modules["isaaclab.utils.math"] = isaaclab_utils_math
sys.modules["isaaclab.assets.articulation"] = isaaclab_articulation

custom_lab = package("custom_lab")
custom_lab_managers = package("custom_lab.managers")
action_counter_manager = types.ModuleType("custom_lab.managers.action_counter_manager")
custom_lab.managers = custom_lab_managers
custom_lab_managers.action_counter_manager = action_counter_manager
action_counter_manager.ActionCounterTerm = type("ActionCounterTerm", (), {{}})
sys.modules["custom_lab.managers.action_counter_manager"] = action_counter_manager
sys.modules["carb"] = types.ModuleType("carb")

spec = importlib.util.spec_from_file_location("action_batch_under_test", {str(_ACTION_BATCH_SOURCE)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert module._CUROBO_LOADED is False
try:
    module._require_curobo()
except ModuleNotFoundError as exc:
    assert "requires CuRobo" in str(exc)
else:
    raise AssertionError("CuRobo import should be deferred until collection action construction")
assert "curobo" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
