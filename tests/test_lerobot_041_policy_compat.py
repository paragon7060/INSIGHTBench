"""Static contracts for the pinned LeRobot 0.4.1 policy runtime."""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLICY_CONSTANTS = {
    "pi0.py": {"ACTION", "OBS_STATE"},
    "smolvla.py": {"OBS_STATE"},
    "diffusion.py": {"OBS_STATE"},
    "instruction_gpt.py": {"OBS_STATE"},
}
_TRANSFORMERS_PIN = "transformers==4.57.1"


def _imports_from(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == module
        for alias in node.names
    }


def test_policy_adapters_use_lerobot_041_constants_module() -> None:
    policy_root = _REPO_ROOT / "insightbench/policies"
    for filename, expected_names in _POLICY_CONSTANTS.items():
        source = policy_root / filename
        assert expected_names.issubset(_imports_from(source, "lerobot.utils.constants"))
        assert not _imports_from(source, "lerobot.constants")


def test_transformers_pin_matches_validated_groot_runtime() -> None:
    requirements = (_REPO_ROOT / "requirements.txt").read_text().splitlines()
    assert _TRANSFORMERS_PIN in requirements

    project = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text())
    assert _TRANSFORMERS_PIN in project["project"]["dependencies"]

    lerobot_install = 'install -e "${LEROBOT_ROOT}[${LEROBOT_EXTRAS}]"'
    for installer_name in ("install.sh", "install_eval.sh"):
        installer = (_REPO_ROOT / installer_name).read_text()
        assert installer.index(_TRANSFORMERS_PIN) > installer.index(lerobot_install)
