"""CLI coverage for the portable video-frame counting utility."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "dataset" / "calculate_video_frame.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("calculate_video_frame_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_video_frame_cli_requires_a_portable_root_path(monkeypatch, tmp_path) -> None:
    module = _load_module()

    monkeypatch.setattr(sys, "argv", [str(_SCRIPT)])
    with pytest.raises(SystemExit):
        module.parse_args()

    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), str(tmp_path)])

    args = module.parse_args()

    assert args.root_path == str(tmp_path)
