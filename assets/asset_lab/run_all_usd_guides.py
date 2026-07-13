# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""모든 USD를 격자로 배치해 시각화하는 스크립트.

사용법 (예):

    ./isaaclab.sh -p source/project/asset_lab/run_all_usd_guides.py

"""

import os
import argparse
import math
from typing import List, Tuple

from isaaclab.app import AppLauncher

# -------------------------------------------------
# CLI
# -------------------------------------------------
parser = argparse.ArgumentParser(description="Scan and visualize all USD files under a directory.")
parser.add_argument(
    "--base_path",
    type=str,
    default="Assets/guides",
    help="USD 파일들을 탐색할 루트 디렉토리",
)
# Isaac Lab의 AppLauncher 기본 인자 추가
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -------------------------------------------------
# Imports after simulator
# -------------------------------------------------
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
import isaacsim.core.utils.prims as prim_utils


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def list_usd_files(root_dir: str) -> List[str]:
    """루트 디렉토리 이하에서 모든 USD 파일(.usd/.usda/.usdc)을 재귀적으로 수집."""
    usd_exts = {".usd", ".usda", ".usdc"}
    results: List[str] = []
    if not os.path.isdir(root_dir):
        print(f"[WARN] Base path not found: {root_dir}")
        return results
    for cur_root, _dirs, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in usd_exts:
                results.append(os.path.join(cur_root, f))
    results.sort()
    return results


def compute_grid_origins(num_items: int, spacing: float = 2.0, start_xy: Tuple[float, float] = (-2.0, -2.0), z_height: float = 0.0) -> List[List[float]]:
    """개수에 맞춰 정사각 격자 배치를 위한 월드 원점 목록을 생성."""
    if num_items <= 0:
        return []
    grid_size = int(math.ceil(math.sqrt(num_items)))
    origins: List[List[float]] = []
    sx, sy = start_xy
    for idx in range(num_items):
        row = idx // grid_size
        col = idx % grid_size
        x = sx + row * spacing
        y = sy + col * spacing
        origins.append([x, y, z_height])
    return origins


def make_safe_name(base: str, rel_path: str) -> str:
    """상대 경로를 Prim 이름으로 안전하게 변환."""
    name = rel_path.replace(os.sep, "_")
    # Prim 이름에 사용할 수 있도록 영숫자/언더스코어 이외는 언더스코어로 대체
    safe = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            safe.append(ch)
        else:
            safe.append("_")
    return f"{base}_{''.join(safe)}"


# -------------------------------------------------
# Scene setup
# -------------------------------------------------

def design_scene(base_path: str) -> Tuple[list, list]:
    """USD 파일들을 찾아 격자 배치로 월드에 소환."""
    # 바닥/라이트 생성
    gp_cfg = sim_utils.GroundPlaneCfg()
    gp_cfg.func("/World/defaultGroundPlane", gp_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    usd_paths = list_usd_files(base_path)
    if len(usd_paths) == 0:
        print(f"[INFO] No USD files found under: {base_path}")
        return [], []

    origins = compute_grid_origins(len(usd_paths), spacing=2.5, start_xy=(-5.0, -5.0), z_height=0.0)

    spawned_prim_paths: list = []

    for i, usd_path in enumerate(usd_paths):
        origin = origins[i]
        rel = os.path.relpath(usd_path, base_path)
        prim_group = make_safe_name("Origin", rel)
        xform_path = f"/World/{prim_group}"
        prim_utils.create_prim(xform_path, "Xform", translation=origin)

        # USD 스폰
        usd_cfg = sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            scale=(1.0, 1.0, 1.0),
        )
        asset_prim_path = f"{xform_path}/Asset"
        usd_cfg.func(asset_prim_path, usd_cfg)
        spawned_prim_paths.append(asset_prim_path)

    return spawned_prim_paths, origins


# -------------------------------------------------
# Main sim loop
# -------------------------------------------------

def run_simulator(sim: SimulationContext):
    """단순 스텝 진행만 수행 (시각화 목적)."""
    frame = 0
    while simulation_app.is_running():
        sim.step()
        frame += 1


# -------------------------------------------------
# Entry point
# -------------------------------------------------

def main():
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)

    spawned, origins = design_scene(args_cli.base_path)

    # 카메라 뷰 설정: 모든 격자가 보이도록 약간 높은 위치에서 바라보게 함
    sim.set_camera_view((10.0, 10.0, 12.0), (0.0, 0.0, 0.0))

    sim.reset()
    print(f"[INFO] Spawned {len(spawned)} USD assets from: {args_cli.base_path}")
    run_simulator(sim)


if __name__ == "__main__":
    main()
    simulation_app.close() 