"""Launch Isaac Sim Simulator first."""
import random
import argparse
import json
import numpy as np
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext
from isaacsim.util.debug_draw import _debug_draw

"""Standalone script: loads bottle USDs, computes and visualizes bounding boxes per environment origin."""
import os
import random
import argparse
import json
import numpy as np
import torch

# debug draw interface
_draw = _debug_draw.acquire_debug_draw_interface()

# ----------------- bbox helpers -----------------
def draw_point_cross(world_pt, size=0.02, color=(1.0, 1.0, 0.0, 1.0), thickness=2.0):
    """작은 십자(+-)로 포인트를 표시."""
    x, y, z = world_pt
    p0_list = [
        [x - size, y, z], [x, y - size, z], [x, y, z - size],
    ]
    p1_list = [
        [x + size, y, z], [x, y + size, z], [x, y, z + size],
    ]
    color_list = [color, color, color]
    thickness_list = [thickness, thickness, thickness]
    _draw.draw_lines(p0_list, p1_list, color_list, thickness_list)

def compute_aabb_corners(min_xyz, max_xyz):
    mn = np.array(min_xyz, dtype=np.float32)
    mx = np.array(max_xyz, dtype=np.float32)
    return np.array([
        [mn[0], mn[1], mn[2]],
        [mn[0], mn[1], mx[2]],
        [mn[0], mx[1], mn[2]],
        [mn[0], mx[1], mx[2]],
        [mx[0], mn[1], mn[2]],
        [mx[0], mn[1], mx[2]],
        [mx[0], mx[1], mn[2]],
        [mx[0], mx[1], mx[2]],
    ], dtype=np.float32)


def draw_aabb_batch(min_xyz, max_xyz, offset, color=(1.0, 0.0, 0.0, 1.0), thickness=2.0):
    """Draw one axis-aligned bounding box (expanded in Z) using draw_lines."""
    corners = compute_aabb_corners(min_xyz, max_xyz)  # (8,3)
    world_pts = corners + np.array(offset, dtype=np.float32)
    edges = [
        (0,1),(0,2),(0,4),
        (1,3),(1,5),
        (2,3),(2,6),
        (3,7),
        (4,5),(4,6),
        (5,7),
        (6,7),
    ]
    p0_list, p1_list, color_list, thickness_list = [], [], [], []
    for a, b in edges:
        p0_list.append(world_pts[a].tolist())
        p1_list.append(world_pts[b].tolist())
        color_list.append(color)
        thickness_list.append(thickness)
    _draw.draw_lines(p0_list, p1_list, color_list, thickness_list)

def visualize_bboxes_from_mobility_paths(mobility_paths: list[str], origins: np.ndarray,
                                         scale: float = 1.0, z_extend: float = 0.0):
    """각 USD 폴더의 bounding_box.json을 읽어 handle/lock 박스와 goal_pos를 그림.
    - *_min/_max는 AABB로 해석하여 draw
    - goal_pos는 십자로 포인트 표시
    - origin 보정 및 scale 적용
    """
    for i, usd_path in enumerate(mobility_paths):
        folder = os.path.dirname(usd_path)
        bbox_json = os.path.join(folder, "handle_bounding.json")
        if not os.path.isfile(bbox_json):
            continue

        try:
            with open(bbox_json, "r") as f:
                bb = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {bbox_json}: {e}")
            continue

        origin = np.array(origins[i], dtype=np.float32)

        def _scaled(v):
            arr = np.array(v, dtype=np.float32) * scale
            if z_extend != 0.0 and arr.shape[-1] == 3:
                arr = arr.copy()
                arr[2] += z_extend
            return arr

        # --- handle bbox (red)
        if "handle_min" in bb and "handle_max" in bb:
            hmin = _scaled(bb["handle_min"])
            hmax = _scaled(bb["handle_max"])
            draw_aabb_batch(hmin.tolist(), hmax.tolist(), origin, color=(1.0, 0.2, 0.2, 1.0), thickness=2.0)

        # --- lock bbox (green)
        if "lock_min" in bb and "lock_max" in bb:
            lmin = _scaled(bb["lock_min"])
            lmax = _scaled(bb["lock_max"])
            draw_aabb_batch(lmin.tolist(), lmax.tolist(), origin, color=(0.2, 1.0, 0.2, 1.0), thickness=2.0)

        # --- (옵션) cover/body가 있는 경우도 그대로 지원
        if "cover_min" in bb and "cover_max" in bb:
            cmin = _scaled(bb["cover_min"])
            cmax = _scaled(bb["cover_max"])
            draw_aabb_batch(cmin.tolist(), cmax.tolist(), origin, color=(1.0, 0.0, 0.0, 1.0), thickness=1.5)

        if "body_min" in bb and "body_max" in bb:
            bmin = _scaled(bb["body_min"])
            bmax = _scaled(bb["body_max"])
            draw_aabb_batch(bmin.tolist(), bmax.tolist(), origin, color=(0.0, 1.0, 0.0, 1.0), thickness=1.5)

        # --- goal_pos (yellow cross)
        if "goal_pos" in bb:
            gp_local = _scaled(bb["goal_pos"])
            gp_world = (gp_local + origin).tolist()
            draw_point_cross(gp_world, size=0.03, color=(1.0, 1.0, 0.0, 1.0), thickness=2.0)

# -------------- scene construction ----------------

def grep_usd_paths():
    base_path = "./Assets/AdaManip/door"
    # base_path = "./Assets/TestSuite/door_suite"
    # base_path = "./Assets/UniDoorManip/LeverDoor"
    paths = []  # list of (usd_path, folder_name)
    folders = []
    for folder in sorted(os.listdir(base_path)):
        d = os.path.join(base_path, folder)
        usd = os.path.join(d, "mobility_push_cw.usd")
        if os.path.isfile(usd):
            paths.append(usd)
            folders.append(folder)
    return paths, folders


def env_spacing(mobility_paths):
    spacing = 2.0
    n = len(mobility_paths)
    max_per_row = int(n**0.5) if n > 0 else 1
    origins = []
    for i in range(n):
        x = -2 + (i // max_per_row) * spacing
        y = -2 + (i % max_per_row) * spacing
        origins.append([x, y, 1.0])
    return origins


def design_scene():
    # ground and lighting
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75,0.75,0.75))
    cfg.func("/World/Light", cfg)

    mobility_paths, folders = grep_usd_paths()
    origins = env_spacing(mobility_paths)
    door_cfgs = []
    for i, (mobility_path, origin, folder) in enumerate(zip(mobility_paths, origins, folders)):
        prim_utils.create_prim(f"/World/Origin{folder}", "Xform", translation=origin)
        door_cfg = ArticulationCfg(
            prim_path=f"/World/Origin{folder}/Door",
            spawn=sim_utils.UsdFileCfg(
                usd_path=mobility_path,
                activate_contact_sensors=True,
                scale=(1.0,1.0,1.0),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0,0.0,0.5),
                rot=(0.707,0.0,0.707,0.0),
                joint_pos={".*": 0.0},
            ),
            actuators={
                "door": ImplicitActuatorCfg(
                    joint_names_expr=["joint_1"],
                    effort_limit=87.0,
                    velocity_limit=10.0,
                    stiffness=0.0,
                    damping=0.0,
                    friction=0.5,
                ),
                "lever": ImplicitActuatorCfg(
                    joint_names_expr=["joint_2"],
                    effort_limit=87.0,
                    velocity_limit=10.0,
                    stiffness=0.0,
                    damping=0.0,
                    friction=0.0001,
                )
            }
        )
        door_cfgs.append(door_cfg)
    scene_entities = {}
    for i, cfg in enumerate(door_cfgs):
        art = Articulation(cfg=cfg)
        scene_entities[f"door{i+1}"] = art
    return scene_entities, origins, mobility_paths

# -------------- simulation loop ----------------

def run_simulator(sim: sim_utils.SimulationContext, entities: dict, origins: torch.Tensor, mobility_paths: list[str]):
    sim_dt = sim.get_physics_dt()
    origins_np = origins.cpu().numpy()
    # visualize per-USD bounding boxes with z expansion
    # visualize_bboxes_from_mobility_paths(mobility_paths, origins_np, scale=1.0, z_extend=0.5)

    count = 0
    # pick the first existing bottle as placeholder
    robot_key = list(entities.keys())[0]
    robot = entities[robot_key]
    joint_targets = robot.data.default_joint_pos.clone()
    joint_direction = 1
    while simulation_app.is_running():
        if count % 500 == 0:
            count = 0
            joint_direction = 1
        sim.step()
        count += 1

# -------------- entry point ----------------
def main():
    sim_cfg = sim_utils.SimulationCfg(device="cpu")
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))
    scene_entities, scene_origins, mobility_paths = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    sim.reset()
    print("[INFO]: Setup complete...")
    run_simulator(sim, scene_entities, scene_origins, mobility_paths)

if __name__ == "__main__":
    main()
    simulation_app.close()
