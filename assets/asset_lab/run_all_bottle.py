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
import os
import numpy as np

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils

from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim import SimulationContext
from isaacsim.util.debug_draw import _debug_draw

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils

# debug draw interface
_draw = _debug_draw.acquire_debug_draw_interface()

# ----------------- bbox helpers -----------------

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


def visualize_bboxes_from_mobility_paths(mobility_paths: list[str], origins: np.ndarray, scale: float = 1.0, z_extend: float = 0.5):
    """For each USD path and corresponding origin, load its bounding_box.json and draw cover/body with symmetric z expansion."""
    for i, usd_path in enumerate(mobility_paths):
        folder = os.path.dirname(usd_path)
        bbox_json = os.path.join(folder, "bounding_box.json")
        if not os.path.isfile(bbox_json):
            continue
        with open(bbox_json, 'r') as f:
            bb = json.load(f)
        origin = origins[i]
        # cover: red, expand z both directions
        cover_min = np.array(bb["cover_min"], dtype=np.float32) * scale
        cover_max = np.array(bb["cover_max"], dtype=np.float32) * scale
        cover_min[2] += z_extend
        cover_max[2] += z_extend
        draw_aabb_batch(cover_min.tolist(), cover_max.tolist(), origin, color=(1,0,0,1))
        # body: green, symmetric z expansion
        body_min = np.array(bb["body_min"], dtype=np.float32) * scale
        body_max = np.array(bb["body_max"], dtype=np.float32) * scale
        body_min[2] += z_extend
        body_max[2] += z_extend
        draw_aabb_batch(body_min.tolist(), body_max.tolist(), origin, color=(0,1,0,1))

# -------------- scene construction ----------------

def grep_usd_paths():
    base_path = "./Assets/AdaManip/bottle"
    paths = []  # list of (usd_path, folder_name)
    folders = []
    for folder in sorted(os.listdir(base_path)):
        d = os.path.join(base_path, folder)
        usd = os.path.join(d, "mobility.usd")
        if os.path.isfile(usd):
            paths.append(usd)
            folders.append(folder)
    return paths, folders


def env_spacing(mobility_paths):
    spacing = 1.0
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
    bottle_cfgs = []
    for i, (mobility_path, origin, folder) in enumerate(zip(mobility_paths, origins, folders)):
        prim_utils.create_prim(f"/World/Origin{folder}", "Xform", translation=origin)
        bottle_cfg = ArticulationCfg(
            prim_path=f"/World/Origin{folder}/Bottle",
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
                "prismatic": ImplicitActuatorCfg(
                    joint_names_expr=["joint_1"],
                    effort_limit=87.0,
                    velocity_limit=1.0,
                    stiffness=1200.0,
                    damping=0.0,
                    friction=1.0,
                ),
                "cap_revolute": ImplicitActuatorCfg(
                    joint_names_expr=["joint_2"],
                    effort_limit=87.0,
                    velocity_limit=1.0,
                    stiffness=0.0,
                    damping=0.0,
                    friction=1.0,
                ),
            },
        )
        bottle_cfgs.append(bottle_cfg)
    scene_entities = {}
    for i, cfg in enumerate(bottle_cfgs):
        art = Articulation(cfg=cfg)
        scene_entities[f"bottle{i+1}"] = art
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
