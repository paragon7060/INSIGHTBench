"""Interactive bottle scene — keyboard control + EE target visualization.

Isaac Sim startup output is redirected to /tmp/isaacsim_interactive.log so
the curses TUI is never garbled.

Keyboard Controls
-----------------
  Skill:    0=Approach(IK)  1=Grasp  2=Pull  3=Rotate  4=Push/Pull(MG)  5=Gripper Open  6=Gripper Close
  Position: W/S=Y±   A/D=X±   Q/E=Z±
  Orient:   ↑↓=pitch  ←→=yaw  [/]=roll
  Gauge:    A/D (skills 2,3) — decrease / increase gauge
  Step:     +/-  scale position step size
  Execute:  ENTER or SPACE
  Reset:    R
  Quit:     ESC or Q

Env choice (edit below):  BottleSkillSqueezeCfg | BottleSkillOpenCfg | BottleSkillCloseCfg

Run:
  python scene_interact/interact_scene5_interactive.py \\
      --num_envs 1 --livestream 2 --enable_cameras
  (Isaac Sim log → /tmp/isaacsim_interactive.log)
"""

import sys, os

this_file = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(this_file)))

# ── fd helpers — save terminal fds BEFORE any redirect ───────────────────────
_LOG_PATH = '/tmp/isaacsim_interactive.log'
_TERM_OUT  = os.dup(1)
_TERM_ERR  = os.dup(2)

def _to_log():
    fd = os.open(_LOG_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.dup2(fd, 1);  os.dup2(fd, 2);  os.close(fd)

def _to_term():
    os.dup2(_TERM_OUT, 1);  os.dup2(_TERM_ERR, 2)

_to_log()

# ── Isaac Sim launch ──────────────────────────────────────────────────────────
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="squeeze",
                    choices=["squeeze", "open", "open_rev", "close", "close_rev"],
                    help="Bottle task variant")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────
from cfg.scene5Cfg import (
    BottleSkillSqueezeCfg,
    BottleSkillOpenCfg,
    BottleSkillOpenReversedCfg,
    BottleSkillCloseCfg,
    BottleSkillCloseReversedCfg,
)
from scene_interact._interactive_base import SceneCfg, InteractiveScene

# ── select task variant ────────────────────────────────────────────────────────
_CFG_MAP = {
    "squeeze"   : (BottleSkillSqueezeCfg,        "Bottle — Squeeze"),
    "open"      : (BottleSkillOpenCfg,            "Bottle — Open"),
    "open_rev"  : (BottleSkillOpenReversedCfg,    "Bottle — Open (Rev)"),
    "close"     : (BottleSkillCloseCfg,           "Bottle — Close"),
    "close_rev" : (BottleSkillCloseReversedCfg,   "Bottle — Close (Rev)"),
}

cfg_class, label = _CFG_MAP[args_cli.task]
env_cfg = cfg_class()
env_cfg.scene.num_envs = args_cli.num_envs

scene_cfg = SceneCfg(
    env_cfg     = env_cfg,
    scene_label = label,
    pull_dir    = [0., 0., 1.],   # bottle cap rotates up (+Z)
    init_pos    = [0.50, 0.00, 0.85],
    init_gauge  = 0.30,
)

# Build InteractiveScene while still redirected (env init is noisy)
scene = InteractiveScene(scene_cfg, args_cli, simulation_app)

# ── restore terminal for curses ────────────────────────────────────────────────
sys.stdout.flush()
sys.stderr.flush()
_to_term()
sys.stdout = os.fdopen(os.dup(1), 'w', 1)
sys.stderr = os.fdopen(os.dup(2), 'w', 1)

if __name__ == "__main__":
    try:
        scene.run(_TERM_OUT, _TERM_ERR)
    finally:
        _to_log()
        simulation_app.close()
    _to_term()
    print(f"\nDone. Full log: {_LOG_PATH}")
