"""Interactive door scene — keyboard control + EE target visualization.

Isaac Sim startup output is redirected to /tmp/isaacsim_interactive.log so
the curses TUI is never garbled.  Markers update via env.sim.render() every
frame so they move immediately in the WebRTC view.

Keyboard Controls
-----------------
  Skill:    1=Approach  2=Grasp  3=Pull  4=Rotate  5=Gripper Open  6=Gripper Close
  Position: W/S=Y±   A/D=X±   Q/E=Z±
  Orient:   ↑↓=pitch  ←→=yaw  [/]=roll
  Gauge:    A/D (skills 3,4) — decrease / increase
  Step:     +/-  scale position step size
  Execute:  ENTER or SPACE
  Reset:    R
  Quit:     ESC or Q

Run:
  python scene_interact/interact_scene3_interactive.py \\
      --num_envs 1 --livestream 2 --enable_cameras
  (Isaac Sim log → /tmp/isaacsim_interactive.log)
"""

import sys, os

this_file = os.path.abspath(__file__)
sys.path.insert(0, os.path.dirname(os.path.dirname(this_file)))

# ── fd helpers — save terminal fds BEFORE any redirect ───────────────────────
_LOG_PATH  = '/tmp/isaacsim_interactive.log'
_TERM_OUT  = os.dup(1)   # permanent copy of original terminal stdout
_TERM_ERR  = os.dup(2)   # permanent copy of original terminal stderr

def _to_log():
    """Redirect fd 1/2 → log file (suppresses all C++ output)."""
    fd = os.open(_LOG_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.dup2(fd, 1);  os.dup2(fd, 2);  os.close(fd)

def _to_term():
    """Restore fd 1/2 → original terminal (so curses can render)."""
    os.dup2(_TERM_OUT, 1);  os.dup2(_TERM_ERR, 2)

# Redirect for noisy Isaac Sim startup
_to_log()
_save_out = _TERM_OUT   # alias kept for compat with restore block below
_save_err = _TERM_ERR

# ── Isaac Sim launch (all its noisy output goes to log file) ──────────────────
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher    = AppLauncher(args_cli)
simulation_app  = app_launcher.app

# ── post-launch imports ────────────────────────────────────────────────────────
import math, curses
import torch
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG

from custom_lab.envs.manager_based_rl_step_env import ManagerBasedRLStepEnv
from cfg.scene3Cfg import DoorSkillPullEnvCfg

# ── build env (still redirected — noisy) ─────────────────────────────────────
env_cfg = DoorSkillPullEnvCfg()
env_cfg.scene.num_envs = args_cli.num_envs
env = ManagerBasedRLStepEnv(cfg=env_cfg)
device = env.device

# ── markers ────────────────────────────────────────────────────────────────────
_FAR = torch.tensor([[999., 999., 999.]])
_IDQ = torch.tensor([[1.,  0.,  0.,  0.]])

def _make_frame(path, scale=0.08):
    cfg = FRAME_MARKER_CFG.copy()
    cfg.prim_path = path
    cfg.markers["frame"].scale = (scale, scale, scale)
    return VisualizationMarkers(cfg)

def _make_sphere(path, scale=0.05):
    cfg = SPHERE_MARKER_CFG.copy()
    cfg.prim_path = path
    cfg.markers["sphere"].scale = (scale, scale, scale)
    return VisualizationMarkers(cfg)

ee_marker   = _make_frame("/Visuals/ee_target")
goal_marker = _make_sphere("/Visuals/goal_target")

ee_marker.visualize(translations=_FAR, orientations=_IDQ)
goal_marker.visualize(translations=_FAR, orientations=_IDQ)

# ── Restore terminal for curses (startup noise is done) ──────────────────────
sys.stdout.flush()
sys.stderr.flush()
_to_term()
sys.stdout = os.fdopen(os.dup(1), 'w', 1)
sys.stderr = os.fdopen(os.dup(2), 'w', 1)

# ── shared state ───────────────────────────────────────────────────────────────
SKILL_NAMES = {
    0: "Approach (IK)",
    1: "Grasp",
    2: "Constrained Pull",
    3: "Rotate joint-7",
    4: "Push/Pull (MG)",
    5: "Gripper Open",
    6: "Gripper Close",
}
_POS_SKILLS  = {0, 1}
_PULL_SKILLS = {2, 4}
_ROT_SKILLS  = {3}

S = {
    'skill'    : 1,
    'pos'      : [0.50, 0.00, 0.75],
    'rpy_deg'  : [0.0, 0.0, 0.0],
    'gauge'    : 0.30,
    'pos_step' : 0.02,
    'execute'  : False,
    'reset'    : False,
    'quit'     : False,
    'msg'      : 'Ready',
}

# ── key handler ────────────────────────────────────────────────────────────────
def _handle(key):
    if key == 27:  # ESC only — Q/q is used for Z+
        S['quit'] = True;  return
    if key in (ord('r'), ord('R')):
        S['reset'] = True; return
    if key in (10, 13, ord(' ')):
        S['execute'] = True; return
    if ord('0') <= key <= ord('6'):
        S['skill'] = key - ord('0'); return
    if key == ord('+'):
        S['pos_step'] = min(S['pos_step'] * 2, 0.5);  return
    if key == ord('-'):
        S['pos_step'] = max(S['pos_step'] / 2, 0.001); return

    d  = S['pos_step']
    g  = 0.05
    r  = 5.0
    sk = S['skill']

    if sk in _POS_SKILLS:
        if   key == ord('w'): S['pos'][1] += d
        elif key == ord('s'): S['pos'][1] -= d
        elif key == ord('a'): S['pos'][0] -= d
        elif key == ord('d'): S['pos'][0] += d
        elif key in (ord('q'), ord('Q')): S['pos'][2] += d
        elif key == ord('e'): S['pos'][2] -= d
        elif key == curses.KEY_UP:    S['rpy_deg'][1] += r
        elif key == curses.KEY_DOWN:  S['rpy_deg'][1] -= r
        elif key == curses.KEY_RIGHT: S['rpy_deg'][2] += r
        elif key == curses.KEY_LEFT:  S['rpy_deg'][2] -= r
        elif key == ord('['): S['rpy_deg'][0] -= r
        elif key == ord(']'): S['rpy_deg'][0] += r

    elif sk == 2:
        # Skill 2: reference pos editable (WASD/QE) + gauge (arrows/[/])
        if   key == ord('w'): S['pos'][1] += d
        elif key == ord('s'): S['pos'][1] -= d
        elif key == ord('a'): S['pos'][0] -= d
        elif key == ord('d'): S['pos'][0] += d
        elif key in (ord('q'), ord('Q')): S['pos'][2] += d
        elif key == ord('e'): S['pos'][2] -= d
        elif key in (curses.KEY_RIGHT, ord(']')):
            S['gauge'] = min( 1.0, S['gauge'] + g)
        elif key in (curses.KEY_LEFT, ord('[')):
            S['gauge'] = max(-1.0, S['gauge'] - g)

    elif sk == 4:
        # Skill 4: direction fixed to actual EE state, gauge only
        if key in (curses.KEY_RIGHT, ord(']')):
            S['gauge'] = min( 1.0, S['gauge'] + g)
        elif key in (curses.KEY_LEFT, ord('[')):
            S['gauge'] = max(-1.0, S['gauge'] - g)

    elif sk in _ROT_SKILLS:
        if key in (curses.KEY_RIGHT, ord(']'), ord('d')):
            S['gauge'] = min( 1.0, S['gauge'] + g)
        elif key in (curses.KEY_LEFT, ord('['), ord('a')):
            S['gauge'] = max(-1.0, S['gauge'] - g)

# ── marker update ──────────────────────────────────────────────────────────────
def _rpy_to_quat(rpy_deg):
    t = torch.tensor([[math.radians(a) for a in rpy_deg]])
    return math_utils.quat_from_euler_xyz(t[:, 0], t[:, 1], t[:, 2])

def _ee_forward():
    """World-space forward direction of EE (local +Z rotated by current orientation)."""
    quat = _rpy_to_quat(S['rpy_deg'])
    return math_utils.quat_apply(quat, torch.tensor([[0., 0., 1.]]))  # (1,3)

def _update_markers():
    sk = S['skill']
    if sk in _POS_SKILLS:
        pos_t = torch.tensor([S['pos']])
        quat  = _rpy_to_quat(S['rpy_deg'])
        ee_marker.visualize(translations=pos_t, orientations=quat)
        goal_marker.visualize(translations=_FAR, orientations=_IDQ)
    elif sk == 2:
        # Sphere at: user-set reference pos + user-set EE forward * gauge
        ee_marker.visualize(translations=_FAR, orientations=_IDQ)
        dest = torch.tensor([S['pos']]) + _ee_forward() * S['gauge']
        goal_marker.visualize(translations=dest, orientations=_IDQ)
    elif sk == 4:
        # Sphere at: actual current EE pos + actual EE forward * gauge
        ee_marker.visualize(translations=_FAR, orientations=_IDQ)
        try:
            frame = env.scene["ee_frame"]
            ee_pos  = frame.data.target_pos_w[0, 0].unsqueeze(0)
            ee_quat = frame.data.target_quat_w[0, 0].unsqueeze(0)
            fwd  = math_utils.quat_apply(ee_quat, torch.tensor([[0., 0., 1.]]))
            dest = ee_pos + fwd * S['gauge']
            goal_marker.visualize(translations=dest, orientations=_IDQ)
        except Exception:
            goal_marker.visualize(translations=_FAR, orientations=_IDQ)
    else:
        # Skill 3 (rotate), 5, 6 — no spatial marker
        ee_marker.visualize(translations=_FAR, orientations=_IDQ)
        goal_marker.visualize(translations=_FAR, orientations=_IDQ)

# ── action builder ─────────────────────────────────────────────────────────────
def _build_action():
    sk    = S['skill']
    pos   = S['pos']
    quat  = _rpy_to_quat(S['rpy_deg'])[0].tolist()
    gauge = S['gauge']

    if sk in _POS_SKILLS:
        params = [float(sk)] + pos + quat
    elif sk == 2:
        # Skill 2: [2, dx, dy, dz, distance, 0, 0, 0] — direction + distance
        fwd = _ee_forward()[0].tolist()
        params = [float(sk)] + fwd + [gauge, 0., 0., 0.]
    elif sk == 4:
        # Skill 4: [4, distance, 0, 0, 0, 0, 0, 0] — distance only, direction from EE state
        params = [float(sk), gauge, 0., 0., 0., 0., 0., 0.]
    elif sk in _ROT_SKILLS:
        params = [float(sk), math.radians(gauge * 90), 0, 0, 0, 0, 0, 0]
    else:
        params = [float(sk)] + [0.] * 7

    act = torch.tensor([params], dtype=torch.float32, device=device)
    return act.expand(args_cli.num_envs, -1)

# ── curses TUI ─────────────────────────────────────────────────────────────────
def _bar(v, lo=-1.0, hi=1.0, w=18):
    f = max(0, min(w, int((v - lo) / max(hi - lo, 1e-9) * w)))
    return '[' + '█' * f + '─' * (w - f) + f'] {v:+.3f}'

def _draw(win):
    win.erase()
    sk   = S['skill']
    pos  = S['pos']
    rpy  = S['rpy_deg']

    rows, cols = win.getmaxyx()
    W = 52  # inner width between ║ chars

    def put(r, text, attr=0):
        if r < rows:
            win.addnstr(r, 0, text, cols - 1, attr)

    def row(content, attr=0):
        return "║" + content.ljust(W) + "║"

    def sep(l="╠", r="╣"):
        return l + "═" * W + r

    if sk in _POS_SKILLS:
        hint = "W/S=Y  A/D=X  Q/E=Z  ↑↓=pitch  ←→/[]=yaw/roll"
    elif sk in _PULL_SKILLS:
        hint = "W/S=Y  A/D=X  Q/E=Z(ref)  ←→/[]=gauge(dist)"
    elif sk in _ROT_SKILLS:
        hint = "A/D or ←→/[/]: gauge (rotation angle)"
    else:
        hint = ""

    header = "══ InsightBench — Door Scene "
    put(0,  "╔" + header + "═" * (W - len(header)) + "╗", curses.A_BOLD)
    put(1,  row(f"  Skill [{sk}] {SKILL_NAMES.get(sk, '?')}"))
    put(2,  row(f"  Pos   X={pos[0]:+.3f}  Y={pos[1]:+.3f}  Z={pos[2]:+.3f}"))
    put(3,  row(f"  Orient R={rpy[0]:+.1f}°  P={rpy[1]:+.1f}°  Y={rpy[2]:+.1f}°"))
    put(4,  row(f"  Gauge {_bar(S['gauge'])}"))
    put(5,  row(f"  step={S['pos_step']:.3f}  (+/-) scale"))
    put(6,  sep())
    put(7,  row(f"  {hint}"))
    put(8,  row("  SPACE/ENTER=Execute  R=Reset  ESC=Quit"))
    put(9,  sep())
    put(10, row(f"  {S['msg'][:W - 2]}"), curses.A_DIM)
    put(11, sep("╚", "╝"))
    win.refresh()

# ── main interactive loop (inside curses.wrapper) ──────────────────────────────
def _loop(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)   # non-blocking getch
    stdscr.keypad(True)    # enable KEY_UP etc.

    S['msg'] = 'Ready — select skill (1-6), adjust, then SPACE to execute'

    while simulation_app.is_running():
        # Drain all pending keys — render takes time so multiple keys can queue up
        while True:
            key = stdscr.getch()
            if key == curses.ERR:
                break
            _handle(key)

        if S['quit']:
            break

        # Update markers + render — redirect so carb/CuRobo output stays in log
        _to_log()
        _update_markers()
        env.sim.render()
        _to_term()

        # Curses redraw (fd 1 is terminal here)
        _draw(stdscr)

        if S['reset']:
            S['reset'] = False
            S['msg'] = 'Resetting...'
            _draw(stdscr)
            _to_log()
            env.reset()
            _to_term()
            S['msg'] = 'Reset done.'

        elif S['execute']:
            S['execute'] = False
            action = _build_action()
            S['msg'] = f"Executing skill {S['skill']}..."
            _draw(stdscr)
            _to_log()
            env.step(action)
            _to_term()
            S['msg'] = f"Done. pos={[f'{v:.3f}' for v in S['pos']]}"

# ── entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        curses.wrapper(_loop)
    finally:
        _to_log()
        env.close()
        simulation_app.close()
    _to_term()
    os.system('tput reset')
    print(f"\nDone. Full log: {_LOG_PATH}")
