"""Shared interactive framework for InsightBench scene_interact scripts.

Usage (in each scene's interactive script):
    from scene_interact._interactive_base import InteractiveScene, SceneCfg
    cfg = SceneCfg(env_cfg=MyEnvCfg(), pull_dir=[0, 1, 0], scene_label="Cabinet")
    InteractiveScene(cfg, args_cli, simulation_app).run(term_out, term_err)
"""

import os, sys, math, curses
import torch
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG
from dataclasses import dataclass, field
from typing import List

_LOG_PATH = '/tmp/isaacsim_interactive.log'

# ── skill groups ───────────────────────────────────────────────────────────────
_POS_SKILLS   = {0, 1}      # EE target position + orientation
_PULL_SKILLS  = {2, 4}      # reference pos + gauge = pull/push distance
_ROT_SKILLS   = {3}         # gauge only = rotation angle

_FAR = torch.tensor([[999., 999., 999.]])
_IDQ = torch.tensor([[1.,   0.,   0.,   0.]])


# ── scene configuration ────────────────────────────────────────────────────────

@dataclass
class SceneCfg:
    env_cfg:      object
    scene_label:  str   = "Scene"
    pull_dir:     List[float] = field(default_factory=lambda: [0., -1., 0.])
    init_pos:     List[float] = field(default_factory=lambda: [0.50, 0.00, 0.75])
    init_gauge:   float = 0.30
    skill_names:  dict  = field(default_factory=lambda: {
        0: "Approach (IK)",
        1: "Grasp",
        2: "Constrained Pull",
        3: "Rotate joint-7",
        4: "Push/Pull (MG)",
        5: "Gripper Open",
        6: "Gripper Close",
    })


# ── main interactive class ─────────────────────────────────────────────────────

class InteractiveScene:
    def __init__(self, scene_cfg: SceneCfg, args_cli, simulation_app):
        self.cfg            = scene_cfg
        self.args_cli       = args_cli
        self.simulation_app = simulation_app

        from custom_lab.envs.manager_based_rl_step_env import ManagerBasedRLStepEnv
        self.env    = ManagerBasedRLStepEnv(cfg=scene_cfg.env_cfg)
        self.device = self.env.device

        self.ee_marker   = self._make_frame("/Visuals/ee_target")
        self.goal_marker = self._make_sphere("/Visuals/goal_target")
        self.ee_marker.visualize(translations=_FAR, orientations=_IDQ)
        self.goal_marker.visualize(translations=_FAR, orientations=_IDQ)

        self.S = {
            'skill'    : 1,
            'pos'      : list(scene_cfg.init_pos),
            'rpy_deg'  : [0.0, 0.0, 0.0],
            'gauge'    : scene_cfg.init_gauge,
            'pos_step' : 0.02,
            'execute'  : False,
            'reset'    : False,
            'quit'     : False,
            'msg'      : 'Ready — select skill (0-6), adjust, then SPACE',
        }
        self._term_out = self._term_err = None

    # ── marker helpers ─────────────────────────────────────────────────────────
    def _make_frame(self, path, scale=0.08):
        cfg = FRAME_MARKER_CFG.copy()
        cfg.prim_path = path
        cfg.markers["frame"].scale = (scale, scale, scale)
        return VisualizationMarkers(cfg)

    def _make_sphere(self, path, scale=0.05):
        cfg = SPHERE_MARKER_CFG.copy()
        cfg.prim_path = path
        cfg.markers["sphere"].scale = (scale, scale, scale)
        return VisualizationMarkers(cfg)

    def _rpy_to_quat(self, rpy_deg):
        t = torch.tensor([[math.radians(a) for a in rpy_deg]])
        return math_utils.quat_from_euler_xyz(t[:, 0], t[:, 1], t[:, 2])

    def _ee_forward(self):
        """World-space forward direction of EE (local +Z rotated by current orientation)."""
        quat = self._rpy_to_quat(self.S['rpy_deg'])   # (1,4)
        local_z = torch.tensor([[0., 0., 1.]])
        return math_utils.quat_apply(quat, local_z)    # (1,3)

    def _current_ee_pose(self):
        """Return (pos (1,3), quat (1,4)) of the actual EE from ee_frame sensor."""
        frame = self.env.scene["ee_frame"]
        pos  = frame.data.target_pos_w[0, 0].unsqueeze(0)   # (1,3)
        quat = frame.data.target_quat_w[0, 0].unsqueeze(0)  # (1,4)
        return pos, quat

    # ── marker update ──────────────────────────────────────────────────────────
    def _update_markers(self):
        sk = self.S['skill']
        if sk in _POS_SKILLS:
            pos_t = torch.tensor([self.S['pos']])
            quat  = self._rpy_to_quat(self.S['rpy_deg'])
            self.ee_marker.visualize(translations=pos_t, orientations=quat)
            self.goal_marker.visualize(translations=_FAR, orientations=_IDQ)
        elif sk == 2:
            # Sphere at: user-set reference pos + user-set EE forward * gauge
            self.ee_marker.visualize(translations=_FAR, orientations=_IDQ)
            dest = torch.tensor([self.S['pos']]) + self._ee_forward() * self.S['gauge']
            self.goal_marker.visualize(translations=dest, orientations=_IDQ)
        elif sk == 4:
            # Sphere at: actual current EE pos + actual EE forward * gauge
            self.ee_marker.visualize(translations=_FAR, orientations=_IDQ)
            try:
                ee_pos, ee_quat = self._current_ee_pose()
                local_z = torch.tensor([[0., 0., 1.]], device=self.device)
                fwd  = math_utils.quat_apply(ee_quat, local_z)
                dest = ee_pos + fwd * self.S['gauge']
                self.goal_marker.visualize(translations=dest, orientations=_IDQ)
            except Exception:
                self.goal_marker.visualize(translations=_FAR, orientations=_IDQ)
        else:
            self.ee_marker.visualize(translations=_FAR, orientations=_IDQ)
            self.goal_marker.visualize(translations=_FAR, orientations=_IDQ)

    # ── action builder ─────────────────────────────────────────────────────────
    def _build_action(self):
        sk    = self.S['skill']
        pos   = self.S['pos']
        quat  = self._rpy_to_quat(self.S['rpy_deg'])[0].tolist()
        gauge = self.S['gauge']
        pull  = self.cfg.pull_dir

        if sk in _POS_SKILLS:
            params = [float(sk)] + pos + quat
        elif sk == 2:
            # Skill 2: [2, dx, dy, dz, distance, 0, 0, 0] — direction + distance
            fwd = self._ee_forward()[0].tolist()
            params = [float(sk)] + fwd + [gauge, 0., 0., 0.]
        elif sk == 4:
            # Skill 4: [4, distance, 0, 0, 0, 0, 0, 0] — distance only, direction from EE state
            params = [float(sk), gauge, 0., 0., 0., 0., 0., 0.]
        elif sk in _ROT_SKILLS:
            params = [float(sk), math.radians(gauge * 90), 0, 0, 0, 0, 0, 0]
        else:
            params = [float(sk)] + [0.] * 7

        act = torch.tensor([params], dtype=torch.float32, device=self.device)
        return act.expand(self.args_cli.num_envs, -1)

    # ── key handler ───────────────────────────────────────────────────────────
    def _handle(self, key):
        S = self.S

        # Global keys
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
            # Position: WASD = X/Y, Q/E = Z+/-
            if   key == ord('w'): S['pos'][1] += d
            elif key == ord('s'): S['pos'][1] -= d
            elif key == ord('a'): S['pos'][0] -= d
            elif key == ord('d'): S['pos'][0] += d
            elif key in (ord('q'), ord('Q')): S['pos'][2] += d
            elif key == ord('e'): S['pos'][2] -= d
            # Rotation: arrows = pitch/yaw, [/] = roll
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
            # Gauge only (rotation angle): arrows or [/] or A/D
            if key in (curses.KEY_RIGHT, ord(']'), ord('d')):
                S['gauge'] = min( 1.0, S['gauge'] + g)
            elif key in (curses.KEY_LEFT, ord('['), ord('a')):
                S['gauge'] = max(-1.0, S['gauge'] - g)

    # ── TUI ───────────────────────────────────────────────────────────────────
    def _bar(self, v, lo=-1.0, hi=1.0, w=18):
        f = max(0, min(w, int((v - lo) / max(hi - lo, 1e-9) * w)))
        return '[' + '█' * f + '─' * (w - f) + f'] {v:+.3f}'

    def _hint(self, sk):
        if sk in _POS_SKILLS:
            return "W/S=Y  A/D=X  Q/E=Z  ↑↓=pitch  ←→/[]=yaw/roll"
        elif sk == 2:
            return "W/S=Y  A/D=X  Q/E=Z(ref)  ←→/[]=gauge(dist)"
        elif sk == 4:
            return "←→/[/]=gauge  (dir fixed to current EE)      "
        elif sk in _ROT_SKILLS:
            return "A/D or ←→/[/]: gauge (rotation angle)        "
        else:
            return "                                               "

    def _draw(self, win):
        S    = self.S
        sk   = S['skill']
        pos  = S['pos']
        rpy  = S['rpy_deg']
        rows, cols = win.getmaxyx()
        win.erase()

        W = 52  # inner width between ║ chars

        def put(r, text, attr=0):
            if r < rows:
                win.addnstr(r, 0, text, cols - 1, attr)

        def row(content, attr=0):
            return "║" + content.ljust(W) + "║"

        def sep(l="╠", r="╣"):
            return l + "═" * W + r

        sname = self.cfg.skill_names.get(sk, '?')
        label = self.cfg.scene_label
        header = f"══ InsightBench [{label}] "
        put(0,  "╔" + header + "═" * (W - len(header)) + "╗", curses.A_BOLD)
        put(1,  row(f"  Skill [{sk}] {sname}"))
        put(2,  row(f"  Pos   X={pos[0]:+.3f}  Y={pos[1]:+.3f}  Z={pos[2]:+.3f}"))
        put(3,  row(f"  Orient R={rpy[0]:+.1f}°  P={rpy[1]:+.1f}°  Y={rpy[2]:+.1f}°"))
        put(4,  row(f"  Gauge {self._bar(S['gauge'])}"))
        put(5,  row(f"  step={S['pos_step']:.3f}  (+/-) scale"))
        put(6,  sep())
        put(7,  row(f"  {self._hint(sk)}"))
        put(8,  row("  SPACE/ENTER=Execute  R=Reset  ESC=Quit"))
        put(9,  sep())
        put(10, row(f"  {S['msg'][:W - 2]}"), curses.A_DIM)
        put(11, sep("╚", "╝"))
        win.refresh()

    # ── fd helpers ────────────────────────────────────────────────────────────
    def _to_log(self):
        fd = os.open(_LOG_PATH, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)

    def _to_term(self):
        os.dup2(self._term_out, 1); os.dup2(self._term_err, 2)

    # ── curses loop ───────────────────────────────────────────────────────────
    def _curses_loop(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.keypad(True)

        while self.simulation_app.is_running():
            # Drain all pending keys — render takes time so multiple keys can queue up
            while True:
                key = stdscr.getch()
                if key == curses.ERR:
                    break
                self._handle(key)

            if self.S['quit']:
                break

            self._to_log()
            self._update_markers()
            self.env.sim.render()
            self._to_term()

            self._draw(stdscr)

            if self.S['reset']:
                self.S['reset'] = False
                self.S['msg'] = 'Resetting...'
                self._draw(stdscr)
                self._to_log()
                self.env.reset()
                self._to_term()
                self.S['msg'] = 'Reset done.'

            elif self.S['execute']:
                self.S['execute'] = False
                action = self._build_action()
                self.S['msg'] = f"Executing skill {self.S['skill']}..."
                self._draw(stdscr)
                self._to_log()
                self.env.step(action)
                self._to_term()
                self.S['msg'] = f"Done. pos={[f'{v:.3f}' for v in self.S['pos']]}"

    # ── public entry ──────────────────────────────────────────────────────────
    def run(self, term_out, term_err):
        self._term_out = term_out
        self._term_err = term_err
        try:
            curses.wrapper(self._curses_loop)
        finally:
            self._to_log()
            self.env.close()
        # Fully restore terminal — curses may leave raw mode or wrong size
        self._to_term()
        os.system('tput reset')
