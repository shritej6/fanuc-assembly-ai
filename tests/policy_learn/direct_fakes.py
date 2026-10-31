"""Test-only stand-ins for the direct runtime tests: a scripted GR00T and a tiny in-process cell. Never for the arm.

ScriptedGroot plays what a well-trained GR00T would for the runtime's current step (it reads the step from the
runtime, which the real model cannot): over the pad, down, close, up, over the seat, down, open, up. Every row
stays inside the guard's speed limits, so the guard has nothing to refuse on this path.
"""
from __future__ import annotations

import numpy as np

FAST_MM, SLOW_MM, FAST_DEG, SLOW_DEG = 25.0, 1.5, 10.0, 2.5   # per 0.2 s row: under 150 / 10 mm/s, 60 / 15 deg/s
HOVER_MM, ROWS = 30.0, 12
EDGE_MM = 6.5   # every path breaks here: just above the guard's 5 mm slow zone
HOME_TCP = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class ScriptedGroot:
    def __init__(self, runtime=None):
        self.runtime, self.resets, self.instructions = runtime, 0, []
        self._key, self._i = None, 0

    def reset(self) -> None:
        self.resets += 1

    def _waypoints(self, step, phase):
        g, s = step.grasp_tcp, step.seat_tcp
        if phase == "pick":
            return [(g[0], g[1], g[2] + HOVER_MM, g[5]), (g[0], g[1], g[2] + EDGE_MM, g[5]), (g[0], g[1], g[2], g[5]),
                    1.0]
        if phase == "insert":
            transit = max(g[2], s[2]) + 60.0
            return [(g[0], g[1], transit, s[5]), (s[0], s[1], transit, s[5]), (s[0], s[1], s[2] + EDGE_MM, s[5]),
                    (s[0], s[1], s[2], s[5]), 0.0]
        return [(s[0], s[1], s[2] + EDGE_MM, s[5]), (s[0], s[1], s[2] + HOVER_MM, s[5])]

    def predict(self, obs: dict) -> np.ndarray:
        pr = self.runtime.progress
        self.instructions.append(obs["instruction"])
        key = (pr.index, pr.phase)
        if key != self._key:
            self._key, self._i = key, 0
        wps = self._waypoints(pr.step, pr.phase)
        pose = np.asarray(obs["tcp_pose"], float)[:6].copy()
        grip = 1.0 if float(obs["joints"][6]) >= 0.5 else 0.0
        contact_z = (pr.step.grasp_tcp if pr.phase == "pick" else pr.step.seat_tcp)[2]
        rows = []
        while len(rows) < ROWS and self._i < len(wps):
            wp = wps[self._i]
            if not isinstance(wp, tuple):          # gripper change: alone, standing still
                if rows:
                    break
                self._i += 1
                return np.array([[*pose, wp]])
            pose, reached = self._toward(pose, wp, contact_z)
            rows.append([*pose, grip])
            if reached:
                self._i += 1
        return np.array(rows) if rows else np.array([[*pose, grip]])

    @staticmethod
    def _toward(pose: np.ndarray, wp, contact_z: float):
        slow = max(pose[2], wp[2]) <= contact_z + EDGE_MM + 0.1
        d = np.array([wp[0] - pose[0], wp[1] - pose[1], wp[2] - pose[2]])
        dr = _wrap(wp[3] - pose[5])
        n = float(np.linalg.norm(d))
        lin, rot = (SLOW_MM, SLOW_DEG) if slow else (FAST_MM, FAST_DEG)
        k = min(1.0, lin / n if n > 1e-9 else 1.0, rot / abs(dr) if abs(dr) > 1e-9 else 1.0)
        out = pose.copy()
        out[:3] += d * k
        out[5] += dr * k
        out[3], out[4] = -180.0, 0.0
        reached = np.linalg.norm(out[:3] - np.array(wp[:3])) < 1e-6 and abs(_wrap(wp[3] - out[5])) < 1e-6
        return out, bool(reached)


class TinyCell:
    """Each chunk is played instantly and exactly: tcp_pose = last row, gripper = last row's gripper."""

    def __init__(self, instruction: str, index: int = 0, tcp=None):
        self.tcp = list(tcp or HOME_TCP)
        self.grip = 0.0
        self.trial = {"instruction": instruction, "index": index}

    def obs(self) -> dict:
        joints = np.array([0.0, 0.0, 0.0, 0.0, -90.0, 0.0, self.grip], np.float32)
        return {"images": {}, "joints": joints, "tcp_pose": list(self.tcp), "trial": self.trial}

    def play(self, rows) -> None:
        last = np.asarray(rows, float).reshape(-1, 7)[-1]
        self.tcp, self.grip = list(last[:6]), float(last[6])
