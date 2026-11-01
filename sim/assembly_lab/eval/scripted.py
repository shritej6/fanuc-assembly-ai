"""Sim-only scripted baseline that executes the director's current step from CAD targets.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import numpy as np

CLEAR_MM = 40.0          # approach / retreat height above a contact
FREE_MM = 8.0            # per row at dt 0.2: 40 mm/s (guard: 150 mm/s)
SLOW_MM = 0.4            # per row: 2 mm/s in the last 5 mm (guard: 10 mm/s)
SLOW_ZONE_MM = 5.0
SETTLE_ROWS = 4          # rows held still while the fingers open / close
HORIZON = 16
HOLD_ROWS = 10


def _segment(a: np.ndarray, b: np.ndarray, grip: float, step_mm: float) -> list[np.ndarray]:
    d = b[:3] - a[:3]
    turn = np.array([(b[3] - a[3] + 180.0) % 360.0 - 180.0, b[4] - a[4], (b[5] - a[5] + 180.0) % 360.0 - 180.0])
    n = max(1, int(np.ceil(max(np.linalg.norm(d) / step_mm, np.abs(turn).max() / 5.0))))
    return [np.concatenate([a[:3] + d * (k / n), a[3:6] + turn * (k / n), [grip]]) for k in range(1, n + 1)]


def _descend(a: np.ndarray, b: np.ndarray, grip: float) -> list[np.ndarray]:
    mid = b.copy()
    mid[2] = b[2] + SLOW_ZONE_MM + 1.0  # the fast part ends 1 mm above the guard's slow zone
    return _segment(a, mid, grip, FREE_MM) + _segment(mid, b, grip, SLOW_MM)


def _ascend(a: np.ndarray, b: np.ndarray, grip: float) -> list[np.ndarray]:
    mid = a.copy()
    mid[2] = a[2] + SLOW_ZONE_MM + 1.0  # leave the slow zone slowly too
    return _segment(a, mid, grip, SLOW_MM) + _segment(mid, b, grip, FREE_MM)


def plan_rows(start: np.ndarray, grasp: np.ndarray, goal: np.ndarray, release_mm: float = 0.0) -> np.ndarray:
    """Full step from the current pose; release_mm > 0 lets go that far above the goal (defect drop)."""
    up_g, up_t = grasp.copy(), goal.copy()
    up_g[2] += CLEAR_MM
    up_t[2] += CLEAR_MM
    hi = max(start[2], up_g[2], up_t[2])
    s_hi, g_hi = start.copy(), up_g.copy()
    s_hi[2] = g_hi[2] = hi
    rows = _segment(start, s_hi, 0.0, FREE_MM) + _segment(s_hi, g_hi, 0.0, FREE_MM)
    rows += _segment(g_hi, up_g, 0.0, FREE_MM) + _descend(up_g, grasp, 0.0)
    rows += [np.concatenate([grasp, [1.0]])] * SETTLE_ROWS
    t_hi = up_t.copy()
    t_hi[2] = hi
    rows += _ascend(grasp, up_g, 1.0) + _segment(up_g, g_hi, 1.0, FREE_MM)
    rows += _segment(g_hi, t_hi, 1.0, FREE_MM) + _segment(t_hi, up_t, 1.0, FREE_MM)
    put = goal.copy()
    put[2] += release_mm
    rows += _descend(up_t, put, 1.0) if release_mm == 0 else _segment(up_t, put, 1.0, FREE_MM)
    rows += [np.concatenate([put, [0.0]])] * SETTLE_ROWS
    rows += _ascend(put, up_t, 0.0)
    return np.array(rows)


class ScriptedOracleModel:
    """ChunkModel for GrootPolicy(model=...): plays plan_rows() for the director's current step, HORIZON at a time."""

    def __init__(self, director_ref):
        self.director_ref = director_ref   # callable -> the live Director
        self.key, self.rows, self.cursor, self.last = None, None, 0, 0

    def reset(self) -> None:
        """Clear the current step."""
        self.key, self.rows, self.cursor, self.last = None, None, 0, 0

    def predict(self, obs: dict) -> np.ndarray:
        """Return the next action chunk for the director's current step."""
        t = self.director_ref().current
        tcp = np.asarray(obs["tcp_pose"], float)[:6]
        key = (t.uid, t.skill, t.attempts, t.started_s)
        if key != self.key:
            release = 15.0 if t.skill == "defect_drop" else 0.0
            self.key, self.rows, self.cursor = key, plan_rows(tcp, t.grasp_tcp, t.goal_tcp, release), 0
        elif self.cursor and np.linalg.norm(self.rows[self.cursor - 1, :3] - tcp[:3]) > 2.0:
            self.cursor = self.last  # the last chunk was held by the guard: offer it again, never skip ahead
        self.last = self.cursor
        chunk = self.rows[self.cursor:self.cursor + HORIZON]
        self.cursor += len(chunk)
        if len(chunk) == 0:  # finished: hold above the goal until the director moves on (2 s per chunk, not 0.2 s)
            return np.repeat(np.concatenate([tcp, [float(obs["joints"][6] >= 0.5)]])[None], HOLD_ROWS, axis=0)
        return chunk
