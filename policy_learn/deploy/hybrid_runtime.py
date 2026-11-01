"""Hybrid runtime: GR00T drives approach and carry, the planner does the last centimetres of grasp and insert.
Authored by Shritej Gunda."""
from __future__ import annotations

import numpy as np

from policy_learn.deploy.direct_runtime import DirectGrootRuntime

TAKEOVER_XY_MM = 25.0
TAKEOVER_Z_MM = 40.0
INSERT_Z_MM = 80.0
HOVER_MM = 40.0          # turn the wrist here: GR00T can arrive with the roll ~140 deg off
LIFT_MM = 40.0
RETREAT_UP_MM = 20.0
REACHED_MM = 1.0
EDGE_MM = 6.5          # just above the guard's 5 mm slow zone, as tests/policy_learn/direct_fakes.py
FAST_MM, SLOW_MM, FAST_DEG, SLOW_DEG = 20.0, 1.5, 8.0, 2.5   # per 0.2 s row: under 150 / 10 mm/s, 60 / 15 deg/s


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _path(tcp: np.ndarray, goal, grip_after: float, after_mm: float) -> list:
    """Return waypoints and gripper commands from the TCP to a planned pose and back up."""
    x, y, z, r = goal[0], goal[1], goal[2], goal[5]
    hover = z + HOVER_MM
    return [(tcp[0], tcp[1], max(tcp[2], hover), tcp[5]), (x, y, hover, r), (x, y, z + EDGE_MM, r), (x, y, z, r),
            grip_after, (x, y, z + EDGE_MM, r), (x, y, z + after_mm, r)]


class HybridRuntime(DirectGrootRuntime):
    """Direct runtime with planned final approach for grasp and insert."""
    STEP_MM_DEG = (FAST_MM, SLOW_MM, FAST_DEG, SLOW_DEG)

    def reset(self) -> None:
        """Start a new trial."""
        super().reset()
        self._plan_path: list | None = None
        self._plan_i = 0

    def _maybe_take_over(self, tcp: np.ndarray, closed: bool) -> None:
        pr = self.progress
        if getattr(self, "_plan_path", None) is not None or pr.ended is not None or pr.step is None:
            return
        if pr.phase == "pick" and not closed:
            goal, z_max, grip_after, after, kind = pr.step.grasp_tcp, TAKEOVER_Z_MM, 1.0, LIFT_MM, "grasp"
        elif pr.phase == "insert" and closed:
            goal, z_max, grip_after, after, kind = pr.step.seat_tcp, INSERT_Z_MM, 0.0, RETREAT_UP_MM, "insert"
        else:
            return
        dxy, dz = float(np.hypot(tcp[0] - goal[0], tcp[1] - goal[1])), float(tcp[2] - goal[2])
        if dxy <= TAKEOVER_XY_MM and -1.0 <= dz <= z_max:
            self._plan_path, self._plan_i, self._plan_contact = _path(tcp, goal, grip_after, after), 0, goal[2]
            self._event("takeover", mode=kind, part=pr.step.object_id, dxy=round(dxy, 1), dz=round(dz, 1))

    def _planned_rows(self, tcp: np.ndarray, grip: float) -> np.ndarray:
        """Return rows from the reported TCP toward the next unreached waypoint."""
        path, pose = self._plan_path, tcp.copy()
        while self._plan_i < len(path):
            wp = path[self._plan_i]
            if isinstance(wp, tuple) and np.linalg.norm(pose[:3] - wp[:3]) <= REACHED_MM or not isinstance(wp, tuple) and (grip >= 0.5) == (wp >= 0.5):
                self._plan_i += 1
            else:
                break
        if self._plan_i >= len(path):
            self._plan_path = None
            return np.empty((0, 7))
        rows, i = [], self._plan_i
        if not isinstance(path[i], tuple) and np.linalg.norm(pose[:3] - path[i - 1][:3]) > REACHED_MM:
            i -= 1                                     # not at the grasp/seat pose yet: get there first
        rows_i = self._chunk_from(pose, grip, i, rows)
        self._plan_i = rows_i
        return np.array(rows, float)

    def _chunk_from(self, pose: np.ndarray, grip: float, i: int, rows: list) -> int:
        path = self._plan_path
        while len(rows) < self.exec_horizon and i < len(path):
            wp = path[i]
            if not isinstance(wp, tuple):              # gripper change: standing still, alone at a chunk start
                if rows:
                    break
                rows.append([*pose, wp])
                grip, i = wp, i + 1
                continue
            pose, reached = self._toward(pose, wp)
            rows.append([*pose, grip])
            i += reached
        return i

    def _toward(self, pose: np.ndarray, wp):
        edge = self._plan_contact + EDGE_MM
        if pose[2] > edge and wp[2] < edge - 0.1:      # a lagging arm far above the slow zone: fast down to its edge first
            out, _ = self._toward(pose, (wp[0], wp[1], edge, wp[3]))
            return out, False
        slow = min(pose[2], wp[2]) <= edge - 0.1       # as the guard: either end in the slow zone
        d = np.array(wp[:3]) - pose[:3]
        dr = _wrap(wp[3] - pose[5])
        n = float(np.linalg.norm(d))
        fast_mm, slow_mm, fast_deg, slow_deg = self.STEP_MM_DEG
        lin, rot = (slow_mm, slow_deg) if slow else (fast_mm, fast_deg)
        k = min(1.0, lin / n if n > 1e-9 else 1.0, rot / abs(dr) if abs(dr) > 1e-9 else 1.0)
        out = pose.copy()
        out[:3] += d * k
        out[5] += dr * k
        out[3], out[4] = -180.0, 0.0
        return out, bool(k >= 1.0)

    def act(self, obs: dict) -> np.ndarray:
        """Return guarded rows for one observation."""
        if obs.get("tcp_pose") is None or self.progress.rollout is None or self.progress.ended is not None:
            return super().act(obs)
        tcp = np.asarray(obs["tcp_pose"], float)[:6]
        grip = float(np.asarray(obs["joints"], float)[6])
        self._update(obs, tcp, grip >= 0.5)
        if self.progress.ended is not None:
            self.steps += 1
            return super().act(obs)
        self._maybe_take_over(tcp, grip >= 0.5)
        rows = self._planned_rows(tcp, grip) if getattr(self, "_plan_path", None) is not None else np.empty((0, 7))
        if not len(rows):
            return self._groot_rows(obs)
        self.steps += 1
        res = self.guard.guard(rows, tcp, grip, self.DT)
        out = self._record(res)
        self.progress.last_rows = np.asarray(out, float).reshape(-1, 7)
        if res.held:
            self.progress.holds_in_row += 1
            if self.progress.holds_in_row >= self.budgets.stuck_holds:
                self._end(f"{self.progress.step.label}: guard refused {self.progress.holds_in_row} planner chunks")
        else:
            self.progress.holds_in_row = 0
        return out

    def _groot_rows(self, obs: dict) -> np.ndarray:
        """Return the GR00T rows for this step without repeating the progress update."""
        pr = self.progress
        logged = len(self.guard_log)
        rows = super(DirectGrootRuntime, self).act(obs)
        pr.last_rows = np.asarray(rows, float).reshape(-1, 7)
        held = len(self.guard_log) > logged and self.guard_log[-1]["held"]
        pr.holds_in_row = pr.holds_in_row + 1 if held else 0
        if pr.holds_in_row >= self.budgets.stuck_holds:
            self._end(f"{pr.step.label}: guard refused {pr.holds_in_row} chunks in a row (stuck)")
        return rows
