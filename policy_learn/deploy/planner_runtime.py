"""PLANNER-DRIVEN runtime, scripted motion (user-approved 2026-10-31 ~15:15 CT, last slot): the planner drives every
step from placement.yaml (approach at a safe transit height, grasp, carry, insert, retreat); GR00T is not asked for motion. It is
HybridRuntime with the takeover always on, so every row still goes through the same ChunkGuard and the same
grasp / seated / jam / time / corridor checks of DirectGrootRuntime. Every planned leg is logged as a "takeover".

    from policy_learn.deploy.planner_runtime import PlannerRuntime
    policy = PlannerRuntime("groot://127.0.0.1:5557")   # the checkpoint is kept for the interface; no chunk is requested

No perception: a base hand-placed farther off than the part's capture (0.5-3 mm) jams and the seated detector holds.
TRANSIT_MM is PRELIMINARY.
"""
from __future__ import annotations

import numpy as np

from policy_learn.deploy.hybrid_runtime import EDGE_MM, HOVER_MM, LIFT_MM, RETREAT_UP_MM, HybridRuntime

TRANSIT_MM = 100.0   # above the target's contact height while travelling: clears the other parts and the base


def _planner_path(tcp: np.ndarray, goal, grip_after: float, after_mm: float) -> list:
    x, y, z, r = goal[0], goal[1], goal[2], goal[5]
    transit = max(tcp[2], z + TRANSIT_MM)
    return [(tcp[0], tcp[1], transit, tcp[5]), (x, y, transit, r), (x, y, z + HOVER_MM, r), (x, y, z + EDGE_MM, r),
            (x, y, z, r), grip_after, (x, y, z + EDGE_MM, r), (x, y, z + after_mm, r)]


class PlannerRuntime(HybridRuntime):
    # user 2026-10-31 15:3x: the arm was too slow. Per 0.2 s row: 140 / 9 mm/s and 55 / 12.5 deg/s, still under the
    # guard's 150 / 10 mm/s and 60 / 15 deg/s (the limits are unchanged)
    STEP_MM_DEG = (28.0, 1.8, 11.0, 2.5)

    def act(self, obs: dict) -> np.ndarray:
        """Start the trial here so its first act is planned too (the parent would ask GR00T for it)."""
        pr = self.progress
        if obs.get("tcp_pose") is not None and pr.rollout is None and pr.ended is None:
            grip = float(np.asarray(obs["joints"], float)[6])
            self._begin_trial(obs, np.asarray(obs["tcp_pose"], float)[:6], grip >= 0.5)
        return super().act(obs)

    def _maybe_take_over(self, tcp: np.ndarray, closed: bool) -> None:
        pr = self.progress
        if getattr(self, "_plan_path", None) is not None or pr.ended is not None or pr.step is None:
            return
        if pr.phase == "pick" and not closed:
            goal, grip_after, after, kind = pr.step.grasp_tcp, 1.0, LIFT_MM, "grasp"
        elif pr.phase == "insert" and closed:
            goal, grip_after, after, kind = pr.step.seat_tcp, 0.0, RETREAT_UP_MM, "insert"
        else:
            return
        self._plan_path, self._plan_i, self._plan_contact = _planner_path(tcp, goal, grip_after, after), 0, goal[2]
        self._event("takeover", mode=f"planner {kind}", part=pr.step.object_id,
                    dxy=round(float(np.hypot(tcp[0] - goal[0], tcp[1] - goal[1])), 1),
                    dz=round(float(tcp[2] - goal[2]), 1))

    def _groot_rows(self, obs: dict) -> np.ndarray:
        """Between planned legs (e.g. the retreat check before the next step starts) stand still: no GR00T motion."""
        tcp = np.asarray(obs["tcp_pose"], float)[:6]
        grip = float(np.asarray(obs["joints"], float)[6])
        self.steps += 1
        hold = np.array([[*tcp, 1.0 if grip >= 0.5 else 0.0]])
        self.progress.last_rows = hold
        return hold
