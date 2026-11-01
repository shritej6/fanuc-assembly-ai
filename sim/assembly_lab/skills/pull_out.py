"""Pull-out skill: slow straight lift with a small wiggle, then up the planner's clear path.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np

from .geometry import tilted
from .motion import APPROACH, CLOSED, FREE, OPEN, SLOW, Motion, Segment, above
from .plan import PULL_WIGGLE_DEG, SEAT_FLOOR_MM, SLOW_ZONE_MM, PullOutPlan

WIGGLE_STEPS = 4  # +1, -1, +1, 0 deg while backing out of the engagement
# (deg, steps) of the wiggle; record.py --pull-wiggle sets it for an A/B test on second-layer pulls (run10
# probe M1_03: the stack under the pulled part shifted 1-1.5 mm). Default = +-1 deg in 4 steps.
WIGGLE = (PULL_WIGGLE_DEG, WIGGLE_STEPS)


def set_wiggle(deg: float, steps: int) -> None:
    """Wiggle amplitude (deg, 0 = straight lift) and number of equal rise steps (the last one ends at 0 deg)."""
    global WIGGLE
    if deg < 0 or deg > PULL_WIGGLE_DEG or steps < 1:
        raise ValueError(f"pull wiggle {deg} deg / {steps} steps: need 0 <= deg <= {PULL_WIGGLE_DEG}, steps >= 1")
    WIGGLE = (float(deg), int(steps))


def wiggle() -> tuple[float, int]:
    return WIGGLE


def wiggle_signs(steps: int) -> list[float]:
    """+1, -1, +1, ... ending at 0 (back to the grasp yaw for the lift)."""
    return [1.0 if i % 2 == 0 else -1.0 for i in range(steps - 1)] + [0.0]


def subtask_name(plan: PullOutPlan) -> str:
    return f"pull_out:{plan.object_id}"


def carry(m: Motion, via: list[np.ndarray], goal) -> None:
    """Go up to the higher of here / the goal height, follow the planner's clear path, arrive above the goal."""
    goal = np.asarray(goal, dtype=float)
    m.move("carry", above(m.pose, max(m.pose[2], goal[2])), FREE)
    for wp in via:
        m.move("carry", wp, FREE)
    m.move("carry", above(goal, max(m.pose[2], goal[2])), FREE)
    m.move("carry", goal, FREE)


def _wiggle_out(m: Motion, plan: PullOutPlan) -> None:
    z0 = m.pose[2]
    base = m.pose.copy()
    rise = plan.engage_mm + 1.0  # 1 mm past the end of the engagement
    deg, steps = WIGGLE
    signs = wiggle_signs(steps)
    for i, s in enumerate(signs, start=1):
        m.move("wiggle_out", tilted(above(base, z0 + rise * i / len(signs)), 0.0, 0.0, s * deg), SLOW)


def pull_out(plan: PullOutPlan, start, gripper: float = OPEN, holding: bool = False,
             learn: bool = True) -> list[Segment]:
    m = Motion(np.asarray(start, dtype=float), gripper, "pull_out", subtask_name(plan), plan.object_id, learn=learn)
    g = plan.grasp_tcp
    if not holding:
        m.enter()
        carry(m, [], above(g, plan.clear_z_mm))
        if m.gripper != OPEN:
            m.grip("open", OPEN)
        m.floor_z = g[2] - SEAT_FLOOR_MM
        m.contact_z = g[2]
        m.move("descend", g, APPROACH)
        m.grip("close", CLOSED)
    m.contact_z = m.pose[2] + plan.engage_mm + 1.0 - SLOW_ZONE_MM  # slow through the engagement
    _wiggle_out(m, plan)
    m.contact_z = None
    m.move("lift", above(m.pose, plan.clear_z_mm), APPROACH)
    for wp in plan.via:
        m.move("lift", wp, FREE)
    return m.segments
