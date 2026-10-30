"""pick(plan): vertical corridor above the grip pad, open, descend, close, short check lift, lift straight up.

The grasp check itself is closed-loop and lives in the runner: after the `lift_check` phase it confirms in sim that
the object rose with the fingers (`checks.grasp_ok`), and only then runs the `lift` phase.
"""

from __future__ import annotations

import numpy as np

from .motion import APPROACH, CLOSED, FREE, OPEN, SLOW, Motion, Segment, above
from .plan import SEAT_FLOOR_MM, SLOW_ZONE_MM, PickPlan

CHECK_LIFT_MM = SLOW_ZONE_MM  # lift this far slowly, then verify the part came with the fingers


def subtask_name(plan: PickPlan) -> str:
    return f"pick:{plan.object_id}"


def pick(plan: PickPlan, start, gripper: float = OPEN, learn: bool = True) -> list[Segment]:
    m = Motion(np.asarray(start, dtype=float), gripper, "pick", subtask_name(plan), plan.object_id, learn=learn)
    g = plan.grasp_tcp
    m.enter()
    m.move("approach", above(m.pose, max(m.pose[2], plan.approach_z_mm)), FREE)
    m.move("approach", above(g, plan.approach_z_mm), FREE)
    m.grip("open", OPEN)  # always: every demo shows the fingers opening above the pad
    m.floor_z = g[2] - SEAT_FLOOR_MM
    m.contact_z = g[2]
    m.move("descend", g, APPROACH)
    m.grip("close", CLOSED)
    m.move("lift_check", above(g, g[2] + CHECK_LIFT_MM), SLOW)
    m.contact_z = None
    m.move("lift", above(g, plan.lift_z_mm), APPROACH)
    return m.segments
