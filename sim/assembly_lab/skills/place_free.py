"""place_free(plan): set a removed good part upright on an empty remembered test-bed spot, or drop a defect /
unknown piece into the defect area.

The part is carried in its installed orientation, so setting it down at `spot_tcp` leaves it upright.
"""

from __future__ import annotations

import numpy as np

from .insert import FINE, FINE_ZONE_MM, PRESS, SEAT_APPROACH_MM
from .motion import APPROACH, OPEN, Motion, Segment, above
from .plan import SEAT_FLOOR_MM, FreePlacePlan
from .pull_out import carry

RELEASE_SETTLE_ROWS = 8
WORLD = "world"  # Mimic object_ref for fixed cell spots (test bed, defect area): the robot frame


def subtask_name(plan: FreePlacePlan) -> str:
    return f"place_free:{plan.object_id}->{plan.kind}"


def place_free(plan: FreePlacePlan, start, gripper: float, learn: bool = True) -> list[Segment]:
    """Carry, set down (test bed: the last 1 mm at PRESS speed, phase `touch_down`) or drop, release, retreat."""
    segs = set_down(plan, start, gripper, learn)
    last = segs[-1].rows[-1]
    return segs + release_and_retreat(plan, last[:6], float(last[6]), learn)


def set_down(plan: FreePlacePlan, start, gripper: float, learn: bool = True) -> list[Segment]:
    """Carry to above the spot and lower the part. On the test bed the last SEAT_APPROACH_MM goes at PRESS speed: a
    0.25 mm row onto the table measured 14-19 N single-step impulses (fa-gen1 scen1 fix_above M1_02)."""
    m = _motion(plan, start, gripper, learn)
    s = plan.spot_tcp
    carry(m, plan.via, above(s, plan.clear_z_mm))
    if plan.kind != "test_bed":
        m.move("drop", above(s, s[2] + plan.drop_mm), APPROACH)
        return m.segments
    m.move("set_down", above(s, s[2] + SEAT_APPROACH_MM), APPROACH)
    m.move("touch_down", above(s, s[2] + FINE_ZONE_MM), PRESS)
    m.move("touch_down", s, FINE)  # the table is as stiff as a seat: the last 0.3 mm in 0.02 mm rows
    return m.segments


def release_and_retreat(plan: FreePlacePlan, start, gripper: float, learn: bool = True) -> list[Segment]:
    """Open where the hand is (after a soft stop that may be above the planned spot) and go back up."""
    m = _motion(plan, start, gripper, learn)
    m.grip("release", OPEN, RELEASE_SETTLE_ROWS)
    m.move("retreat", above(m.pose, plan.clear_z_mm), APPROACH)
    return m.segments


def _motion(plan: FreePlacePlan, start, gripper: float, learn: bool) -> Motion:
    m = Motion(np.asarray(start, dtype=float), gripper, "place_free", subtask_name(plan), WORLD, learn=learn)
    m.floor_z = plan.spot_tcp[2] - SEAT_FLOOR_MM
    m.contact_z = plan.spot_tcp[2]
    return m
