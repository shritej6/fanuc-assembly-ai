"""place_free(plan): set a removed good part upright on an empty remembered test-bed spot, or drop a defect /
unknown piece into the defect area.

The part is carried in its installed orientation, so setting it down at `spot_tcp` leaves it upright.
"""

from __future__ import annotations

import numpy as np

from .motion import APPROACH, OPEN, Motion, Segment, above
from .plan import SEAT_FLOOR_MM, FreePlacePlan
from .pull_out import carry

RELEASE_SETTLE_ROWS = 4
WORLD = "world"  # Mimic object_ref for fixed cell spots (test bed, defect area): the robot frame


def subtask_name(plan: FreePlacePlan) -> str:
    return f"place_free:{plan.object_id}->{plan.kind}"


def place_free(plan: FreePlacePlan, start, gripper: float, learn: bool = True) -> list[Segment]:
    m = Motion(np.asarray(start, dtype=float), gripper, "place_free", subtask_name(plan), WORLD, learn=learn)
    s = plan.spot_tcp
    carry(m, plan.via, above(s, plan.clear_z_mm))
    m.floor_z = s[2] - SEAT_FLOOR_MM
    release_z = s[2] + plan.drop_mm
    m.contact_z = s[2]
    m.move("set_down" if plan.kind == "test_bed" else "drop", above(s, release_z), APPROACH)
    m.grip("release", OPEN, RELEASE_SETTLE_ROWS)
    m.move("retreat", above(s, plan.clear_z_mm), APPROACH)
    return m.segments
