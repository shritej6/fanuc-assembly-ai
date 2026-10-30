"""insert(plan): human-style insertion (user decision 2026-10-29, docs/ROBOT_BRAIN.md section 2).

carry along the planner's clear path -> tilt ~3 deg -> touch one edge into the entry chamfer -> level out while sliding
in -> ±2 deg twist -> light push to the CAD seated height. Never below seated - 1 mm; slow in the last 5 mm.

The seated check and the single retry are closed-loop (runner): `retry_entry` lifts 3 mm and re-enters with the
correction the runner measured; `release_and_retreat` lets go and leaves straight up.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import tilted
from .motion import APPROACH, CLOSED, OPEN, SLOW, Motion, Segment, above
from .plan import INSERT_TILT_DEG, INSERT_TWIST_DEG, PUSH_MM, RETRY_LIFT_MM, SEAT_FLOOR_MM, InsertPlan
from .pull_out import carry

PUSH_HOLD_ROWS = 3
ENTRY_TOP_MM = 3.0  # start the tilted descent this far above first chamfer contact


def subtask_name(plan: InsertPlan) -> str:
    return f"insert:{plan.object_id}@{plan.base_id}"


def _motion(plan: InsertPlan, start, gripper: float, learn: bool) -> Motion:
    m = Motion(np.asarray(start, dtype=float), gripper, "insert", subtask_name(plan), plan.base_id, learn=learn)
    m.floor_z = plan.seated_tcp[2] - SEAT_FLOOR_MM
    m.contact_z = plan.seated_tcp[2]
    return m


def entry_shift(plan: InsertPlan) -> float:
    """TCP offset toward the lead edge so the part's bottom lands mid-chamfer (inside the capture), corrected for the
    tilt pivoting about the TCP (which swings the bottom back by hang * sin(tilt)). No lead-in: no offset, the tilt
    alone puts one edge down first."""
    bottom = 0.5 * (plan.clearance_mm + plan.capture_mm) if plan.chamfer_mm > 0 else 0.0
    return bottom + plan.hang_mm * math.sin(math.radians(INSERT_TILT_DEG))


def _entry_heights(plan: InsertPlan) -> dict[str, float]:
    """Monotone TCP heights for the entry phases, never below the seated height before the push."""
    zs, e, c = plan.seated_tcp[2], plan.engage_mm, plan.chamfer_mm
    touch = zs + max(e - 0.5 * c, 0.0)
    level = min(zs + 0.5 * e, touch)
    return {"top": zs + e + ENTRY_TOP_MM, "touch": touch, "level": level,
            "twist1": min(zs + 0.35 * e, level), "twist2": min(zs + 0.15 * e, level), "push": zs - PUSH_MM}


def _target(plan: InsertPlan, offset) -> np.ndarray:
    """Seated TCP corrected by the runner's measured offset (dx, dy mm, dyaw deg)."""
    t = plan.seated_tcp.copy()
    t[0] += offset[0]
    t[1] += offset[1]
    t[5] += offset[2]
    return t


def _entry(m: Motion, plan: InsertPlan, offset) -> None:
    tgt = _target(plan, offset)
    h = _entry_heights(plan)
    lead = plan.lead_dir_deg
    shift = entry_shift(plan)
    u = np.array([math.cos(math.radians(lead)), math.sin(math.radians(lead)), 0.0, 0, 0, 0])
    lean = tilted(above(tgt, h["top"]) + shift * u, lead, INSERT_TILT_DEG)
    m.move("tilt", lean, APPROACH)
    m.move("touch", above(lean, h["touch"]), SLOW)
    m.move("level", above(tgt, h["level"]), SLOW)
    m.move("twist", tilted(above(tgt, h["twist1"]), lead, 0.0, +INSERT_TWIST_DEG), SLOW)
    m.move("twist", tilted(above(tgt, h["twist2"]), lead, 0.0, -INSERT_TWIST_DEG), SLOW)
    m.move("twist", above(tgt, plan.seated_tcp[2]), SLOW)
    m.move("push", above(tgt, h["push"]), SLOW)
    m.hold("push", PUSH_HOLD_ROWS)


def insert(plan: InsertPlan, start, gripper: float, offset=(0.0, 0.0, 0.0), learn: bool = True) -> list[Segment]:
    """Carry to above the base and make the first insertion attempt (ends pushed, still gripping)."""
    m = _motion(plan, start, gripper, learn)
    m.contact_z = None
    carry(m, plan.via, above(_target(plan, offset), plan.approach_z_mm))
    m.contact_z = plan.seated_tcp[2]
    m.move("approach", above(_target(plan, offset), _entry_heights(plan)["top"]), APPROACH)
    _entry(m, plan, offset)
    return m.segments


def retry_lift(plan: InsertPlan, start, gripper: float, learn: bool = True) -> list[Segment]:
    """Not seated: lift 3 mm straight up so the runner can re-measure."""
    m = _motion(plan, start, gripper, learn)
    m.move("retry_lift", above(m.pose, m.pose[2] + RETRY_LIFT_MM), SLOW)
    return m.segments


def retry_entry(plan: InsertPlan, start, gripper: float, offset, learn: bool = True) -> list[Segment]:
    """Re-align with the measured correction and enter again (the one allowed retry)."""
    m = _motion(plan, start, gripper, learn)
    top = _entry_heights(plan)["top"]
    if m.pose[2] < top:  # leave the engagement before tilting again
        m.move("retry_clear", above(m.pose, top), SLOW)
    _entry(m, plan, offset)
    return m.segments


def release_and_retreat(plan: InsertPlan, start, learn: bool = True) -> list[Segment]:
    m = _motion(plan, start, CLOSED, learn)
    m.grip("release", OPEN)
    m.move("retreat", above(m.pose, plan.approach_z_mm), APPROACH)
    m.contact_z = None
    return m.segments

