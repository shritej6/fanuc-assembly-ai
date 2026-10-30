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
from .motion import APPROACH, CLOSED, OPEN, SLOW, Motion, Segment, Speed, above
from .plan import INSERT_TILT_DEG, INSERT_TWIST_DEG, PUSH_MM, RETRY_LIFT_MM, SEAT_FLOOR_MM, InsertPlan
from .pull_out import carry

PUSH_HOLD_ROWS = 6
RELIEVE_HOLD_ROWS = 3  # let the unloaded part settle in the fingers before they open
SEAT_APPROACH_MM = 1.0  # the last 1 mm to the seat and the push go at PRESS speed
PRESS = Speed(0.05, 0.05)  # 0.5 mm/s: run7 measured the seat force jump past 15 N within one 0.25 mm row (15.3, 15.8 N
#                            from below the 9 N soft stop), so each row now adds at most a fifth of that
FINE = Speed(0.02, 0.02)  # 0.2 mm/s over the seat: ~260 N/mm adds ~5 N per row, so the 2 N stop ends the press near
#                         7 N worst case (run10 0.05 mm rows: 6-11.6 N kept, 15.4 / 17.0 N discarded)
FINE_ZONE_MM = 0.3  # FINE from this far above the CAD seated height: run10 first seat contact came at -0.085 to
#                     -0.099 mm and already above 2 N, so a contact trigger is one row too late
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
    """Monotone TCP heights for the entry phases, never below the seated height before the push.

    The part is LEVEL before it enters the bore (agent 2 measured that a held tilt does not level by itself and that a
    tilted part jams a 0.5 mm clearance fit): the lead edge touches mid-chamfer, the part levels inside the chamfer
    (at the bore mouth when there is no lead-in), twists there, and only then goes straight down to the seat.
    """
    zs, e, c = plan.seated_tcp[2], plan.engage_mm, plan.chamfer_mm
    drop = plan.half_extent_mm * math.sin(math.radians(INSERT_TILT_DEG))  # lead edge below the bottom centre
    level = zs + max(e - 0.5 * c, 0.0)
    touch = level + drop
    return {"top": max(zs + e + ENTRY_TOP_MM, touch + ENTRY_TOP_MM), "touch": touch, "level": level,
            "push": zs - PUSH_MM}


def _target(plan: InsertPlan, offset) -> np.ndarray:
    """Seated TCP corrected by the runner's measured offset (dx, dy mm, dyaw deg[, w, p deg that level the held
    part])."""
    t = plan.seated_tcp.copy()
    t[0] += offset[0]
    t[1] += offset[1]
    t[5] += offset[2]
    if len(offset) > 3:
        t[3], t[4] = offset[3], offset[4]
    return t


def twisted_about(pose, pivot_xy, deg: float) -> np.ndarray:
    """The TCP turned by `deg` about the vertical line through `pivot_xy` (the part's own insert axis), so the twist
    turns the part in place. Turning about the TCP instead swings a part gripped on an edge pad sideways: run3
    measured M1_01 (pad 21 mm off centre) left 0.3-0.5 mm off after the +-2 deg twist, beyond its 0.5 mm clearance,
    and it then jammed tilted 1.5-2 deg."""
    out = tilted(pose, 0.0, 0.0, deg)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    d = np.asarray(pose[:2], dtype=float) - np.asarray(pivot_xy, dtype=float)
    out[0], out[1] = pivot_xy[0] + c * d[0] - s * d[1], pivot_xy[1] + s * d[0] + c * d[1]
    return out


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
    pivot = plan.seated_object_pose[:2]
    m.move("twist", twisted_about(above(tgt, h["level"]), pivot, +INSERT_TWIST_DEG), SLOW)
    m.move("twist", twisted_about(above(tgt, h["level"]), pivot, -INSERT_TWIST_DEG), SLOW)
    m.move("twist", above(tgt, h["level"]), SLOW)
    m.move("slide", above(tgt, plan.seated_tcp[2] + SEAT_APPROACH_MM), SLOW)
    m.move("slide", above(tgt, plan.seated_tcp[2] + FINE_ZONE_MM), PRESS)
    m.move("slide", above(tgt, plan.seated_tcp[2]), FINE)
    m.move("push", above(tgt, h["push"]), FINE)
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


def retry_clear(plan: InsertPlan, start, gripper: float, learn: bool = True) -> list[Segment]:
    """After the 3 mm lift: up to the entry top so the part hangs free in the fingers (no base contact) before the
    runner re-measures how it is held."""
    m = _motion(plan, start, gripper, learn)
    top = _entry_heights(plan)["top"]
    if m.pose[2] < top:
        m.move("retry_clear", above(m.pose, top), SLOW)
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
    """Unload, then let go, then leave straight up. The push leaves the TCP up to 0.5 mm below the seat with the part
    preloaded in the fingers; opening under that preload dragged seated parts 0.2-0.33 mm in Isaac (run9: M3_01,
    M5_01, M6_01), so the TCP first backs off to the seated height (fingers at their grasp height, no axial load)."""
    m = _motion(plan, start, CLOSED, learn)
    if m.pose[2] < plan.seated_tcp[2]:
        m.move("relieve", above(m.pose, plan.seated_tcp[2]), PRESS)
        m.hold("relieve", RELIEVE_HOLD_ROWS)
    m.grip("release", OPEN)
    m.move("retreat", above(m.pose, plan.approach_z_mm), APPROACH)
    m.contact_z = None
    return m.segments

