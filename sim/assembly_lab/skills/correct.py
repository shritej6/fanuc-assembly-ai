"""Correction demos (user decision 2026-10-31): start a step from a WRONG pose near the target and fix it.

GR00T checkpoint-3000 grasped M1_01 well in the sim eval but arrived above F01 ~8 mm off and ~90 deg turned, tipped
the part on the base and dropped it: the nominal demos only show one smooth path, never how to correct. Here the
staged error (part held above the base, or the open fingers above the grip pad, off in x, y, height, yaw and tilt)
is reached with masked rows (`learn=False`, left out by export_lerobot), and the demo that is trained on starts at the
wrong pose: line up at the current height, then the normal descent (pick) or human-style entry (insert).

The staging comes first in the episode, so the exported episode has no gap in the middle (actions are absolute
act_tcp rows: a gap would teach a jump to every frame whose 16-step chunk crosses it).
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import tilted
from .insert import _entry, _entry_heights, _motion, _target
from .motion import APPROACH, CLOSED, FREE, OPEN, SLOW, Motion, Segment, above
from .pick import CHECK_LIFT_MM, subtask_name
from .plan import SEAT_FLOOR_MM, InsertPlan, PickPlan
from .pull_out import carry

# PRELIMINARY ranges (user asked for side offset, turn angle, tilt and grasp height): the eval miss was 8 mm / 90 deg
INSERT_XY_MM = (3.0, 10.0)
INSERT_YAW_DEG = (5.0, 90.0)
INSERT_TILT_DEG = 4.0  # staged lean; the 6 deg tool limit leaves room for the level correction
INSERT_DZ_MM = (3.0, 20.0)  # above the entry top: the held part never starts lower than the entry
GRASP_XY_MM = (3.0, 8.0)
GRASP_YAW_DEG = (0.0, 15.0)
GRASP_DZ_MM = (8.0, 25.0)  # above the grasp pose: open fingers clear the part top


def _xy(rng: np.random.Generator, lo: float, hi: float) -> tuple[float, float]:
    a, d = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(lo, hi)
    return d * math.cos(a), d * math.sin(a)


def _signed(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.choice((-1.0, 1.0)) * rng.uniform(lo, hi))


def staged_insert_pose(plan: InsertPlan, offset, rng: np.random.Generator) -> np.ndarray:
    """A wrong pose above the base: the corrected entry target moved in x, y, raised, turned and leaned."""
    good = above(_target(plan, offset), _entry_heights(plan)["top"] + rng.uniform(*INSERT_DZ_MM))
    dx, dy = _xy(rng, *INSERT_XY_MM)
    good[0] += dx
    good[1] += dy
    good[5] += _signed(rng, *INSERT_YAW_DEG)
    return tilted(good, rng.uniform(0.0, 360.0), rng.uniform(0.0, INSERT_TILT_DEG))


def staged_grasp_pose(plan: PickPlan, rng: np.random.Generator) -> np.ndarray:
    """A wrong pose above the grip pad: off in x, y, turned, higher than the grasp."""
    g = above(plan.grasp_tcp, plan.grasp_tcp[2] + rng.uniform(*GRASP_DZ_MM))
    dx, dy = _xy(rng, *GRASP_XY_MM)
    g[0] += dx
    g[1] += dy
    g[5] += _signed(rng, *GRASP_YAW_DEG)
    return g


def stage_near_base(plan: InsertPlan, start, gripper: float, staged) -> list[Segment]:
    """Masked: carry the held part along the planner's path, then down to the staged wrong pose."""
    m = _motion(plan, start, gripper, learn=False)
    m.contact_z = None
    carry(m, plan.via, above(staged, plan.approach_z_mm))
    m.move("stage", staged, APPROACH)
    return m.segments


def stage_near_pad(plan: PickPlan, start, staged) -> list[Segment]:
    """Masked: from home, open the fingers and go to the staged wrong pose above the pad."""
    m = Motion(np.asarray(start, dtype=float), OPEN, "pick", subtask_name(plan), plan.object_id, learn=False)
    m.enter()
    m.move("stage", above(m.pose, max(m.pose[2], plan.approach_z_mm)), FREE)
    m.move("stage", above(staged, plan.approach_z_mm), FREE)
    m.grip("stage", OPEN)
    m.move("stage", staged, APPROACH)
    return m.segments


def insert_from_near(plan: InsertPlan, start, gripper: float, offset) -> list[Segment]:
    """Learned: line the held part up over the base at the current height, then the normal entry and push."""
    m = _motion(plan, start, gripper, learn=True)
    tgt = _target(plan, offset)
    top = _entry_heights(plan)["top"]
    m.move("align", above(tgt, max(m.pose[2], top)), APPROACH)
    m.move("approach", above(tgt, top), APPROACH)
    _entry(m, plan, offset)
    return m.segments


def pick_from_near(plan: PickPlan, start) -> list[Segment]:
    """Learned: line the open fingers up over the pad at the current height, descend, close, check lift, lift."""
    m = Motion(np.asarray(start, dtype=float), OPEN, "pick", subtask_name(plan), plan.object_id, learn=True)
    g = plan.grasp_tcp
    m.move("align", above(g, m.pose[2]), APPROACH)
    m.floor_z = g[2] - SEAT_FLOOR_MM
    m.contact_z = g[2]
    m.move("descend", g, APPROACH)
    m.grip("close", CLOSED)
    m.move("lift_check", above(g, g[2] + CHECK_LIFT_MM), SLOW)
    m.contact_z = None
    m.move("lift", above(g, plan.lift_z_mm), APPROACH)
    return m.segments
