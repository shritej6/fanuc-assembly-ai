"""What each skill needs from the planner (agent 2), in mm / deg, UF0.

`planner.py` (agent 2) returns `GraspPlan` / `PlacePlan`; `from_planner` reads them by field name (attribute or dict
key) so the two modules stay decoupled. Every pose is a TCP pose `[x, y, z, w, p, r]`. Nothing here is guessed:
a field the planner does not give raises `PlanError`, except the documented human-style insert constants, which are
the user's decisions (docs/ROBOT_BRAIN.md section 2) and the CAD entry chamfer (cad/products/parameters.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GAP_CLOSED_MAX_MM = 4.5  # cad/grasp_frames.json gripper_gap_mm: fingers fully closed
GAP_OPEN_MIN_MM = 16.5  # fully open
ENTRY_CHAMFER_MM = 1.5  # parameters.ENTRY_CHAMFER
CLEARANCE_PER_SIDE_MM = 0.5  # parameters.LOCATING_CLEARANCE_PER_SIDE
INSERT_TILT_DEG = 3.0  # user: come in tilted about 3 deg
INSERT_TWIST_DEG = 2.0  # user: twist ±2 deg
PULL_WIGGLE_DEG = 1.0  # user: pull out with a ±1 deg wiggle
SEAT_FLOOR_MM = 1.0  # user: never command below the seated height - 1 mm
SLOW_ZONE_MM = 5.0  # user: slow last 5 mm
RETRY_LIFT_MM = 3.0  # user: if not seated, lift 3 mm, re-measure, re-align, retry once
PUSH_MM = 0.5  # light push: target this far below the CAD seated TCP height (must stay <= SEAT_FLOOR_MM)


class PlanError(ValueError):
    """The planner output is missing a field or breaks a hard rule (pad width, push depth)."""


def _get(src, name, default=None, required=True):
    val = src.get(name, default) if isinstance(src, dict) else getattr(src, name, default)
    if val is None and required:
        raise PlanError(f"planner output has no '{name}'")
    return val


def _pose(src, name, required=True):
    val = _get(src, name, required=required)
    if val is None:
        return None
    arr = np.asarray(val, dtype=float)
    if arr.shape != (6,):
        raise PlanError(f"'{name}' must be [x, y, z, w, p, r], got shape {arr.shape}")
    return arr


def _waypoints(src, name) -> list[np.ndarray]:
    pts = _get(src, name, default=[], required=False) or []
    return [np.asarray(p, dtype=float) for p in pts]


@dataclass(frozen=True)
class PickPlan:
    object_id: str
    grasp_tcp: np.ndarray  # TCP at the grip pad, fingers closed here
    pad_width_mm: float  # CAD pad width across the closing axis
    approach_z_mm: float  # top of the clear vertical corridor above the pad
    lift_z_mm: float  # straight-up lift target after the grasp
    via: list[np.ndarray] = field(default_factory=list)  # planner's clear path after the lift
    grip_index: int = 0  # which CAD grip pad (cad/grasp_frames.json order), for the plan_ids observation

    def __post_init__(self):
        if not GAP_CLOSED_MAX_MM < self.pad_width_mm < GAP_OPEN_MIN_MM:
            raise PlanError(f"{self.object_id}: pad {self.pad_width_mm} mm outside the finger range "
                            f"({GAP_CLOSED_MAX_MM}, {GAP_OPEN_MIN_MM}); closing would crush or miss it")
        if self.approach_z_mm <= self.grasp_tcp[2] or self.lift_z_mm <= self.grasp_tcp[2]:
            raise PlanError(f"{self.object_id}: approach and lift must be above the grasp")


@dataclass(frozen=True)
class InsertPlan:
    object_id: str
    base_id: str
    seated_tcp: np.ndarray  # TCP when the part is fully seated (CAD)
    seated_object_pose: np.ndarray  # object pose when fully seated (CAD), for the seated check
    engage_mm: float  # travel along the insert axis from first chamfer contact to seated (peg / socket depth)
    approach_z_mm: float  # clear height above the base for the final approach
    chamfer_mm: float = ENTRY_CHAMFER_MM  # lead-in width beyond the clearance (objects.yaml capture - clearance)
    clearance_mm: float = CLEARANCE_PER_SIDE_MM
    lead_dir_deg: float = 0.0  # base-frame heading of the edge that touches the chamfer first
    via: list[np.ndarray] = field(default_factory=list)  # planner's clear carry path to above the base
    hang_mm: float = 0.0  # TCP height above the part's lowest point (the tilt pivots about the TCP)
    yaw_tol_deg: float | None = None  # objects.yaml insertion.yaw_tol_deg (None: yaw-free fit)
    half_extent_mm: float = 0.0  # half the part's largest footprint side: how far the tilt drops the lead edge

    @property
    def capture_mm(self) -> float:
        return self.clearance_mm + self.chamfer_mm

    def __post_init__(self):
        if PUSH_MM > SEAT_FLOOR_MM:
            raise PlanError("light push deeper than the seated-height floor")
        if self.engage_mm <= 0 or self.chamfer_mm < 0 or self.hang_mm < 0:
            raise PlanError(f"{self.object_id}: engage must be positive, chamfer and hang not negative")
        if self.approach_z_mm <= self.seated_tcp[2] + self.engage_mm:
            raise PlanError(f"{self.object_id}: approach height is inside the engagement")


@dataclass(frozen=True)
class PullOutPlan:
    object_id: str
    grasp_tcp: np.ndarray  # TCP on the installed part's grip pad
    pad_width_mm: float
    engage_mm: float  # how far it travels before it is free of its pegs / socket
    clear_z_mm: float  # height that clears every other installed part
    via: list[np.ndarray] = field(default_factory=list)
    grip_index: int = 0

    def __post_init__(self):
        PickPlan(self.object_id, self.grasp_tcp, self.pad_width_mm, self.clear_z_mm, self.clear_z_mm)


@dataclass(frozen=True)
class FreePlacePlan:
    object_id: str
    spot_tcp: np.ndarray  # TCP when the part stands on the spot, upright (installed orientation)
    kind: str  # "test_bed" (remembered empty spot, set down gently) or "defect" (drop into the defect area)
    clear_z_mm: float
    drop_mm: float = 0.0  # defect area only: release height above the resting TCP height
    via: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        if self.kind not in ("test_bed", "defect"):
            raise PlanError(f"unknown free-place kind {self.kind!r}")
        if self.kind == "test_bed" and self.drop_mm:
            raise PlanError("a good part is set down, never dropped")


def pick_from_planner(g) -> PickPlan:
    return PickPlan(_get(g, "object_id"), _pose(g, "grasp_tcp"), float(_get(g, "pad_width_mm")),
                    float(_get(g, "approach_z_mm")), float(_get(g, "lift_z_mm")), _waypoints(g, "via"),
                    int(_get(g, "grip_index", 0)))


def insert_from_planner(p) -> InsertPlan:
    return InsertPlan(
        _get(p, "object_id"), _get(p, "base_id"), _pose(p, "seated_tcp"), _pose(p, "seated_object_pose"),
        float(_get(p, "engage_mm")), float(_get(p, "approach_z_mm")),
        float(_get(p, "chamfer_mm", ENTRY_CHAMFER_MM)), float(_get(p, "clearance_mm", CLEARANCE_PER_SIDE_MM)),
        float(_get(p, "lead_dir_deg", 0.0)), _waypoints(p, "via"))


def pull_out_from_planner(p) -> PullOutPlan:
    return PullOutPlan(_get(p, "object_id"), _pose(p, "grasp_tcp"), float(_get(p, "pad_width_mm")),
                       float(_get(p, "engage_mm")), float(_get(p, "clear_z_mm")), _waypoints(p, "via"),
                       int(_get(p, "grip_index", 0)))


def free_place_from_planner(p) -> FreePlacePlan:
    return FreePlacePlan(_get(p, "object_id"), _pose(p, "spot_tcp"), _get(p, "kind"), float(_get(p, "clear_z_mm")),
                         float(_get(p, "drop_mm", 0.0)), _waypoints(p, "via"))
