"""Look-alike size gate: measure the part first, then match the measured size to the CAD (user rule 2026-10-29).

Look-alikes differ in dimensions (sim/assembly_lab/objects/LOOKALIKES.md on feature/T12-mimic). The measurement is
the depth points on the part, put into the part's installed-local frame with the FoundationPose fit:

- footprint: the extent (max - min) of the points along local x and y, compared sorted (so a 90 deg yaw ambiguity
  cannot swap the axes) with the CAD bbox size. Stray depth clusters are already removed by the connected-group
  selection; a percentile extent would shrink a uniform face by the trimmed fraction (4 % for 2-98) and is not used.
  Flying pixels on edges would inflate the extent; that error is part of the (PRELIMINARY) bound;
- height: the depth points are binned on a HEIGHT_CELL_MM grid in the part frame and each cell's median z taken
  (about 30 wrist pixels per cell at 150 mm, so per-pixel noise drops about five-fold); the height is the median of
  the highest TOP_FRACTION of those cell medians above the support plane. A plain high percentile of raw pixels would
  read about two noise sigmas high (measured wrist noise below); this reads about 0.3 sigma high on a flat face;
- top cues: where objects.yaml lists a region whose top height differs between two look-alikes (e.g. the M1_02 post
  is 17 mm, the M2_02 post 16 mm), the median local z of the points in that region is compared with both heights.
  Regions narrower than MIN_CUE_SIDE_MM are not used: LOOKALIKES.md section 3 says stereo depth smooths the 1 mm lock
  ribs and 0.8 mm insert markers away, so those pairs are read from colour by the wrist classifier.

A dimension agrees with a CAD value when it is within the stated uncertainty, which is a bound and carries its status.
Nothing here is calibrated on the cell: the wrist numbers below are PRELIMINARY until measured on real parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from perception.cell.inspect import same_shape
from perception.port.cad import CadCatalog, TopCue
from perception.port.errors import PerceptionPortError

MIN_POINTS = 30          # fewer depth points than this on a part is a hole in the depth map, not a size
MIN_CUE_POINTS = 12
MIN_CUE_SIDE_MM = 2.0
TOP_FRACTION = 0.2
HEIGHT_CELL_MM = 2.0
MIN_CELL_POINTS = 5


class Status(StrEnum):
    PRELIMINARY = "PRELIMINARY"
    MEASURED = "MEASURED"


@dataclass(frozen=True, slots=True)
class SizeUncertainty:
    """Bound on a measured length (xy) and height (z), with where the numbers come from."""

    xy_mm: float
    z_mm: float
    status: Status
    source: str

    def __post_init__(self) -> None:
        if not (np.isfinite(self.xy_mm) and np.isfinite(self.z_mm) and self.xy_mm > 0 and self.z_mm > 0):
            raise PerceptionPortError("size uncertainty must be positive and finite")
        if not self.source:
            raise PerceptionPortError("size uncertainty must name its source")


WRIST_150MM_PRELIMINARY = SizeUncertainty(
    0.8, 1.3, Status.PRELIMINARY,
    "xy: 0.8 mm = 2.2 px wrist colour at 150 mm (LOOKALIKES.md s3, geometry only). z: wrist depth per-pixel noise "
    "MEASURED on the real captures r20261029-995508 (evaluation only) at 364 mm range: local-plane residual std median "
    "7.64 mm over 80 patches of 16 frames, scaled by range squared to 150 mm (x0.170) = 1.3 mm. The scaling is an "
    "extrapolation; nothing was measured at 150 mm.")
TABLE_DEPTH_PRELIMINARY = SizeUncertainty(
    6.42, 2.31, Status.PRELIMINARY,
    "xy: two table-depth pixels on the table (3.21 mm each, perception/cell/localize.py); z: measured table-depth "
    "rms_plane_mm 2.31 (robot/cell/calibration/cameras.json). The xy bound is assumed, not measured.")


class SizeOutcome(StrEnum):
    CONFIRMS = "confirms"                # the expected object fits and no look-alike does
    CANNOT_SEPARATE = "cannot_separate"  # the expected object and a look-alike both fit: identity rests on colour
    CONTRADICTS = "contradicts"          # the expected object does not fit, another of the 30 does
    MATCHES_NOTHING = "matches_nothing"  # none of the 30 objects fits: foreign piece or wrong-size part
    UNMEASURED = "unmeasured"            # too few depth points


@dataclass(frozen=True, slots=True)
class SizeMeasurement:
    footprint_mm: tuple[float, float]    # sorted (short, long)
    height_mm: float
    points: int
    cue_z_mm: dict[tuple[float, float, float, float], float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SizeVerdict:
    outcome: SizeOutcome
    expected: str
    matching: tuple[str, ...]            # objects whose CAD fits the measurement (after cues)
    residual_mm: float | None            # largest |measured - CAD| of the expected object
    uncertainty: SizeUncertainty


def _top_height(points_local: np.ndarray, z_uf0: np.ndarray) -> float:
    cells = np.floor(points_local[:, :2] / HEIGHT_CELL_MM).astype(np.int64)
    _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    medians = np.array([np.median(z_uf0[inverse.ravel() == k]) for k in np.flatnonzero(counts >= MIN_CELL_POINTS)])
    if len(medians) == 0:
        medians = np.array([np.median(z_uf0)])
    top = np.sort(medians)[int(len(medians) * (1.0 - TOP_FRACTION)):]
    return float(np.median(top))


def measure(points_local: np.ndarray, points_uf0: np.ndarray, support_z_uf0: float,
            cues: tuple[TopCue, ...] = ()) -> SizeMeasurement | None:
    """Size of the part from its depth points (same rows in both frames); None when there are too few points."""
    if len(points_local) < MIN_POINTS:
        return None
    lo, hi = points_local[:, :2].min(axis=0), points_local[:, :2].max(axis=0)
    short, long_ = sorted(float(v) for v in hi - lo)
    height = _top_height(points_local, points_uf0[:, 2]) - support_z_uf0
    cue_z = {}
    for cue in cues:
        x0, y0, x1, y1 = cue.region_xy_mm
        if min(x1 - x0, y1 - y0) < MIN_CUE_SIDE_MM:
            continue
        inside = (points_local[:, 0] >= x0) & (points_local[:, 0] <= x1) & \
                 (points_local[:, 1] >= y0) & (points_local[:, 1] <= y1)
        if inside.sum() >= MIN_CUE_POINTS:
            cue_z[cue.region_xy_mm] = float(np.median(points_local[inside, 2]))
    return SizeMeasurement((short, long_), height, len(points_local), cue_z)


def measure_unposed(points_uf0: np.ndarray, support_z_uf0: float) -> SizeMeasurement | None:
    """Size without a pose (scan): footprint along the points' principal axes in x, y; height above the support."""
    if len(points_uf0) < MIN_POINTS:
        return None
    xy = points_uf0[:, :2] - points_uf0[:, :2].mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov(xy.T))
    return measure(np.column_stack([xy @ vectors, points_uf0[:, 2]]), points_uf0, support_z_uf0)


def _residual(measured: SizeMeasurement, size: tuple[float, float, float]) -> tuple[float, float]:
    """(largest footprint error, height error) against one CAD bbox size."""
    cad_short, cad_long = sorted(size[:2])
    xy = max(abs(measured.footprint_mm[0] - cad_short), abs(measured.footprint_mm[1] - cad_long))
    return xy, abs(measured.height_mm - size[2])


def _fits(measured: SizeMeasurement, size: tuple[float, float, float], unc: SizeUncertainty) -> bool:
    xy, z = _residual(measured, size)
    return xy <= unc.xy_mm and z <= unc.z_mm


def _cue_rules_out(measured: SizeMeasurement, cues: tuple[TopCue, ...], unc: SizeUncertainty) -> bool | None:
    """True when a measured cue height matches the other object and not this one; False when the reverse; else None."""
    votes = set()
    for cue in cues:
        z = measured.cue_z_mm.get(cue.region_xy_mm)
        if z is None:
            continue
        here, other = abs(z - cue.top_z_here_mm) <= unc.z_mm, abs(z - cue.top_z_other_mm) <= unc.z_mm
        if here != other:
            votes.add(other)
    return votes.pop() if len(votes) == 1 else None


def _apply_cues(expected: str, matching: set[str], measured: SizeMeasurement, cad: CadCatalog,
                unc: SizeUncertainty) -> set[str]:
    """Drop look-alikes (or the expected object) that a measured top cue rules out."""
    if expected not in matching:
        return matching
    for other in [m for m in matching if m != expected and not same_shape(m, expected)]:
        verdict = _cue_rules_out(measured, cad.get(expected).cues.get(other, ()), unc)
        if verdict is False:
            matching.discard(other)
        elif verdict is True:
            matching.discard(expected)
            return matching
    return matching


def gate(expected: str, measured: SizeMeasurement | None, cad: CadCatalog, unc: SizeUncertainty) -> SizeVerdict:
    """Match a measured size to the expected object, its look-alikes, its size twins and the rest of the 30."""
    if measured is None:
        return SizeVerdict(SizeOutcome.UNMEASURED, expected, (), None, unc)
    matching = {oid for oid, obj in cad.objects.items() if _fits(measured, obj.size_mm, unc)}
    matching = _apply_cues(expected, matching, measured, cad, unc)
    residual = max(_residual(measured, cad.get(expected).size_mm))
    rivals = {c for c in cad.candidates_for(expected) if not same_shape(c, expected)} & matching
    if expected in matching:
        outcome = SizeOutcome.CANNOT_SEPARATE if rivals else SizeOutcome.CONFIRMS
    else:
        outcome = SizeOutcome.CONTRADICTS if matching else SizeOutcome.MATCHES_NOTHING
    return SizeVerdict(outcome, expected, tuple(sorted(matching)), residual, unc)
