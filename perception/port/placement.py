"""Post-place checks: right part, fully down and level, each certified, refuted or unsure.
Authored by Shritej Gunda."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from assembly.controller_ports import ExecutionTolerances, PerceptionTolerances
from perception.port.errors import PerceptionPortError
from perception.port.size_gate import Status

SEAT_GATE_MM = 0.3
TILT_GATE_DEG = 1.0
XY_GATE_MM = 0.3
YAW_GATE_DEG = 1.0
GATE_SOURCE = "sim/assembly_lab/INTERFACES.md Precision gates (feature/T12-mimic, CAD-derived)"


def precision_gate_tolerances() -> tuple[ExecutionTolerances, PerceptionTolerances]:
    """Return the CAD precision gates as controller tolerances."""
    return ExecutionTolerances(XY_GATE_MM, YAW_GATE_DEG), PerceptionTolerances(XY_GATE_MM, YAW_GATE_DEG)


@dataclass(frozen=True, slots=True)
class PoseUncertainty:
    """Bound on a FoundationPose result from one camera: position (mm) and angles (deg), with status and source."""

    position_mm: float
    angle_deg: float
    status: Status
    source: str

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) and v > 0 for v in (self.position_mm, self.angle_deg)):
            raise PerceptionPortError("pose uncertainty must be positive and finite")
        if not self.source:
            raise PerceptionPortError("pose uncertainty must name its source")


class Check(StrEnum):
    """Result of one placement check."""
    CERTIFIED = "certified"
    REFUTED = "refuted"
    UNSURE = "unsure"


def check(error: float, bound: float, gate: float, status: Status) -> Check:
    """Return the check result for an error against a gate."""
    if abs(error) - bound > gate:
        return Check.REFUTED
    if status is Status.MEASURED and abs(error) + bound <= gate:
        return Check.CERTIFIED
    return Check.UNSURE


def combined(a: PoseUncertainty, b: PoseUncertainty) -> PoseUncertainty:
    """Return the combined uncertainty of two poses."""
    status = Status.MEASURED if a.status is b.status is Status.MEASURED else Status.PRELIMINARY
    return PoseUncertainty(math.hypot(a.position_mm, b.position_mm), math.hypot(a.angle_deg, b.angle_deg), status,
                           f"quadrature of [{a.source}] and [{b.source}]")


@dataclass(frozen=True, slots=True)
class SeatReport:
    """Seat and tilt checks of one placed part."""
    seat_error_mm: float
    tilt_deg: float
    seat: Check
    tilt: Check
    uncertainty: PoseUncertainty

    @property
    def unsure(self) -> bool:
        """Return True when a check is unsure and none is refuted."""
        return Check.UNSURE in (self.seat, self.tilt) and Check.REFUTED not in (self.seat, self.tilt)

    @property
    def seated(self) -> bool:
        """Return True when the seat height is certified."""
        return self.seat is Check.CERTIFIED

    @property
    def level(self) -> bool:
        """Return True when the tilt is certified."""
        return self.tilt is Check.CERTIFIED


def seat_report(part_z_mm: float, base_z_mm: float, seated_origin_z_mm: float, tilt_deg: float,
                part_unc: PoseUncertainty, base_unc: PoseUncertainty) -> SeatReport:
    """Return the seat report for a placed part."""
    unc = combined(part_unc, base_unc)
    error = (part_z_mm - base_z_mm) - seated_origin_z_mm
    return SeatReport(error, tilt_deg, check(error, unc.position_mm, SEAT_GATE_MM, unc.status),
                      check(tilt_deg, part_unc.angle_deg, TILT_GATE_DEG, part_unc.status), unc)
