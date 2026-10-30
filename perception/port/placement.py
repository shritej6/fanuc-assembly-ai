"""post_place decisions: right part, fully down, not crooked -- each one certified, refuted, or unsure.

Gates (sim/assembly_lab/INTERFACES.md "Precision gates", derived from the CAD: 0.5 mm clearance per side, pegs about
15.6 mm from the part centre): seated height within +/-0.3 mm, tilt within 1 deg, xy within 0.3 mm, yaw within 1 deg.

A gate is only *certified* when the measurement error bound is MEASURED and |error| + bound <= gate. It is *refuted*
when |error| - bound > gate (the part is clearly high or crooked, e.g. riding on a peg), which is safe to act on with a
PRELIMINARY bound because the controller's answer (lift 3 mm, re-measure, reinsert once) is the recoverable one.
Everything in between is unsure, which goes to the operator. With today's PRELIMINARY wrist numbers (0.8 mm = 2.2 px)
the +/-0.3 mm seat cannot be certified from the camera, and the evidence says so instead of passing.

Seated height = FoundationPose z of the placed part's installed-local origin minus the measured z of its base's origin,
compared with objects.yaml `seated_origin_z_mm`. Both poses come from the same calibration, so a shared offset cancels;
their bounds add in quadrature.
"""

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
    """The CAD precision gates as controller tolerances. They are acceptance limits, not sensor accuracies: the same
    PerceptionTolerances also gates base drift in collision checks, so a pose noisier than 0.3 mm will hold the cell."""
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
    CERTIFIED = "certified"
    REFUTED = "refuted"
    UNSURE = "unsure"


def check(error: float, bound: float, gate: float, status: Status) -> Check:
    if abs(error) - bound > gate:
        return Check.REFUTED
    if status is Status.MEASURED and abs(error) + bound <= gate:
        return Check.CERTIFIED
    return Check.UNSURE


def combined(a: PoseUncertainty, b: PoseUncertainty) -> PoseUncertainty:
    status = Status.MEASURED if a.status is b.status is Status.MEASURED else Status.PRELIMINARY
    return PoseUncertainty(math.hypot(a.position_mm, b.position_mm), math.hypot(a.angle_deg, b.angle_deg), status,
                           f"quadrature of [{a.source}] and [{b.source}]")


@dataclass(frozen=True, slots=True)
class SeatReport:
    seat_error_mm: float
    tilt_deg: float
    seat: Check
    tilt: Check
    uncertainty: PoseUncertainty

    @property
    def unsure(self) -> bool:
        return Check.UNSURE in (self.seat, self.tilt) and Check.REFUTED not in (self.seat, self.tilt)

    @property
    def seated(self) -> bool:
        return self.seat is Check.CERTIFIED

    @property
    def level(self) -> bool:
        return self.tilt is Check.CERTIFIED


def seat_report(part_z_mm: float, base_z_mm: float, seated_origin_z_mm: float, tilt_deg: float,
                part_unc: PoseUncertainty, base_unc: PoseUncertainty) -> SeatReport:
    unc = combined(part_unc, base_unc)
    error = (part_z_mm - base_z_mm) - seated_origin_z_mm
    return SeatReport(error, tilt_deg, check(error, unc.position_mm, SEAT_GATE_MM, unc.status),
                      check(tilt_deg, part_unc.angle_deg, TILT_GATE_DEG, part_unc.status), unc)
