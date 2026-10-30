"""Verdicts after a placement and after a recovery transfer.

After a placement the part is still in the fingers, so the depth size and the defect head (both judged at pre_pick on
the free-standing part) are not repeated: the fingers sit inside the part's footprint. What is checked is identity
(wrist classifier against the expected shape, FoundationPose fit), the seat and the tilt (placement.py).

`seated`/`pose_agrees` for the controller: both true only when the seat and tilt are certified; both false when either
is refuted (the controller then lifts 3 mm and reinserts once); unsure or a doubtful identity -> `uncertain` /
`ask_operator`, which verify_post_place turns into HOLD_ANDON before it looks at seated.
"""

from __future__ import annotations

from perception.cell.inspect import same_shape
from perception.port.judge import CloseupThresholds, Judgement
from perception.port.placement import Check, SeatReport
from perception.port.wrist import WristView


def _identity(expected: str, view: WristView, thresholds: CloseupThresholds) -> Judgement:
    reading = view.reading
    if view.fit_score < thresholds.pose_fit_min:
        return Judgement("uncertain", "ask_operator", None, False, True,
                         f"FoundationPose fit {view.fit_score:.3f} is below its minimum")
    if reading.p_identity < thresholds.classifier_min:
        return Judgement("uncertain", "ask_operator", None, False, True,
                         f"classifier {reading.object_id} at {reading.p_identity:.3f} is below its minimum")
    if not same_shape(reading.object_id, expected):
        return Judgement("wrong_part", "reject", reading.object_id, True, False, f"classifier says {reading.object_id}")
    return Judgement("ok", "use", expected, True, False, "classifier and FoundationPose agree on the shape")


def post_place_judgement(expected: str, view: WristView, seat: SeatReport,
                         thresholds: CloseupThresholds) -> tuple[Judgement, bool, bool]:
    identity = _identity(expected, view, thresholds)
    if identity.unsure:
        return identity, False, False
    if Check.REFUTED in (seat.seat, seat.tilt):
        return identity, False, False
    if seat.unsure:
        reason = (f"seat error {seat.seat_error_mm:+.2f} mm, tilt {seat.tilt_deg:.2f} deg; bound "
                  f"{seat.uncertainty.position_mm:.2f} mm / {seat.uncertainty.angle_deg:.2f} deg "
                  f"({seat.uncertainty.status.value}) cannot certify the 0.3 mm / 1 deg gates")
        return Judgement("uncertain", "ask_operator", identity.object_id, False, True, reason), False, False
    return identity, True, True


def transfer_judgement(expected: str, view: WristView, thresholds: CloseupThresholds) -> Judgement:
    """Identity of the released part at its recovery spot; the zone check is the caller's."""
    return _identity(expected, view, thresholds)
