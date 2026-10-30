"""Verdicts after a placement and after a recovery transfer.

After a placement the part is still in the fingers, so the depth size and the defect head (both judged at pre_pick on
the free-standing part) are not repeated: the fingers sit inside the part's footprint. What is checked is classifier
identity, the seat and the tilt (placement.py); FoundationPose fit is recorded only as an advisory.

FoundationPose seat, tilt, and recovery-zone results are diagnostics only. Even a high-fit result asks the operator,
so this port never causes an automatic release, reinsert, or recovery transfer.
"""

from __future__ import annotations

from perception.cell.inspect import same_shape
from perception.port.judge import CloseupThresholds, Judgement
from perception.port.placement import SeatReport
from perception.port.wrist import WristView


def _identity(expected: str, view: WristView, thresholds: CloseupThresholds) -> Judgement:
    reading = view.reading
    fit_agrees = view.fit_score >= thresholds.pose_fit_min
    warning = "" if fit_agrees else f"; FoundationPose advisory: fit {view.fit_score:.3f} is below its minimum"
    if reading.p_identity < thresholds.classifier_min:
        return Judgement("uncertain", "ask_operator", None, False, True,
                         f"classifier {reading.object_id} at {reading.p_identity:.3f} is below its minimum{warning}")
    if not same_shape(reading.object_id, expected):
        return Judgement(
            "wrong_part", "reject", reading.object_id, fit_agrees, False,
            f"classifier says {reading.object_id}{warning}",
        )
    return Judgement("ok", "use", expected, fit_agrees, False, f"classifier identifies the expected shape{warning}")


def post_place_judgement(expected: str, view: WristView, seat: SeatReport,
                         thresholds: CloseupThresholds) -> tuple[Judgement, bool, bool]:
    identity = _identity(expected, view, thresholds)
    if identity.unsure or identity.action != "use":
        return identity, False, False
    reason = (
        f"{identity.reason}; FoundationPose seat error {seat.seat_error_mm:+.2f} mm and "
        f"tilt {seat.tilt_deg:.2f} deg are advisory; operator placement confirmation is required"
    )
    return Judgement("uncertain", "ask_operator", identity.object_id, False, True, reason), False, False


def transfer_judgement(expected: str, view: WristView, thresholds: CloseupThresholds) -> Judgement:
    """Keep FoundationPose recovery-zone evidence advisory pending operator confirmation."""
    identity = _identity(expected, view, thresholds)
    if identity.unsure or identity.action != "use":
        return identity
    reason = f"{identity.reason}; FoundationPose recovery-zone evidence is advisory"
    return Judgement("uncertain", "ask_operator", identity.object_id, False, True, reason)
