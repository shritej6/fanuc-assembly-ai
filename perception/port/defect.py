"""The wrist defect head's careful decision band and its contract mapping.
Authored by Shritej Gunda."""

from __future__ import annotations

import math
from enum import StrEnum

from perception.port.errors import ModelOutputError

SECOND_LOOK_FROM = 0.5
REJECT_FROM = 0.994
SOURCE = "feature/T27-newset-training d869bad perception/classifier/defect_cutoff.py (validation split, job 291308)"


class DefectCall(StrEnum):
    """Decision of the wrist defect head."""
    USE = "use"
    SECOND_LOOK = "second_look"
    REJECT = "reject"


def defect_call(p_defect: float) -> DefectCall:
    """Return the decision for a defect probability."""
    if isinstance(p_defect, bool) or not isinstance(p_defect, (int, float)) or not math.isfinite(p_defect) \
            or not 0.0 <= p_defect <= 1.0:
        raise ModelOutputError(f"p_defect must be a probability, got {p_defect!r}")
    if p_defect >= REJECT_FROM:
        return DefectCall.REJECT
    return DefectCall.SECOND_LOOK if p_defect >= SECOND_LOOK_FROM else DefectCall.USE
