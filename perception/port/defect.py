"""The wrist defect head's "careful" band (user decision 2026-10-30) and its contract mapping.

Mirrors `SECOND_LOOK_FROM`, `REJECT_FROM` and `decision()` of perception/classifier/defect_cutoff.py on
feature/T27-newset-training commit d869bad (not merged into this branch). Both cutoffs were read off the VALIDATION
split of model 5696b36c (Rosie job 291308); docs/DEFECT_HEAD_INSPECTION.md on that branch:

| p_defect | action |
|---|---|
| >= 0.994 | reject: defect area, take a spare -> verdict `print_defect`, action `reject` |
| 0.5 <= p < 0.994 | second close-up from another angle; still unsure: Andon with both pictures |
| < 0.5 | the verdict comes from the other checks |

At 0.994 the head is mainly a crack detector (test split: cracked M4_04 recall 0.867, all defect types 0.378, good
false reject 0.0066 overall but 0.0977 on set U). Those are synthetic-split numbers; the real-cell rate is unmeasured.
"""

from __future__ import annotations

import math
from enum import StrEnum

from perception.port.errors import ModelOutputError

SECOND_LOOK_FROM = 0.5
REJECT_FROM = 0.994
SOURCE = "feature/T27-newset-training d869bad perception/classifier/defect_cutoff.py (validation split, job 291308)"


class DefectCall(StrEnum):
    USE = "use"
    SECOND_LOOK = "second_look"
    REJECT = "reject"


def defect_call(p_defect: float) -> DefectCall:
    if isinstance(p_defect, bool) or not isinstance(p_defect, (int, float)) or not math.isfinite(p_defect) \
            or not 0.0 <= p_defect <= 1.0:
        raise ModelOutputError(f"p_defect must be a probability, got {p_defect!r}")
    if p_defect >= REJECT_FROM:
        return DefectCall.REJECT
    return DefectCall.SECOND_LOOK if p_defect >= SECOND_LOOK_FROM else DefectCall.USE
