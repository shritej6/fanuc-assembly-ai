"""One wrist close-up -> the contract inspection verdict plus the typed next step (ROBOT_BRAIN section 3).

Inputs are already-validated readings: the wrist classifier (identity + defect head), advisory FoundationPose output,
and the size gate. Order of the authoritative rules (first match wins):

1. Size fits none of the 30 objects and the classifier is not confident -> `foreign` / reject.
2. Defect head p >= 0.994 -> `print_defect` / reject (DEFECT_HEAD_INSPECTION.md mapping, no contract edit).
3. Classifier below its measured minimum -> unsure.
4. Classifier names another shape: size agrees with that shape -> `wrong_part` / reject; otherwise unsure.
5. Size contradicts the expected object, fits nothing, or could not be measured -> unsure (LOOKALIKES.md s3: missing
   depth is "unsure").
6. Defect head in the careful band 0.5 <= p < 0.994 -> unsure.
7. Otherwise `ok` / use.

"Unsure" is `uncertain` / `ask_operator` with checks_agree false, which the controller's verify_pre_pick turns into
HOLD_ANDON. The port's CloseupReport says whether that is a SECOND_LOOK (first view) or the ANDON (second view).
Low-fit or non-upright FoundationPose output only clears `checks_agree` and adds a warning to the reason.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from perception.cell.inspect import same_shape
from perception.port.backends import WristReading
from perception.port.defect import DefectCall, defect_call
from perception.port.errors import PerceptionPortError
from perception.port.size_gate import SizeOutcome, SizeVerdict


class NextStep(StrEnum):
    USE = "use"
    REJECT = "reject"
    SECOND_LOOK = "second_look"
    ANDON = "andon"


@dataclass(frozen=True, slots=True)
class CloseupThresholds:
    """Measured acceptance minimums for the wrist classifier and the FoundationPose fit; no defaults exist."""

    classifier_min: float
    pose_fit_min: float
    source: str

    def __post_init__(self) -> None:
        for name in ("classifier_min", "pose_fit_min"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or not 0.0 <= value <= 1.0:
                raise PerceptionPortError(f"{name} must be a probability, got {value!r}")
        if not isinstance(self.source, str) or not self.source.strip():
            raise PerceptionPortError("close-up thresholds must name the measurement they came from")

    @classmethod
    def load(cls, path: Path) -> CloseupThresholds:
        """The T28 thresholds file format (classifier_min, pose_fit_min, source)."""
        try:
            raw = json.loads(path.read_text())
            return cls(raw["classifier_min"], raw["pose_fit_min"], raw["source"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise PerceptionPortError(f"{path}: unreadable close-up thresholds ({error})") from error


@dataclass(frozen=True, slots=True)
class Judgement:
    verdict: str
    action: str
    object_id: str | None
    checks_agree: bool
    unsure: bool
    reason: str


def _decided(verdict: str, action: str, object_id: str | None, reason: str) -> Judgement:
    return Judgement(verdict, action, object_id, True, False, reason)


def _unsure(object_id: str | None, reason: str) -> Judgement:
    return Judgement("uncertain", "ask_operator", object_id, False, True, reason)


def _identity_rules(expected: str, reading: WristReading, size: SizeVerdict, sure_id: str | None) -> Judgement | None:
    if sure_id is None:
        return _unsure(None, f"classifier {reading.object_id} at {reading.p_identity:.3f} is below its minimum")
    if not same_shape(sure_id, expected):
        if any(same_shape(sure_id, m) for m in size.matching) and size.outcome is not SizeOutcome.CONFIRMS:
            return _decided("wrong_part", "reject", sure_id, f"classifier and size both say {sure_id}")
        return _unsure(sure_id, f"classifier says {sure_id}, size check supports {expected}")
    if size.outcome is not SizeOutcome.CONFIRMS and size.outcome is not SizeOutcome.CANNOT_SEPARATE:
        return _unsure(sure_id, f"size check {size.outcome.value} for {expected} (matching {list(size.matching)})")
    return None


def _authoritative_judgement(
    expected: str,
    reading: WristReading,
    size: SizeVerdict,
    thresholds: CloseupThresholds,
) -> Judgement:
    confident = reading.p_identity >= thresholds.classifier_min
    if size.outcome is SizeOutcome.MATCHES_NOTHING and not confident:
        return _decided("foreign", "reject", None, "size fits none of the 30 objects and the classifier is unsure")
    sure_id = reading.object_id if confident else None
    if sure_id is not None and same_shape(sure_id, expected):
        sure_id = expected      # F01=F02, F04=F05=F06 are one mesh: any base of the type is the expected one
    call = defect_call(reading.p_defect)
    if call is DefectCall.REJECT:
        return _decided("print_defect", "reject", sure_id or expected, f"defect head p={reading.p_defect:.4f}")
    identity = _identity_rules(expected, reading, size, sure_id)
    if identity is not None:
        return identity
    if call is DefectCall.SECOND_LOOK:
        return _unsure(sure_id, f"defect head p={reading.p_defect:.4f} is in the careful band [0.5, 0.994)")
    return _decided("ok", "use", sure_id, "identity, size and defect head agree")


def _foundationpose_advisories(fit_score: float, side: str, thresholds: CloseupThresholds) -> tuple[str, ...]:
    warnings: list[str] = []
    if fit_score < thresholds.pose_fit_min:
        warnings.append(f"fit {fit_score:.3f} is below its minimum")
    if side != "upright":
        warnings.append(f"reported resting side {side}")
    return tuple(warnings)


def _with_advisories(judgement: Judgement, advisories: tuple[str, ...]) -> Judgement:
    if not advisories:
        return judgement
    detail = "; ".join(advisories)
    return replace(judgement, checks_agree=False, reason=f"{judgement.reason}; FoundationPose advisory: {detail}")


def judge(expected: str, reading: WristReading, fit_score: float, side: str, size: SizeVerdict,
          thresholds: CloseupThresholds) -> Judgement:
    judgement = _authoritative_judgement(expected, reading, size, thresholds)
    return _with_advisories(judgement, _foundationpose_advisories(fit_score, side, thresholds))
