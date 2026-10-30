"""One part's inspection verdict (contracts/inspection.schema.json) from the T27 models' outputs at the inspect stop.

Who decides what:
- the detector's class says foreign object or fallen part (`<object_id>__fallen`); a fallen part is rejected (user
  rule 2026-10-29) and its identity is the detector's, since the wrist models only learned installed-orientation
  close-ups;
- the wrist classifier decides identity; below its measured confidence the part goes to the operator;
- `checks_agree` records whether the classifier's identity and the advisory FoundationPose fit agree. Bases of
  one `base_type` are the same mesh (contracts/taxonomy.json: F01=F02, F04=F05=F06, 0.0 mm), so neither model can
  tell them apart and any base of the right type agrees (user 2026-10-29: same-shape bases are interchangeable);
- PatchCore is advisory: a score above its measured threshold asks the operator and never creates a reject verdict.
  Missing PatchCore or FoundationPose evidence is recorded as a warning, not made into motion authority.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from perception.taxonomy import FALLEN_SUFFIX, FOREIGN_LABELS

TAXONOMY = Path(__file__).resolve().parents[2] / "contracts" / "taxonomy.json"
BASE_TYPE = {o["object_id"]: o["base_type"] for o in json.loads(TAXONOMY.read_text())["objects"] if o.get("base_type")}


class InspectError(ValueError):
    """Missing or malformed evidence or thresholds; the part is not judged."""


def same_shape(a: str, b: str) -> bool:
    """Two ids name the same mesh: equal, or bases of one base type."""
    return a == b or (a in BASE_TYPE and BASE_TYPE.get(a) == BASE_TYPE.get(b))


@dataclass(frozen=True, slots=True)
class Thresholds:
    classifier_min: float
    pose_fit_min: float
    patchcore_max: dict[str, float]
    source: str

    @classmethod
    def load(cls, path: Path, parts: tuple[str, ...]) -> Thresholds:
        del parts
        raw = json.loads(path.read_text())
        patchcore = raw.get("patchcore_max", {})
        if not isinstance(patchcore, dict):
            raise InspectError(f"{path}: patchcore_max must be a mapping")
        values = [raw.get("classifier_min"), raw.get("pose_fit_min"), *patchcore.values()]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise InspectError(f"{path}: thresholds must be finite numbers")
        if not raw.get("source"):
            raise InspectError(f"{path}: thresholds must name the measurement they came from")
        return cls(float(raw["classifier_min"]), float(raw["pose_fit_min"]),
                   {k: float(v) for k, v in patchcore.items()}, str(raw["source"]))


@dataclass(frozen=True, slots=True)
class Evidence:
    detector_label: str
    classifier: tuple[str, float] | None      # top identity and its probability
    pose_fit: tuple[str, float] | None        # FoundationPose's best mesh and its fit score
    patchcore_score: float | None
    image_ref: str | None = None


def _result(
    instance_id: str,
    object_id: str | None,
    verdict: str,
    action: str,
    agree: bool,
    evidence: Evidence,
    warnings: tuple[str, ...] = (),
) -> dict:
    return {"instance_id": instance_id, "object_id": object_id, "verdict": verdict, "action": action,
            "checks_agree": agree,
            "evidence": {"classifier_score": evidence.classifier[1] if evidence.classifier else None,
                         "patchcore_score": evidence.patchcore_score, "depth_residual_mm": None,
                         "image_ref": evidence.image_ref},
            "warnings": list(warnings)}


def _foundationpose_warnings(identity: str, evidence: Evidence, thresholds: Thresholds) -> tuple[str, ...]:
    if evidence.pose_fit is None:
        return ("foundationpose_unavailable",)
    fit_id, fit = evidence.pose_fit
    if not same_shape(fit_id, identity):
        return ("foundationpose_identity_disagreement",)
    if fit < thresholds.pose_fit_min:
        return ("foundationpose_low_fit",)
    return ()


def inspect(instance_id: str, evidence: Evidence, thresholds: Thresholds) -> dict:
    label = evidence.detector_label
    if label in FOREIGN_LABELS:
        return _result(instance_id, None, "foreign", "reject", True, evidence)
    if label.endswith(FALLEN_SUFFIX):
        return _result(instance_id, label.removesuffix(FALLEN_SUFFIX), "flipped", "reject", True, evidence)
    if evidence.classifier is None:
        raise InspectError(f"{instance_id}: classifier evidence is required")
    identity, confidence = evidence.classifier
    if confidence < thresholds.classifier_min:
        return _result(instance_id, None, "uncertain", "ask_operator", False, evidence)
    warnings = _foundationpose_warnings(identity, evidence, thresholds)
    agree = not warnings
    if identity.startswith("F"):
        return _result(instance_id, identity, "ok", "use", agree, evidence, warnings)
    if evidence.patchcore_score is None:
        return _result(instance_id, identity, "ok", "use", agree, evidence, (*warnings, "patchcore_unavailable"))
    threshold = thresholds.patchcore_max.get(identity)
    if threshold is None:
        warning = "patchcore_threshold_unavailable"
        return _result(instance_id, identity, "ok", "use", agree, evidence, (*warnings, warning))
    if evidence.patchcore_score > threshold:
        warning = "patchcore_threshold_exceeded"
        return _result(instance_id, identity, "uncertain", "ask_operator", agree, evidence, (*warnings, warning))
    return _result(instance_id, identity, "ok", "use", agree, evidence, warnings)
