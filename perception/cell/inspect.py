"""One part's inspection verdict (contracts/inspection.schema.json) from the T27 models' outputs at the inspect stop.

Who decides what:
- the detector's class says foreign object or fallen part (`<object_id>__fallen`); a fallen part is rejected (user
  rule 2026-10-29) and its identity is the detector's, since the wrist models only learned installed-orientation
  close-ups;
- the wrist classifier decides identity; below its measured confidence the part goes to the operator;
- `checks_agree` is the contract's poka-yoke: the classifier's identity and the FoundationPose fit agree;
- PatchCore, against the part's measured threshold, decides defects.
Thresholds only come from a measured file (`Thresholds.load`); a part without a measured PatchCore threshold fails
the load, so a gap shows up before the slot and not as a silent pass during it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from perception.taxonomy import FALLEN_SUFFIX, FOREIGN_LABELS


class InspectError(ValueError):
    """Missing or malformed evidence or thresholds; the part is not judged."""


@dataclass(frozen=True, slots=True)
class Thresholds:
    classifier_min: float
    pose_fit_min: float
    patchcore_max: dict[str, float]
    source: str

    @classmethod
    def load(cls, path: Path, parts: tuple[str, ...]) -> Thresholds:
        raw = json.loads(path.read_text())
        missing = sorted(set(parts) - set(raw.get("patchcore_max", {})))
        if missing:
            raise InspectError(f"{path}: no measured PatchCore threshold for {missing}")
        values = [raw.get("classifier_min"), raw.get("pose_fit_min"), *raw["patchcore_max"].values()]
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values):
            raise InspectError(f"{path}: thresholds must be finite numbers")
        if not raw.get("source"):
            raise InspectError(f"{path}: thresholds must name the measurement they came from")
        return cls(float(raw["classifier_min"]), float(raw["pose_fit_min"]),
                   {k: float(v) for k, v in raw["patchcore_max"].items()}, str(raw["source"]))


@dataclass(frozen=True, slots=True)
class Evidence:
    detector_label: str
    classifier: tuple[str, float] | None      # top identity and its probability
    pose_fit: tuple[str, float] | None        # FoundationPose's best mesh and its fit score
    patchcore_score: float | None
    image_ref: str | None = None


def _result(instance_id: str, object_id: str | None, verdict: str, action: str, agree: bool,
            evidence: Evidence) -> dict:
    return {"instance_id": instance_id, "object_id": object_id, "verdict": verdict, "action": action,
            "checks_agree": agree,
            "evidence": {"classifier_score": evidence.classifier[1] if evidence.classifier else None,
                         "patchcore_score": evidence.patchcore_score, "depth_residual_mm": None,
                         "image_ref": evidence.image_ref}}


def inspect(instance_id: str, evidence: Evidence, thresholds: Thresholds) -> dict:
    label = evidence.detector_label
    if label in FOREIGN_LABELS:
        return _result(instance_id, None, "foreign", "reject", True, evidence)
    if label.endswith(FALLEN_SUFFIX):
        return _result(instance_id, label.removesuffix(FALLEN_SUFFIX), "flipped", "reject", True, evidence)
    if evidence.classifier is None or evidence.pose_fit is None:
        raise InspectError(f"{instance_id}: classifier and FoundationPose evidence are both required")
    identity, confidence = evidence.classifier
    if confidence < thresholds.classifier_min:
        return _result(instance_id, None, "uncertain", "ask_operator", False, evidence)
    fit_id, fit = evidence.pose_fit
    if fit_id != identity or fit < thresholds.pose_fit_min:
        return _result(instance_id, identity, "uncertain", "ask_operator", False, evidence)
    if identity.startswith("F"):
        return _result(instance_id, identity, "ok", "use", True, evidence)
    if evidence.patchcore_score is None:
        raise InspectError(f"{instance_id}: no PatchCore score for {identity}")
    if evidence.patchcore_score > thresholds.patchcore_max[identity]:
        return _result(instance_id, identity, "broken", "reject", True, evidence)
    return _result(instance_id, identity, "ok", "use", True, evidence)
