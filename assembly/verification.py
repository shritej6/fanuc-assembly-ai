"""Fail-closed perception gates and bounded insertion recovery decisions.
Authored by Shritej Gunda.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import cast


class VerificationInputError(ValueError):
    """Raised when perception evidence is malformed or outside safe ranges."""


class VerificationOutcome(str, Enum):
    ACCEPT = "accept"
    REJECT_TO_SPARE = "reject_to_spare"
    RECOVER = "recover"
    HOLD_ANDON = "hold_andon"


class VerificationReason(str, Enum):
    VERIFIED = "verified"
    IDENTITY_MISMATCH = "identity_mismatch"
    POSE_UNVERIFIED = "pose_unverified"
    UNSAFE_ORIENTATION = "unsafe_orientation"
    INSPECTION_REJECTED = "inspection_rejected"
    CHECK_DISAGREEMENT = "check_disagreement"
    NOT_SEATED = "not_seated"
    PLACEMENT_DISAGREEMENT = "placement_disagreement"
    PATCHCORE_WARNING = "patchcore_warning"


class RecoveryAction(str, Enum):
    RETRY_INSERTION = "retry_insertion"
    DISMANTLE_REBUILD = "dismantle_rebuild"
    HOLD_ANDON = "hold_andon"


class RecoveryOperation(str, Enum):
    LIFT = "lift"
    RE_YAW = "re_yaw"
    REINSERT = "reinsert"


@dataclass(frozen=True, slots=True)
class VerificationDecision:
    outcome: VerificationOutcome
    reason: VerificationReason
    instance_id: str
    object_id: str | None
    advisories: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class VerificationThresholds:
    """Measured acceptance limits supplied by the deployed model bundle."""

    min_detection_score: float
    min_pose_fit_score: float
    min_classifier_score: float
    max_patchcore_score: float
    max_depth_residual_mm: float

    def __post_init__(self) -> None:
        for field in (
            "min_detection_score",
            "min_pose_fit_score",
            "min_classifier_score",
            "max_patchcore_score",
        ):
            _require_number(getattr(self, field), field, minimum=0, maximum=1)
        _require_number(self.max_depth_residual_mm, "max_depth_residual_mm", minimum=0)


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    retry_number: int | None
    operations: tuple[RecoveryOperation, ...]
    dismantle_order: tuple[str, ...]
    rebuild_order: tuple[str, ...]
    replacement_object_id: str | None


_ACTIONS_BY_VERDICT = {
    "ok": frozenset({"use"}),
    "flipped": frozenset({"rework", "reject"}),
    "broken": frozenset({"reject"}),
    "missing_feature": frozenset({"reject"}),
    "print_defect": frozenset({"reject"}),
    "missing_layer": frozenset({"reject"}),
    "wrong_part": frozenset({"reject"}),
    "foreign": frozenset({"reject"}),
    "uncertain": frozenset({"ask_operator", "reject"}),
}
_PART_IDS = frozenset(f"M{product}_{step:02d}" for product in range(1, 7) for step in range(1, 5))
_OBJECT_IDS = _PART_IDS | frozenset(f"F{base:02d}" for base in range(1, 7))
_CAMERAS = frozenset({"table", "side", "wrist"})
_POSE_SOURCES = frozenset({"detector", "foundationpose", "none"})
_ORIENTATIONS = frozenset({"upright", "flipped", "side", "unknown"})
_ACTIONS = frozenset({"use", "rework", "reject", "ask_operator"})
_DETECTION_REQUIRED = ("instance_id", "camera", "label", "score", "bbox_px")
_INSPECTION_REQUIRED = ("instance_id", "verdict", "action", "checks_agree")


def _input_error(path: str, requirement: str) -> None:
    raise VerificationInputError(f"{path} {requirement}")


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _input_error(path, "must be a mapping")
    return cast(Mapping[str, object], value)


def _require_fields(payload: Mapping[str, object], fields: tuple[str, ...], path: str) -> None:
    for field in fields:
        if field not in payload:
            _input_error(f"{path}.{field}", "is required")


def _require_string(value: object, path: str) -> None:
    if not isinstance(value, str) or not value:
        _input_error(path, "must be a non-empty string")


def _require_enum(value: object, allowed: frozenset[str], path: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        _input_error(path, f"must be one of {sorted(allowed)}")


def _require_number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        _input_error(path, "must be a finite number")
    if minimum is not None and value < minimum:
        _input_error(path, f"must be at least {minimum}")
    if maximum is not None and value > maximum:
        _input_error(path, f"must be at most {maximum}")


def _validate_object_id(value: object, path: str) -> None:
    if value is not None and (not isinstance(value, str) or value not in _OBJECT_IDS):
        _input_error(path, "must be a known object_id or null")


def _validate_bbox(value: object) -> None:
    if not isinstance(value, list) or len(value) != 4:
        _input_error("detection.bbox_px", "must contain four numbers")
    for coordinate in value:
        _require_number(coordinate, "detection.bbox_px", minimum=0)
    x0, y0, x1, y1 = value
    if x0 >= x1 or y0 >= y1:
        _input_error("detection.bbox_px", "must have positive width and height")


def _validate_pose(value: object) -> None:
    if value is None:
        return
    pose = _require_mapping(value, "detection.pose")
    if set(pose) != {"x", "y", "z", "yaw"}:
        _input_error("detection.pose", "must contain only x, y, z, and yaw")
    for coordinate in pose.values():
        _require_number(coordinate, "detection.pose")


def _validate_detection(value: object) -> Mapping[str, object]:
    detection = _require_mapping(value, "detection")
    _require_fields(detection, _DETECTION_REQUIRED, "detection")
    _require_string(detection["instance_id"], "detection.instance_id")
    _require_enum(detection["camera"], _CAMERAS, "detection.camera")
    _require_string(detection["label"], "detection.label")
    _require_number(detection["score"], "detection.score", minimum=0, maximum=1)
    _validate_bbox(detection["bbox_px"])
    _validate_object_id(detection.get("object_id"), "detection.object_id")
    _validate_pose(detection.get("pose"))
    if "pose_source" in detection:
        _require_enum(detection["pose_source"], _POSE_SOURCES, "detection.pose_source")
    if detection.get("pose_fit_score") is not None:
        _require_number(detection["pose_fit_score"], "detection.pose_fit_score", minimum=0, maximum=1)
    if "resting_side" in detection:
        _require_enum(detection["resting_side"], _ORIENTATIONS, "detection.resting_side")
    return detection


def _validate_inspection_evidence(value: object) -> None:
    evidence = _require_mapping(value, "inspection.evidence")
    for field in ("classifier_score", "patchcore_score"):
        if evidence.get(field) is not None:
            _require_number(evidence[field], f"inspection.evidence.{field}", minimum=0, maximum=1)
    if evidence.get("depth_residual_mm") is not None:
        _require_number(evidence["depth_residual_mm"], "inspection.evidence.depth_residual_mm", minimum=0)
    image_ref = evidence.get("image_ref")
    if image_ref is not None and not isinstance(image_ref, str):
        _input_error("inspection.evidence.image_ref", "must be a string or null")


def _validate_inspection(value: object) -> Mapping[str, object]:
    inspection = _require_mapping(value, "inspection")
    _require_fields(inspection, _INSPECTION_REQUIRED, "inspection")
    _require_string(inspection["instance_id"], "inspection.instance_id")
    _validate_object_id(inspection.get("object_id"), "inspection.object_id")
    _require_enum(inspection["verdict"], frozenset(_ACTIONS_BY_VERDICT), "inspection.verdict")
    _require_enum(inspection["action"], _ACTIONS, "inspection.action")
    if not isinstance(inspection["checks_agree"], bool):
        _input_error("inspection.checks_agree", "must be a boolean")
    if "evidence" in inspection:
        _validate_inspection_evidence(inspection["evidence"])
    return inspection


def _validate_inputs(
    expected_object_id: object,
    expected_instance_id: object,
    detection: object,
    inspection: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    if not isinstance(expected_object_id, str) or expected_object_id not in _PART_IDS:
        _input_error("expected_object_id", "must be a known part object_id")
    _require_string(expected_instance_id, "expected_instance_id")
    return _validate_detection(detection), _validate_inspection(inspection)


def _validate_installed_prefix(
    expected_object_id: str,
    value: object,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        _input_error("installed_object_ids", "must be a sequence of part object_ids")
    installed = tuple(value)
    prefix = expected_object_id.split("_", maxsplit=1)[0]
    recipe = tuple(f"{prefix}_{step:02d}" for step in range(1, 5))
    if installed != recipe[: len(installed)]:
        _input_error("installed_object_ids", "must be an ordered recipe prefix")
    return cast(tuple[str, ...], installed)


def _decision(
    outcome: VerificationOutcome,
    reason: VerificationReason,
    detection: Mapping[str, object],
    advisories: tuple[str, ...] = (),
) -> VerificationDecision:
    return VerificationDecision(
        outcome=outcome,
        reason=reason,
        instance_id=cast(str, detection["instance_id"]),
        object_id=cast(str | None, detection.get("object_id")),
        advisories=advisories,
    )


def _checks_disagree(
    detection: Mapping[str, object],
    inspection: Mapping[str, object],
) -> bool:
    verdict = cast(str, inspection.get("verdict"))
    return (
        detection.get("instance_id") != inspection.get("instance_id")
        or detection.get("object_id") != inspection.get("object_id")
        or inspection.get("action") not in _ACTIONS_BY_VERDICT[verdict]
    )


def _has_foundationpose(detection: Mapping[str, object]) -> bool:
    return (
        detection.get("pose_source") == "foundationpose"
        and isinstance(detection.get("pose"), Mapping)
        and detection.get("pose_fit_score") is not None
    )


def _advisories(
    detection: Mapping[str, object],
    inspection: Mapping[str, object],
    thresholds: VerificationThresholds,
    *,
    pose_agrees: bool = True,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if cast(float, detection["score"]) < thresholds.min_detection_score:
        warnings.append("detection_low_confidence")
    if not _has_foundationpose(detection):
        warnings.append("foundationpose_unavailable")
    elif cast(float, detection["pose_fit_score"]) < thresholds.min_pose_fit_score:
        warnings.append("foundationpose_low_fit")
    if inspection.get("checks_agree") is not True or not pose_agrees:
        warnings.append("foundationpose_disagreement")
    if detection.get("resting_side") != "upright":
        warnings.append("foundationpose_orientation")
    return tuple(warnings)


def _patchcore_exceeds(inspection: Mapping[str, object], thresholds: VerificationThresholds) -> bool:
    evidence = inspection.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    score = evidence.get("patchcore_score")
    return isinstance(score, (int, float)) and not isinstance(score, bool) and score > thresholds.max_patchcore_score


def verify_pre_pick(
    expected_object_id: str,
    detection: Mapping[str, object],
    inspection: Mapping[str, object],
    *,
    expected_instance_id: str,
    thresholds: VerificationThresholds,
) -> VerificationDecision:
    """Accept on semantic identity; model scores and FoundationPose are advisory."""

    detection, inspection = _validate_inputs(expected_object_id, expected_instance_id, detection, inspection)
    advisories = _advisories(detection, inspection, thresholds)
    if _checks_disagree(detection, inspection):
        return _decision(
            VerificationOutcome.HOLD_ANDON, VerificationReason.CHECK_DISAGREEMENT, detection, advisories
        )
    if detection.get("instance_id") != expected_instance_id:
        return _decision(VerificationOutcome.HOLD_ANDON, VerificationReason.IDENTITY_MISMATCH, detection, advisories)
    if detection.get("object_id") != expected_object_id:
        return _decision(
            VerificationOutcome.REJECT_TO_SPARE, VerificationReason.IDENTITY_MISMATCH, detection, advisories
        )
    if inspection.get("verdict") != "ok" and inspection.get("action") != "ask_operator":
        return _decision(
            VerificationOutcome.REJECT_TO_SPARE, VerificationReason.INSPECTION_REJECTED, detection, advisories
        )
    if _patchcore_exceeds(inspection, thresholds):
        return _decision(VerificationOutcome.HOLD_ANDON, VerificationReason.PATCHCORE_WARNING, detection, advisories)
    if inspection.get("action") == "ask_operator":
        return _decision(
            VerificationOutcome.HOLD_ANDON, VerificationReason.INSPECTION_REJECTED, detection, advisories
        )
    return _decision(VerificationOutcome.ACCEPT, VerificationReason.VERIFIED, detection, advisories)


def verify_post_place(
    expected_object_id: str,
    detection: Mapping[str, object],
    inspection: Mapping[str, object],
    *,
    expected_instance_id: str,
    thresholds: VerificationThresholds,
    seated: bool,
    pose_agrees: bool,
) -> VerificationDecision:
    """Require independent identity and placement agreement after insertion."""

    detection, inspection = _validate_inputs(expected_object_id, expected_instance_id, detection, inspection)
    if not isinstance(seated, bool) or not isinstance(pose_agrees, bool):
        _input_error("post_place", "seated and pose_agrees must be booleans")
    advisories = _advisories(detection, inspection, thresholds, pose_agrees=pose_agrees)
    if _checks_disagree(detection, inspection):
        return _decision(
            VerificationOutcome.HOLD_ANDON, VerificationReason.CHECK_DISAGREEMENT, detection, advisories
        )
    if detection.get("instance_id") != expected_instance_id:
        return _decision(VerificationOutcome.HOLD_ANDON, VerificationReason.IDENTITY_MISMATCH, detection, advisories)
    if detection.get("object_id") != expected_object_id:
        return _decision(VerificationOutcome.RECOVER, VerificationReason.IDENTITY_MISMATCH, detection, advisories)
    if inspection.get("verdict") != "ok" and inspection.get("action") != "ask_operator":
        return _decision(VerificationOutcome.RECOVER, VerificationReason.INSPECTION_REJECTED, detection, advisories)
    if _patchcore_exceeds(inspection, thresholds):
        return _decision(VerificationOutcome.HOLD_ANDON, VerificationReason.PATCHCORE_WARNING, detection, advisories)
    if inspection.get("action") == "ask_operator":
        return _decision(
            VerificationOutcome.HOLD_ANDON, VerificationReason.INSPECTION_REJECTED, detection, advisories
        )
    if not seated:
        return _decision(VerificationOutcome.RECOVER, VerificationReason.NOT_SEATED, detection, advisories)
    return _decision(VerificationOutcome.ACCEPT, VerificationReason.VERIFIED, detection, advisories)


def verify_recovery_transfer(
    expected_object_id: str,
    detection: Mapping[str, object],
    inspection: Mapping[str, object],
    *,
    expected_instance_id: str,
    thresholds: VerificationThresholds,
    pose_agrees: bool,
    allow_rejected: bool,
) -> VerificationDecision:
    """Verify exact identity and a calibrated recovery-zone transfer."""
    detection, inspection = _validate_inputs(expected_object_id, expected_instance_id, detection, inspection)
    if not isinstance(pose_agrees, bool) or not isinstance(allow_rejected, bool):
        _input_error("recovery_transfer", "pose_agrees and allow_rejected must be booleans")
    advisories = _advisories(detection, inspection, thresholds)
    exact_identity = all(
        payload.get("instance_id") == expected_instance_id and payload.get("object_id") == expected_object_id
        for payload in (detection, inspection)
    )
    if not exact_identity:
        return _decision(VerificationOutcome.HOLD_ANDON, VerificationReason.IDENTITY_MISMATCH, detection, advisories)
    if not _transfer_classifier_passes(inspection, thresholds):
        return _decision(VerificationOutcome.HOLD_ANDON, VerificationReason.POSE_UNVERIFIED, detection, advisories)
    if not pose_agrees:
        return _decision(
            VerificationOutcome.HOLD_ANDON, VerificationReason.PLACEMENT_DISAGREEMENT, detection, advisories
        )
    blocker = _transfer_inspection_blocker(detection, inspection, allow_rejected)
    if blocker is not None:
        return _decision(VerificationOutcome.HOLD_ANDON, blocker, detection, advisories)
    if not allow_rejected and _patchcore_exceeds(inspection, thresholds):
        return _decision(VerificationOutcome.HOLD_ANDON, VerificationReason.PATCHCORE_WARNING, detection, advisories)
    return _decision(VerificationOutcome.ACCEPT, VerificationReason.VERIFIED, detection, advisories)


def _transfer_inspection_blocker(
    detection: Mapping[str, object],
    inspection: Mapping[str, object],
    allow_rejected: bool,
) -> VerificationReason | None:
    verdict = inspection.get("verdict")
    if allow_rejected:
        good_disagrees = verdict == "ok" and _checks_disagree(detection, inspection)
        rejected_without_action = verdict != "ok" and inspection.get("action") != "reject"
        return VerificationReason.INSPECTION_REJECTED if good_disagrees or rejected_without_action else None
    if verdict != "ok" or inspection.get("action") != "use":
        return VerificationReason.INSPECTION_REJECTED
    if _checks_disagree(detection, inspection):
        return VerificationReason.CHECK_DISAGREEMENT
    return None


def _transfer_classifier_passes(
    inspection: Mapping[str, object],
    thresholds: VerificationThresholds,
) -> bool:
    evidence = inspection.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    score = evidence.get("classifier_score")
    return isinstance(score, (int, float)) and not isinstance(score, bool) and score >= thresholds.min_classifier_score


def decide_recovery(
    expected_object_id: str,
    failed_attempts: int,
    verified_spare_available: bool,
    *,
    installed_object_ids: Sequence[str],
) -> RecoveryDecision:
    """Bound insertion recovery to two retries before a spare rebuild."""

    if not isinstance(expected_object_id, str) or expected_object_id not in _PART_IDS:
        _input_error("expected_object_id", "must be a known part object_id")
    installed = _validate_installed_prefix(expected_object_id, installed_object_ids)
    if isinstance(failed_attempts, bool) or not isinstance(failed_attempts, int) or failed_attempts < 0:
        _input_error("failed_attempts", "must be a non-negative integer")
    if not isinstance(verified_spare_available, bool):
        _input_error("verified_spare_available", "must be a boolean")
    if failed_attempts < 2:
        operations = (RecoveryOperation.LIFT, RecoveryOperation.RE_YAW, RecoveryOperation.REINSERT)
        return RecoveryDecision(RecoveryAction.RETRY_INSERTION, failed_attempts + 1, operations, (), (), None)
    if not verified_spare_available:
        return RecoveryDecision(RecoveryAction.HOLD_ANDON, None, (), (), (), None)
    return RecoveryDecision(
        RecoveryAction.DISMANTLE_REBUILD,
        None,
        (),
        tuple(reversed(installed)),
        installed,
        expected_object_id,
    )
