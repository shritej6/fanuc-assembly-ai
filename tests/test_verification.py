"""Behavioral tests for fail-closed perception and recovery decisions.
Authored by Shritej Gunda.
"""

from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from assembly.verification import (
    VerificationDecision,
    VerificationInputError,
    VerificationOutcome,
    VerificationReason,
    VerificationThresholds,
    verify_post_place,
    verify_pre_pick,
)

_THRESHOLDS = VerificationThresholds(
    min_detection_score=0.9,
    min_pose_fit_score=0.8,
    min_classifier_score=0.9,
    max_patchcore_score=0.1,
    max_depth_residual_mm=1.0,
)


def _verify_pre(
    expected_object_id: str,
    detection: dict[str, object],
    inspection: dict[str, object],
    *,
    expected_instance_id: str = "obj-07",
) -> VerificationDecision:
    return verify_pre_pick(
        expected_object_id,
        detection,
        inspection,
        expected_instance_id=expected_instance_id,
        thresholds=_THRESHOLDS,
    )


def _verify_post(
    expected_object_id: str,
    detection: dict[str, object],
    inspection: dict[str, object],
    *,
    expected_instance_id: str = "obj-07",
    seated: bool,
    pose_agrees: bool,
) -> VerificationDecision:
    return verify_post_place(
        expected_object_id,
        detection,
        inspection,
        expected_instance_id=expected_instance_id,
        thresholds=_THRESHOLDS,
        seated=seated,
        pose_agrees=pose_agrees,
    )


def _detection() -> dict[str, object]:
    return {
        "instance_id": "obj-07",
        "camera": "wrist",
        "label": "guide",
        "object_id": "M4_03",
        "score": 0.97,
        "bbox_px": [10, 20, 60, 80],
        "pose": {"x": 250.0, "y": -10.0, "z": -320.0, "yaw": 90.0},
        "pose_source": "foundationpose",
        "pose_fit_score": 0.91,
        "resting_side": "upright",
    }


def _inspection() -> dict[str, object]:
    return {
        "instance_id": "obj-07",
        "object_id": "M4_03",
        "verdict": "ok",
        "action": "use",
        "checks_agree": True,
        "evidence": {
            "classifier_score": 0.98,
            "patchcore_score": 0.03,
            "depth_residual_mm": 0.4,
            "image_ref": "wrist://obj-07",
        },
    }


def _failed_inspection(verdict: str, action: str) -> dict[str, object]:
    inspection = _inspection()
    inspection.update(verdict=verdict, action=action)
    inspection["evidence"] = {
        "classifier_score": 0.1,
        "patchcore_score": 0.9,
        "depth_residual_mm": 4.0,
        "image_ref": "wrist://obj-07",
    }
    return inspection


def test_pre_pick_accepts_exact_verified_upright_part() -> None:
    decision = _verify_pre("M4_03", _detection(), _inspection())

    assert decision == VerificationDecision(
        outcome=VerificationOutcome.ACCEPT,
        reason=VerificationReason.VERIFIED,
        instance_id="obj-07",
        object_id="M4_03",
    )


def test_pre_pick_holds_same_class_wrong_physical_instance() -> None:
    detection = _detection()
    inspection = _inspection()
    detection["instance_id"] = "obj-08"
    inspection["instance_id"] = "obj-08"

    decision = _verify_pre("M4_03", detection, inspection)

    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert decision.reason is VerificationReason.IDENTITY_MISMATCH


@pytest.mark.parametrize("orientation", ["installed", "fallen"])
def test_pre_pick_rejects_non_schema_orientation_aliases(orientation: str) -> None:
    detection = _detection()
    detection["resting_side"] = orientation

    with pytest.raises(VerificationInputError, match="detection.resting_side"):
        _verify_pre("M4_03", detection, _inspection())


@pytest.mark.parametrize(
    ("verdict", "action"),
    [
        ("flipped", "rework"),
        ("flipped", "reject"),
        ("broken", "reject"),
        ("missing_feature", "reject"),
        ("print_defect", "reject"),
        ("missing_layer", "reject"),
        ("wrong_part", "reject"),
        ("foreign", "reject"),
        ("uncertain", "reject"),
    ],
)
def test_pre_pick_rejects_unusable_inspection_to_spare(verdict: str, action: str) -> None:
    inspection = _failed_inspection(verdict, action)

    decision = _verify_pre("M4_03", _detection(), inspection)

    assert decision.outcome is VerificationOutcome.REJECT_TO_SPARE
    assert decision.reason is VerificationReason.INSPECTION_REJECTED


def test_pre_pick_holds_when_validated_action_asks_operator() -> None:
    inspection = _failed_inspection("uncertain", "ask_operator")
    inspection["evidence"] = {
        "classifier_score": 0.98,
        "patchcore_score": 0.03,
        "depth_residual_mm": 0.4,
    }

    decision = _verify_pre("M4_03", _detection(), inspection)

    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert decision.reason is VerificationReason.INSPECTION_REJECTED


def test_pre_pick_rejects_agreed_wrong_identity_to_spare() -> None:
    detection = _detection()
    inspection = _inspection()
    detection["object_id"] = "M5_03"
    inspection["object_id"] = "M5_03"

    decision = _verify_pre("M4_03", detection, inspection)

    assert decision.outcome is VerificationOutcome.REJECT_TO_SPARE
    assert decision.reason is VerificationReason.IDENTITY_MISMATCH


def test_pre_pick_foundationpose_disagreement_is_advisory() -> None:
    inspection = _inspection()
    inspection["checks_agree"] = False

    decision = _verify_pre("M4_03", _detection(), inspection)

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert decision.advisories == ("foundationpose_disagreement",)


@pytest.mark.parametrize("disagreement", ["object_id", "instance_id", "action"])
def test_pre_pick_holds_when_independent_checks_disagree(disagreement: str) -> None:
    detection = deepcopy(_detection())
    inspection = deepcopy(_inspection())
    if disagreement == "object_id":
        inspection["object_id"] = "M5_03"
    elif disagreement == "instance_id":
        inspection["instance_id"] = "obj-08"
    else:
        inspection["action"] = "reject"

    decision = _verify_pre("M4_03", detection, inspection)

    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert decision.reason is VerificationReason.CHECK_DISAGREEMENT


@pytest.mark.parametrize(
    ("field", "value"),
    [("pose_source", "detector"), ("pose", None), ("pose_fit_score", None)],
)
def test_pre_pick_accepts_with_missing_foundationpose_advisory(field: str, value: object) -> None:
    detection = _detection()
    detection[field] = value

    decision = _verify_pre("M4_03", detection, _inspection())

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert "foundationpose_unavailable" in decision.advisories


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("camera", "overhead"),
        ("score", True),
        ("score", -0.01),
        ("score", 1.01),
        ("score", float("nan")),
        ("bbox_px", [-1, 20, 60, 80]),
        ("bbox_px", [60, 20, 10, 80]),
        ("bbox_px", [10, 20, float("inf"), 80]),
        ("object_id", "M9_99"),
        ("object_id", []),
        ("pose", {"x": 1, "y": 2, "z": 3}),
        ("pose_source", "icp_guess"),
        ("pose_fit_score", -0.01),
        ("pose_fit_score", 1.01),
        ("resting_side", "diagonal"),
    ],
)
def test_pre_pick_validates_detection_types_and_ranges(field: str, value: object) -> None:
    detection = _detection()
    detection[field] = value

    with pytest.raises(VerificationInputError, match=f"detection.{field}"):
        _verify_pre("M4_03", detection, _inspection())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_id", ""),
        ("object_id", "M9_99"),
        ("object_id", []),
        ("verdict", "maybe"),
        ("action", "use_anyway"),
        ("checks_agree", 1),
        ("evidence", []),
        ("evidence", {"classifier_score": 1.01}),
        ("evidence", {"patchcore_score": -0.01}),
        ("evidence", {"depth_residual_mm": -0.01}),
        ("evidence", {"image_ref": 42}),
    ],
)
def test_pre_pick_validates_inspection_types_and_ranges(field: str, value: object) -> None:
    inspection = _inspection()
    inspection[field] = value

    with pytest.raises(VerificationInputError, match=f"inspection.{field}"):
        _verify_pre("M4_03", _detection(), inspection)


@pytest.mark.parametrize(
    ("source", "field"),
    [("detection", "camera"), ("inspection", "verdict")],
)
def test_pre_pick_requires_contract_fields(source: str, field: str) -> None:
    detection = _detection()
    inspection = _inspection()
    target = detection if source == "detection" else inspection
    del target[field]

    with pytest.raises(VerificationInputError, match=f"{source}.{field}"):
        _verify_pre("M4_03", detection, inspection)


@pytest.mark.parametrize("expected_object_id", ["", "F01", "M7_01", 403, []])
def test_pre_pick_requires_known_part_as_expected_identity(expected_object_id: object) -> None:
    with pytest.raises(VerificationInputError, match="expected_object_id"):
        _verify_pre(expected_object_id, _detection(), _inspection())  # type: ignore[arg-type]


@pytest.mark.parametrize("expected_instance_id", ["", 7, [], None])
def test_pre_pick_requires_expected_physical_instance(expected_instance_id: object) -> None:
    with pytest.raises(VerificationInputError, match="expected_instance_id"):
        _verify_pre(
            "M4_03",
            _detection(),
            _inspection(),
            expected_instance_id=expected_instance_id,  # type: ignore[arg-type]
        )


def test_pre_pick_does_not_mutate_perception_payloads() -> None:
    detection = _detection()
    inspection = _inspection()
    original_detection = deepcopy(detection)
    original_inspection = deepcopy(inspection)

    _verify_pre("M4_03", detection, inspection)

    assert detection == original_detection
    assert inspection == original_inspection


def test_post_place_accepts_independently_verified_seated_part() -> None:
    decision = _verify_post(
        "M4_03",
        _detection(),
        _inspection(),
        seated=True,
        pose_agrees=True,
    )

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert decision.reason is VerificationReason.VERIFIED


def test_post_place_requests_recovery_when_both_placement_checks_fail() -> None:
    decision = _verify_post(
        "M4_03",
        _detection(),
        _inspection(),
        seated=False,
        pose_agrees=False,
    )

    assert decision.outcome is VerificationOutcome.RECOVER
    assert decision.reason is VerificationReason.NOT_SEATED


@pytest.mark.parametrize("orientation", ["flipped", "side", "unknown"])
def test_post_place_foundationpose_orientation_is_advisory_when_seated(orientation: str) -> None:
    detection = _detection()
    detection["resting_side"] = orientation

    decision = _verify_post("M4_03", detection, _inspection(), seated=True, pose_agrees=True)

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert "foundationpose_orientation" in decision.advisories


def test_post_place_accepts_seated_part_with_foundationpose_disagreement_warning() -> None:
    decision = _verify_post(
        "M4_03",
        _detection(),
        _inspection(),
        seated=True,
        pose_agrees=False,
    )

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert "foundationpose_disagreement" in decision.advisories


def test_post_place_recovers_when_independent_seat_check_fails() -> None:
    decision = _verify_post(
        "M4_03",
        _detection(),
        _inspection(),
        seated=False,
        pose_agrees=True,
    )

    assert decision.outcome is VerificationOutcome.RECOVER
    assert decision.reason is VerificationReason.NOT_SEATED


def test_post_place_recovers_an_agreed_wrong_identity() -> None:
    detection = _detection()
    inspection = _inspection()
    detection["object_id"] = "M5_03"
    inspection["object_id"] = "M5_03"

    decision = _verify_post(
        "M4_03",
        detection,
        inspection,
        seated=True,
        pose_agrees=True,
    )

    assert decision.outcome is VerificationOutcome.RECOVER
    assert decision.reason is VerificationReason.IDENTITY_MISMATCH


def test_post_place_holds_same_class_wrong_physical_instance() -> None:
    detection = _detection()
    inspection = _inspection()
    detection["instance_id"] = "obj-08"
    inspection["instance_id"] = "obj-08"

    decision = _verify_post("M4_03", detection, inspection, seated=True, pose_agrees=True)

    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert decision.reason is VerificationReason.IDENTITY_MISMATCH


def test_post_place_foundationpose_check_disagreement_is_advisory() -> None:
    inspection = _inspection()
    inspection["checks_agree"] = False

    decision = _verify_post(
        "M4_03",
        _detection(),
        inspection,
        seated=True,
        pose_agrees=True,
    )

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert decision.advisories == ("foundationpose_disagreement",)


def test_post_place_holds_on_independent_identity_disagreement() -> None:
    detection = _detection()
    inspection = _inspection()
    inspection["object_id"] = "M5_03"

    decision = _verify_post(
        "M4_03",
        detection,
        inspection,
        seated=True,
        pose_agrees=True,
    )

    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert decision.reason is VerificationReason.CHECK_DISAGREEMENT


@pytest.mark.parametrize(("seated", "pose_agrees"), [(1, True), (True, "yes")])
def test_post_place_requires_boolean_placement_checks(seated: object, pose_agrees: object) -> None:
    with pytest.raises(VerificationInputError, match="post_place"):
        _verify_post(
            "M4_03",
            _detection(),
            _inspection(),
            seated=seated,  # type: ignore[arg-type]
            pose_agrees=pose_agrees,  # type: ignore[arg-type]
        )


def test_post_place_accepts_without_foundationpose_evidence() -> None:
    detection = _detection()
    detection["pose_fit_score"] = None

    decision = _verify_post(
        "M4_03",
        detection,
        _inspection(),
        seated=True,
        pose_agrees=True,
    )

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert decision.advisories == ("foundationpose_unavailable",)


def test_post_place_recovers_a_failed_part_inspection() -> None:
    inspection = _failed_inspection("broken", "reject")

    decision = _verify_post(
        "M4_03",
        _detection(),
        inspection,
        seated=True,
        pose_agrees=True,
    )

    assert decision.outcome is VerificationOutcome.RECOVER
    assert decision.reason is VerificationReason.INSPECTION_REJECTED


def test_verification_decision_is_immutable() -> None:
    verification = _verify_pre("M4_03", _detection(), _inspection())

    with pytest.raises(FrozenInstanceError):
        verification.outcome = VerificationOutcome.RECOVER  # type: ignore[misc]
