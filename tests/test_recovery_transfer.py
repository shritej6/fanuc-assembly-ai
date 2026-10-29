"""Stage-specific evidence gates for reject and staging transfers."""

from copy import deepcopy

from assembly.verification import (
    VerificationOutcome,
    VerificationThresholds,
    verify_recovery_transfer,
)

THRESHOLDS = VerificationThresholds(0.9, 0.8, 0.9, 0.1, 1.0)


def _detection() -> dict[str, object]:
    return {
        "instance_id": "p1-1",
        "camera": "wrist",
        "label": "M1_01",
        "object_id": "M1_01",
        "score": 0.97,
        "bbox_px": [10, 20, 60, 80],
        "pose": {"x": 500.0, "y": -265.0, "z": -300.0, "yaw": 0.0},
        "pose_source": "foundationpose",
        "pose_fit_score": 0.91,
        "resting_side": "upright",
    }


def _inspection() -> dict[str, object]:
    return {
        "instance_id": "p1-1",
        "object_id": "M1_01",
        "verdict": "ok",
        "action": "use",
        "checks_agree": True,
        "evidence": {
            "classifier_score": 0.98,
            "patchcore_score": 0.03,
            "depth_residual_mm": 0.4,
            "image_ref": "mock://p1-1",
        },
    }


def _verify(detection, inspection, *, pose_agrees: bool = True, allow_rejected: bool = False):
    return verify_recovery_transfer(
        "M1_01",
        detection,
        inspection,
        expected_instance_id="p1-1",
        thresholds=THRESHOLDS,
        pose_agrees=pose_agrees,
        allow_rejected=allow_rejected,
    )


def test_staging_transfer_requires_good_identity_and_installed_pose() -> None:
    assert _verify(_detection(), _inspection()).outcome is VerificationOutcome.ACCEPT


def test_reject_transfer_accepts_confirmed_defect_with_classifier_identity() -> None:
    inspection = _inspection()
    inspection.update(verdict="print_defect", action="reject")
    inspection["evidence"] = {
        "classifier_score": 0.98,
        "patchcore_score": 0.95,
        "depth_residual_mm": 3.0,
        "image_ref": "mock://p1-1",
    }

    decision = _verify(_detection(), inspection, allow_rejected=True)

    assert decision.outcome is VerificationOutcome.ACCEPT


def test_transfer_rejects_wrong_physical_instance_from_either_check() -> None:
    for field in ("detection", "inspection"):
        detection, inspection = deepcopy(_detection()), deepcopy(_inspection())
        payload = detection if field == "detection" else inspection
        payload["instance_id"] = "p1-other"

        assert _verify(detection, inspection).outcome is VerificationOutcome.HOLD_ANDON


def test_transfer_rejects_unconfirmed_target_pose() -> None:
    assert _verify(_detection(), _inspection(), pose_agrees=False).outcome is VerificationOutcome.HOLD_ANDON


def test_rejected_transfer_still_requires_reject_action() -> None:
    inspection = _inspection()
    inspection.update(verdict="flipped", action="rework")

    decision = _verify(_detection(), inspection, allow_rejected=True)

    assert decision.outcome is VerificationOutcome.HOLD_ANDON


def test_reject_transfer_does_not_trust_false_good_verdict() -> None:
    inspection = _inspection()
    inspection["evidence"] = {
        "classifier_score": 0.98,
        "patchcore_score": 0.95,
        "depth_residual_mm": 3.0,
        "image_ref": "mock://p1-1",
    }

    decision = _verify(_detection(), inspection, allow_rejected=True)

    assert decision.outcome is VerificationOutcome.HOLD_ANDON
