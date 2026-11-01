"""PatchCore and FoundationPose advisory-authority regressions.
Authored by Shritej Gunda.
"""

import pytest

from assembly.verification import VerificationOutcome, VerificationReason
from tests.test_verification import (
    _detection,
    _failed_inspection,
    _inspection,
    _verify_post,
    _verify_pre,
)


def test_low_detector_and_foundationpose_scores_are_advisory() -> None:
    detection = _detection()
    inspection = _inspection()
    detection.update(score=0.0, pose_fit_score=0.0)
    inspection["evidence"] = {
        "classifier_score": 0.98,
        "patchcore_score": 0.03,
        "depth_residual_mm": 0.4,
    }

    pre = _verify_pre("M4_03", detection, inspection)
    post = _verify_post("M4_03", detection, inspection, seated=True, pose_agrees=True)

    assert pre.outcome is VerificationOutcome.ACCEPT
    assert post.outcome is VerificationOutcome.ACCEPT
    assert pre.advisories == ("detection_low_confidence", "foundationpose_low_fit")
    assert post.advisories == pre.advisories


def test_patchcore_threshold_warning_requires_operator_without_reject_motion() -> None:
    inspection = _inspection()
    inspection["evidence"] = {
        "classifier_score": 0.98,
        "patchcore_score": 0.9,
        "depth_residual_mm": 0.4,
    }

    pre = _verify_pre("M4_03", _detection(), inspection)
    post = _verify_post("M4_03", _detection(), inspection, seated=True, pose_agrees=True)

    assert pre.outcome is VerificationOutcome.HOLD_ANDON
    assert post.outcome is VerificationOutcome.HOLD_ANDON
    assert pre.reason is VerificationReason.PATCHCORE_WARNING
    assert post.reason is VerificationReason.PATCHCORE_WARNING


def test_patchcore_generated_operator_request_keeps_patchcore_warning_reason() -> None:
    inspection = _failed_inspection("uncertain", "ask_operator")
    inspection["evidence"] = {
        "classifier_score": 0.98,
        "patchcore_score": 0.9,
        "depth_residual_mm": 0.4,
    }

    pre = _verify_pre("M4_03", _detection(), inspection)
    post = _verify_post("M4_03", _detection(), inspection, seated=True, pose_agrees=True)

    assert pre.outcome is VerificationOutcome.HOLD_ANDON
    assert post.outcome is VerificationOutcome.HOLD_ANDON
    assert pre.reason is VerificationReason.PATCHCORE_WARNING
    assert post.reason is VerificationReason.PATCHCORE_WARNING


@pytest.mark.parametrize("orientation", ["flipped", "side", "unknown"])
def test_foundationpose_orientation_is_advisory_at_pre_pick(orientation: str) -> None:
    detection = _detection()
    detection["resting_side"] = orientation

    decision = _verify_pre("M4_03", detection, _inspection())

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert "foundationpose_orientation" in decision.advisories
