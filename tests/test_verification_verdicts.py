"""Explicit verdict and identity-agreement regressions for T09."""

from assembly.verification import VerificationOutcome, VerificationReason
from tests.test_verification import _detection, _failed_inspection, _inspection, _verify_post, _verify_pre


def test_pre_pick_trusts_agreed_defect_verdict_instead_of_recomputing_raw_scores() -> None:
    inspection = _failed_inspection("broken", "reject")
    inspection["evidence"] = {
        "classifier_score": 0.99,
        "patchcore_score": 0.99,
        "depth_residual_mm": 0.2,
    }

    decision = _verify_pre("M4_03", _detection(), inspection)

    assert decision.outcome is VerificationOutcome.REJECT_TO_SPARE
    assert decision.reason is VerificationReason.INSPECTION_REJECTED


def test_pre_pick_records_foundationpose_disagreement_without_blocking_acceptance() -> None:
    inspection = _inspection()
    inspection["checks_agree"] = False

    decision = _verify_pre("M4_03", _detection(), inspection)

    assert decision.outcome is VerificationOutcome.ACCEPT
    assert decision.reason is VerificationReason.VERIFIED
    assert decision.advisories == ("foundationpose_disagreement",)


def test_post_place_holds_when_validated_action_asks_operator() -> None:
    inspection = _failed_inspection("uncertain", "ask_operator")
    inspection["evidence"] = {
        "classifier_score": 0.98,
        "patchcore_score": 0.03,
        "depth_residual_mm": 0.4,
    }

    decision = _verify_post("M4_03", _detection(), inspection, seated=True, pose_agrees=True)

    assert decision.outcome is VerificationOutcome.HOLD_ANDON
    assert decision.reason is VerificationReason.INSPECTION_REJECTED
