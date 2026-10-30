"""Behavioral tests for bounded insertion recovery decisions."""

from dataclasses import FrozenInstanceError

import pytest

from assembly.verification import (
    RecoveryAction,
    RecoveryDecision,
    RecoveryOperation,
    VerificationInputError,
    decide_recovery,
)


@pytest.mark.parametrize(("failed_attempts", "retry_number"), [(0, 1), (1, 2)])
def test_recovery_retries_lift_re_yaw_reinsert_twice_at_most(
    failed_attempts: int,
    retry_number: int,
) -> None:
    decision = decide_recovery("M4_03", failed_attempts, verified_spare_available=False, installed_object_ids=())

    assert decision == RecoveryDecision(
        action=RecoveryAction.RETRY_INSERTION,
        retry_number=retry_number,
        operations=(RecoveryOperation.LIFT, RecoveryOperation.RE_YAW, RecoveryOperation.REINSERT),
        dismantle_order=(),
        rebuild_order=(),
        replacement_object_id=None,
    )


@pytest.mark.parametrize("failed_attempts", [2, 9])
def test_exhausted_recovery_dismantles_reverse_then_rebuilds_with_spare(failed_attempts: int) -> None:
    installed = ("M4_01", "M4_02", "M4_03", "M4_04")
    decision = decide_recovery("M4_03", failed_attempts, verified_spare_available=True, installed_object_ids=installed)

    assert decision.action is RecoveryAction.DISMANTLE_REBUILD
    assert decision.retry_number is None
    assert decision.operations == ()
    assert decision.dismantle_order == ("M4_04", "M4_03", "M4_02", "M4_01")
    assert decision.rebuild_order == ("M4_01", "M4_02", "M4_03", "M4_04")
    assert decision.replacement_object_id == "M4_03"


def test_exhausted_recovery_only_orders_inventory_installed_prefix() -> None:
    installed = ("M4_01", "M4_02")

    decision = decide_recovery("M4_03", 2, verified_spare_available=True, installed_object_ids=installed)

    assert decision.dismantle_order == ("M4_02", "M4_01")
    assert decision.rebuild_order == installed


def test_exhausted_recovery_holds_when_verified_spare_is_missing() -> None:
    decision = decide_recovery("M4_03", 2, verified_spare_available=False, installed_object_ids=("M4_01", "M4_02"))

    assert decision == RecoveryDecision(
        action=RecoveryAction.HOLD_ANDON,
        retry_number=None,
        operations=(),
        dismantle_order=(),
        rebuild_order=(),
        replacement_object_id=None,
    )


@pytest.mark.parametrize(
    ("object_id", "failed_attempts", "spare", "installed"),
    [
        ("F01", 0, True, ()),
        ("M4_03", -1, True, ()),
        ("M4_03", True, True, ()),
        ("M4_03", 1.5, True, ()),
        ("M4_03", 2, 1, ()),
        ("M4_03", 2, True, "M4_01"),
        ("M4_03", 2, True, ("M4_02",)),
        ("M4_03", 2, True, ("M5_01",)),
    ],
)
def test_recovery_validates_policy_inputs(
    object_id: object, failed_attempts: object, spare: object, installed: object
) -> None:
    with pytest.raises(VerificationInputError):
        decide_recovery(
            object_id,  # type: ignore[arg-type]
            failed_attempts,  # type: ignore[arg-type]
            spare,  # type: ignore[arg-type]
            installed_object_ids=installed,  # type: ignore[arg-type]
        )


def test_recovery_decision_is_immutable() -> None:
    recovery = decide_recovery("M4_03", 0, verified_spare_available=True, installed_object_ids=())

    with pytest.raises(FrozenInstanceError):
        recovery.action = RecoveryAction.HOLD_ANDON  # type: ignore[misc]
