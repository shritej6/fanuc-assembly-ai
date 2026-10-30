"""Behavioral tests for the fail-closed PackML state machine."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[1]


def test_initial_record_uses_exact_contract_state_and_shape() -> None:
    from assembly.state_machine import PackMLState, PackMLStateMachine

    now = datetime(2026, 10, 29, 12, 30, tzinfo=UTC)
    machine = PackMLStateMachine(clock=lambda: now)
    schema = json.loads((ROOT / "contracts" / "packml_state.schema.json").read_text())
    record = machine.current_record.as_dict()

    assert {state.value for state in PackMLState} == {
        "Idle",
        "Starting",
        "Execute",
        "Held",
        "Suspended",
        "Completing",
        "Complete",
        "Stopping",
        "Stopped",
        "Aborting",
        "Aborted",
        "Resetting",
    }
    assert record == {
        "state": "Idle",
        "reason": None,
        "product": None,
        "timestamp": "2026-10-29T12:30:00Z",
    }
    assert list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(record)) == []


def test_happy_path_records_one_product_through_hold_suspend_and_completion() -> None:
    from assembly.state_machine import PackMLState, PackMLStateMachine

    machine = PackMLStateMachine()
    machine.start(2)
    machine.transition(PackMLState.EXECUTE)
    machine.hold("operator inspection")
    machine.resume()
    machine.suspend("waiting for M2_03 spare")
    machine.resume()
    machine.complete()
    machine.transition(PackMLState.COMPLETE)

    assert machine.state is PackMLState.COMPLETE
    assert machine.current_product == 2
    assert [record.state for record in machine.history] == [
        PackMLState.IDLE,
        PackMLState.STARTING,
        PackMLState.EXECUTE,
        PackMLState.HELD,
        PackMLState.EXECUTE,
        PackMLState.SUSPENDED,
        PackMLState.EXECUTE,
        PackMLState.COMPLETING,
        PackMLState.COMPLETE,
    ]
    assert machine.history[3].reason == "operator inspection"
    assert machine.history[5].reason == "waiting for M2_03 spare"


def test_invalid_transition_is_typed_and_does_not_mutate_history() -> None:
    from assembly.state_machine import InvalidTransitionError, PackMLState, PackMLStateMachine

    machine = PackMLStateMachine()
    with pytest.raises(InvalidTransitionError) as caught:
        machine.transition(PackMLState.COMPLETE)

    assert caught.value.current is PackMLState.IDLE
    assert caught.value.target is PackMLState.COMPLETE
    assert machine.state is PackMLState.IDLE
    assert len(machine.history) == 1


def test_unknown_state_fails_closed_with_domain_error() -> None:
    from assembly.state_machine import PackMLState, PackMLStateMachine, UnknownStateError

    machine = PackMLStateMachine()
    with pytest.raises(UnknownStateError, match="NotAState"):
        machine.transition("NotAState")

    assert machine.state is PackMLState.IDLE
    assert len(machine.history) == 1


def test_non_string_reason_cannot_create_schema_invalid_record() -> None:
    from assembly.state_machine import InvalidReasonError, PackMLState, PackMLStateMachine

    machine = PackMLStateMachine()

    with pytest.raises(InvalidReasonError):
        machine.transition(PackMLState.STARTING, product=1, reason=42)  # type: ignore[arg-type]

    assert machine.state is PackMLState.IDLE
    assert machine.current_product is None
    assert len(machine.history) == 1


def test_invalid_transition_timestamp_cannot_claim_product() -> None:
    from assembly.state_machine import InvalidTimestampError, PackMLState, PackMLStateMachine

    times = iter(
        [
            datetime(2026, 10, 29, 12, 30, tzinfo=UTC),
            datetime(2026, 10, 29, 12, 31),
        ]
    )
    machine = PackMLStateMachine(clock=lambda: next(times))

    with pytest.raises(InvalidTimestampError):
        machine.start(1)

    assert machine.state is PackMLState.IDLE
    assert machine.current_product is None
    assert len(machine.history) == 1


def test_invalid_fault_timestamp_cannot_release_reservations() -> None:
    from assembly.state_machine import InvalidTimestampError, PackMLState, PackMLStateMachine

    times = iter(
        [
            datetime(2026, 10, 29, 12, 30, tzinfo=UTC),
            datetime(2026, 10, 29, 12, 31, tzinfo=UTC),
            datetime(2026, 10, 29, 12, 32, tzinfo=UTC),
            datetime(2026, 10, 29, 12, 33),
        ]
    )
    releaser = _RecordingReleaser()
    machine = PackMLStateMachine(clock=lambda: next(times), reservation_releaser=releaser)
    machine.start(1)
    machine.transition(PackMLState.EXECUTE)

    with pytest.raises(InvalidTimestampError):
        machine.abort("fault")

    assert machine.state is PackMLState.EXECUTE
    assert machine.current_product == 1
    assert releaser.products == []


@pytest.mark.parametrize("product", [True, 0, 7, "1"])
def test_start_rejects_invalid_product_without_claiming_ownership(product: object) -> None:
    from assembly.state_machine import InvalidProductError, PackMLState, PackMLStateMachine

    machine = PackMLStateMachine()
    with pytest.raises(InvalidProductError):
        machine.start(product)  # type: ignore[arg-type]

    assert machine.state is PackMLState.IDLE
    assert machine.current_product is None


def test_active_product_cannot_be_replaced_mid_cycle() -> None:
    from assembly.state_machine import PackMLState, PackMLStateMachine, ProductOwnershipError

    machine = PackMLStateMachine()
    machine.start(1)
    with pytest.raises(ProductOwnershipError) as caught:
        machine.start(2)

    assert caught.value.owner == 1
    assert caught.value.requested == 2
    assert machine.state is PackMLState.STARTING
    assert machine.current_product == 1


class _RecordingReleaser:
    def __init__(self, failure: Exception | None = None) -> None:
        self.products: list[int] = []
        self.failure = failure

    def release_reservations(self, product: int | None = None) -> tuple[str, ...]:
        assert product is not None
        if self.failure is not None:
            raise self.failure
        self.products.append(product)
        return ("released",)


def _execute_machine(releaser: _RecordingReleaser | None = None):
    from assembly.state_machine import PackMLState, PackMLStateMachine

    machine = PackMLStateMachine(reservation_releaser=releaser)
    machine.start(4)
    machine.transition(PackMLState.EXECUTE)
    return machine


def test_abort_releases_reservations_before_entering_fault_path() -> None:
    from assembly.state_machine import PackMLState

    releaser = _RecordingReleaser()
    machine = _execute_machine(releaser)
    record = machine.abort("camera verification failed")

    assert record.state is PackMLState.ABORTING
    assert releaser.products == [4]
    machine.transition(PackMLState.ABORTED)


def test_reset_releases_reservations_and_idle_clears_product() -> None:
    from assembly.state_machine import PackMLState

    releaser = _RecordingReleaser()
    machine = _execute_machine(releaser)
    machine.complete()
    machine.transition(PackMLState.COMPLETE)
    machine.reset("next product")
    machine.transition(PackMLState.IDLE)

    assert releaser.products == [4]
    assert machine.state is PackMLState.IDLE
    assert machine.current_product is None
    assert machine.current_record.product is None


def test_release_failure_is_typed_and_keeps_previous_state() -> None:
    from assembly.state_machine import PackMLState, ReservationReleaseError

    machine = _execute_machine(_RecordingReleaser(RuntimeError("store offline")))
    with pytest.raises(ReservationReleaseError, match="store offline"):
        machine.abort("fault")

    assert machine.state is PackMLState.EXECUTE
    assert len(machine.history) == 3


def test_stop_path_is_explicit_and_releases_active_reservations() -> None:
    from assembly.state_machine import PackMLState

    releaser = _RecordingReleaser()
    machine = _execute_machine(releaser)
    machine.stop("controlled stop")
    machine.transition(PackMLState.STOPPED)

    assert machine.state is PackMLState.STOPPED
    assert releaser.products == [4]
