"""Tests for operator compatibility.
Authored by Shritej Gunda.
"""

from datetime import UTC, datetime, timedelta

import pytest

from agent.authority import TrialInstructionEvidence
from agent.commands import AssemblyScope, TaskRequest
from agent.task_port import ControllerSource, OperatorTaskPortError, T09TaskCoordinator
from assembly.state_machine import PackMLState, StateRecord
from robot.cell.t09_mock_runtime import MockPerception, _build_controller


def _current_trial() -> TrialInstructionEvidence:
    captured = datetime.now(UTC)
    return TrialInstructionEvidence(
        "run-actual",
        1,
        "obs-actual",
        "assemble products 1, 3, 4",
        captured,
        captured + timedelta(minutes=1),
        "trial-port",
    )


def test_actual_t09_controller_satisfies_task_port_read_boundary() -> None:
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _build_controller(MockPerception()),
        trial_instruction=_current_trial,
        clock=lambda: datetime.now(UTC),
        task_id_factory=lambda: "actual-t09",
    )

    receipt = coordinator.submit(TaskRequest(AssemblyScope.ALL_VISIBLE))
    report = coordinator.report()

    assert receipt.selected_products == (1, 3, 4)
    assert report.packml["state"] == "Idle"
    assert report.selected_products == (1, 3, 4)
    assert report.execution_authorized is True


class _StateMachine:
    def __init__(self, record: StateRecord) -> None:
        self.current_record = record


class _UnboundController:
    requested_products = ()
    events = ()
    plan_records = ()

    def __init__(self, state: PackMLState) -> None:
        self.completed_products = (1, 3, 4) if state is PackMLState.COMPLETE else ()
        self.finished = state is PackMLState.COMPLETE
        product = None if state is PackMLState.IDLE else 1
        self.state_machine = _StateMachine(StateRecord(state, None, product, datetime.now(UTC).isoformat()))


@pytest.mark.parametrize("state", [PackMLState.EXECUTE, PackMLState.COMPLETE])
def test_unbound_products_are_rejected_after_controller_leaves_pristine_idle(state: PackMLState) -> None:
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _UnboundController(state),
        trial_instruction=_current_trial,
        clock=lambda: datetime.now(UTC),
        task_id_factory=lambda: "unbound-t09",
    )

    with pytest.raises(OperatorTaskPortError, match="requested products|trial instruction"):
        coordinator.submit(TaskRequest(AssemblyScope.ALL_VISIBLE))


def test_empty_requested_products_are_allowed_only_for_pristine_idle() -> None:
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _UnboundController(PackMLState.IDLE),
        trial_instruction=_current_trial,
        clock=lambda: datetime.now(UTC),
        task_id_factory=lambda: "pristine-t09",
    )

    receipt = coordinator.submit(TaskRequest(AssemblyScope.ALL_VISIBLE))

    assert receipt.selected_products == (1, 3, 4)
