from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agent.cosmos_reason import (
    BuildCheckOutcome,
    CosmosEvidenceSource,
    CosmosReasonResult,
    CosmosReasonUnavailable,
    FinalBuildGate,
    FinalBuildRequest,
)
from agent.operator import OperatorAgent, OperatorResponseKind
from agent.task_port import (
    ControllerSource,
    OperatorTaskPortError,
    QualityRelease,
    T09TaskCoordinator,
)
from assembly.controller_ports import ControllerEvent
from assembly.planner import ContractPoseUF0, PlanStepRecord
from assembly.state_machine import PackMLState, StateRecord
from robot.cell.t09_mock_runtime import MockPerception, _build_controller

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)
TIMESTAMP = "2026-10-29T15:00:00Z"


class FakeStateMachine:
    def __init__(self, state: StateRecord | None = None) -> None:
        self.current_record = state or StateRecord(PackMLState.IDLE, None, None, TIMESTAMP)


class FakeController:
    def __init__(self, products: tuple[int, ...]) -> None:
        self.products = products
        self.completed_products: tuple[int, ...] = ()
        self.events: tuple[ControllerEvent, ...] = ()
        self.plan_records: tuple[PlanStepRecord, ...] = ()
        self.finished = False
        self.state_machine = FakeStateMachine()


class PassingCosmos:
    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        return CosmosReasonResult(
            request.task_id,
            request.request_id,
            request.product,
            True,
            "visual build matches",
            "Cosmos-Reason2-2B",
            CosmosEvidenceSource.COSMOS_REASON_2,
            request.observed_at,
        )


class UnavailableCosmos:
    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        raise CosmosReasonUnavailable(f"no result for {request.request_id}")


def _coordinator(
    *,
    visible: tuple[int, ...] = (3, 1, 2),
    factory: object | None = None,
) -> tuple[T09TaskCoordinator, list[FakeController]]:
    created: list[FakeController] = []

    def default_factory(products: tuple[int, ...]) -> FakeController:
        controller = FakeController(products)
        created.append(controller)
        return controller

    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=factory or default_factory,  # type: ignore[arg-type]
        visible_products=lambda: visible,
        clock=lambda: NOW,
        task_id_factory=lambda: "task-1",
    )
    return coordinator, created


def _agent(coordinator: T09TaskCoordinator) -> OperatorAgent:
    return OperatorAgent(coordinator, FinalBuildGate(PassingCosmos()))


def _complete(controller: FakeController, products: tuple[int, ...]) -> None:
    controller.completed_products = products
    controller.finished = True
    controller.state_machine.current_record = StateRecord(PackMLState.COMPLETE, None, products[-1], TIMESTAMP)


def _request(product: int = 1) -> FinalBuildRequest:
    return FinalBuildRequest(
        "task-1",
        f"run-1-product-{product}",
        product,
        tuple(f"M{product}_{step:02d}" for step in range(1, 5)),
        (f"eval/run-1/product-{product}-side.png",),
        NOW,
    )


def _record(product: int = 1) -> PlanStepRecord:
    pose = ContractPoseUF0(400.0, 0.0, -300.0, 0.0)
    return PlanStepRecord(
        f"product-{product}-step-01-attempt-1", product, "part-1", f"M{product}_01", pose, pose, 0.0, 1
    )


def test_status_before_task_is_read_only_idle_report() -> None:
    coordinator, created = _coordinator()

    response = _agent(coordinator).handle("status")

    assert response.kind is OperatorResponseKind.REPORT
    assert response.report is not None
    assert response.report.packml["state"] == "Idle"
    assert response.report.source is ControllerSource.MOCK
    assert created == []


def test_assemble_everything_resolves_visible_products_deterministically() -> None:
    coordinator, created = _coordinator()

    response = _agent(coordinator).handle("assemble everything on the table")

    assert response.kind is OperatorResponseKind.ACCEPTED
    assert response.receipt is not None
    assert response.receipt.selected_products == (1, 2, 3)
    assert created[0].products == (1, 2, 3)


def test_explicit_product_does_not_depend_on_visible_scan() -> None:
    coordinator, created = _coordinator(visible=())

    response = _agent(coordinator).handle("assemble product 4")

    assert response.kind is OperatorResponseKind.ACCEPTED
    assert created[0].products == (4,)


def test_unsafe_language_is_rejected_without_controller_creation() -> None:
    coordinator, created = _coordinator()

    response = _agent(coordinator).handle("move tcp to 400, 0, -300")

    assert response.kind is OperatorResponseKind.REJECTED
    assert created == []


def test_second_task_is_held_until_first_is_quality_released() -> None:
    coordinator, _ = _coordinator()
    agent = _agent(coordinator)
    assert agent.handle("assemble product 1").kind is OperatorResponseKind.ACCEPTED

    response = agent.handle("assemble product 2")

    assert response.kind is OperatorResponseKind.HELD
    assert "quality-released" in response.message


def test_live_report_preserves_t09_state_step_event_and_source() -> None:
    coordinator, created = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    controller = created[0]
    controller.state_machine.current_record = StateRecord(PackMLState.HELD, "missing spare", 1, TIMESTAMP)
    controller.events = (ControllerEvent("andon", 1, None, "missing spare"),)
    controller.plan_records = (_record(),)

    response = agent.handle("show cell status")

    assert response.report is not None
    assert response.report.current_step_id == "product-1-step-01-attempt-1"
    assert response.report.event_count == 1
    assert response.report.active_fault == "missing spare"
    assert response.report.execution_authorized is False
    assert response.report.as_dict()["source"] == "mock"


@pytest.mark.parametrize(
    "state",
    [
        PackMLState.HELD,
        PackMLState.SUSPENDED,
        PackMLState.COMPLETE,
        PackMLState.STOPPING,
        PackMLState.STOPPED,
        PackMLState.ABORTING,
        PackMLState.ABORTED,
        PackMLState.RESETTING,
    ],
)
def test_non_active_packml_state_never_authorizes_execution(state: PackMLState) -> None:
    coordinator, created = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    created[0].state_machine.current_record = StateRecord(state, "operator hold", 1, TIMESTAMP)

    report = agent.handle("status").report

    assert report is not None
    assert report.execution_authorized is False
    if state is not PackMLState.COMPLETE:
        assert report.active_fault == "operator hold"


def test_mechanical_completion_remains_pending_until_final_check() -> None:
    coordinator, created = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))

    before = agent.handle("status")
    after = agent.evaluate_final_build(_request())

    assert before.report is not None
    assert before.report.controller_finished is True
    assert before.report.quality_release is QualityRelease.PENDING
    assert before.report.execution_authorized is False
    assert after.kind is OperatorResponseKind.FINAL_CHECK
    assert after.report is not None
    assert after.report.quality_release is QualityRelease.RELEASED
    assert after.report.execution_authorized is False
    assert after.final_check is not None
    assert after.final_check.outcome is BuildCheckOutcome.DONE
    evidence = after.report.as_dict()["final_checks"][0]  # type: ignore[index]
    assert evidence["task_id"] == "task-1"  # type: ignore[index]
    assert evidence["observed_at"] == "2026-10-29T15:00:00Z"  # type: ignore[index]
    assert evidence["evidence_source"] == "cosmos_reason_2"  # type: ignore[index]


def test_final_check_before_mechanical_completion_is_held() -> None:
    coordinator, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")

    response = agent.evaluate_final_build(_request())

    assert response.kind is OperatorResponseKind.HELD
    assert "mechanically complete" in response.message


def test_failed_final_check_blocks_quality_release() -> None:
    coordinator, created = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))
    request = _request()
    request = replace(request, installed_object_ids=("M1_01", "M1_02", "M1_04"))

    response = agent.evaluate_final_build(request)

    assert response.report is not None
    assert response.report.quality_release is QualityRelease.BLOCKED
    assert response.report.execution_authorized is False
    assert response.final_check is not None
    assert response.final_check.outcome is BuildCheckOutcome.REBUILD
    replayed = agent.evaluate_final_build(request)
    assert replayed.kind is OperatorResponseKind.HELD
    assert "fresh image evidence" in replayed.message


def test_cosmos_unavailability_is_a_held_response_not_success() -> None:
    coordinator, created = _coordinator()
    agent = OperatorAgent(coordinator, FinalBuildGate(UnavailableCosmos()))
    agent.handle("assemble product 1")
    _complete(created[0], (1,))

    response = agent.evaluate_final_build(_request())

    assert response.kind is OperatorResponseKind.HELD
    assert response.report is not None
    assert response.report.quality_release is QualityRelease.BLOCKED
    assert response.final_check is not None
    assert response.final_check.outcome is BuildCheckOutcome.HOLD_ANDON
    assert response.final_check.evidence_source is None


@pytest.mark.parametrize(("changes", "state_timestamp"), [
    ({"task_id": "old-task"}, TIMESTAMP), ({"observed_at": NOW - timedelta(seconds=1)}, TIMESTAMP),
    ({"observed_at": NOW + timedelta(days=1)}, TIMESTAMP), ({}, "2026-10-29T15:00:10Z")])
def test_final_check_replay_or_impossible_time_is_rejected(changes: dict[str, object], state_timestamp: str) -> None:
    coordinator, created = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))
    created[0].state_machine.current_record = replace(created[0].state_machine.current_record, timestamp=state_timestamp)

    response = agent.evaluate_final_build(replace(_request(), **changes))

    assert response.kind is OperatorResponseKind.HELD
    assert "active task" in response.message or "predates" in response.message
    assert response.report is None


def test_visible_scan_and_factory_failures_are_held() -> None:
    def bad_visible() -> tuple[int, ...]:
        raise RuntimeError("camera offline")

    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: FakeController(products),
        visible_products=bad_visible,
        clock=lambda: NOW,
        task_id_factory=lambda: "task-1",
    )
    response = _agent(coordinator).handle("assemble everything on the table")
    assert response.kind is OperatorResponseKind.HELD
    assert "visible-product scan failed" in response.message

    def bad_factory(products: tuple[int, ...]) -> FakeController:
        raise RuntimeError(f"cannot build {products}")

    coordinator, _ = _coordinator(factory=bad_factory)
    response = _agent(coordinator).handle("assemble product 1")
    assert response.kind is OperatorResponseKind.HELD
    assert "controller creation failed" in response.message


@pytest.mark.parametrize("visible", [(), (1, 1), (0,), (True,)])
def test_malformed_visible_products_fail_closed(visible: tuple[int, ...]) -> None:
    coordinator, created = _coordinator(visible=visible)

    response = _agent(coordinator).handle("assemble everything on the table")

    assert response.kind is OperatorResponseKind.HELD
    assert created == []


def test_invalid_controller_report_is_held() -> None:
    coordinator, created = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    created[0].finished = 1  # type: ignore[assignment]

    response = agent.handle("status")

    assert response.kind is OperatorResponseKind.HELD
    assert "finished must be a boolean" in response.message


def test_done_decision_requires_both_checks() -> None:
    coordinator, created = _coordinator()
    _agent(coordinator).handle("assemble product 1")
    _complete(created[0], (1,))
    valid = FinalBuildGate(PassingCosmos()).evaluate(_request())
    invalid = replace(valid, cosmos_passed=False)

    with pytest.raises(OperatorTaskPortError, match="both independent"):
        coordinator.record_final_check(invalid)


@pytest.mark.parametrize(
    "changes",
    [
        {"evidence_source": None},
        {"model_id": None},
        {"cosmos_passed": None},
        {"evidence_source": CosmosEvidenceSource.STUB},
    ],
)
def test_partial_or_substituted_cosmos_provenance_is_rejected(changes: dict[str, object]) -> None:
    coordinator, created = _coordinator()
    _agent(coordinator).handle("assemble product 1")
    _complete(created[0], (1,))
    valid = FinalBuildGate(PassingCosmos()).evaluate(_request())

    with pytest.raises(OperatorTaskPortError, match="checks|provenance|identify"):
        coordinator.record_final_check(replace(valid, **changes))


def test_released_final_check_cannot_be_overwritten() -> None:
    coordinator, created = _coordinator()
    _agent(coordinator).handle("assemble product 1")
    _complete(created[0], (1,))
    decision = FinalBuildGate(PassingCosmos()).evaluate(_request())
    coordinator.record_final_check(decision)

    with pytest.raises(OperatorTaskPortError, match="cannot be overwritten"):
        coordinator.record_final_check(decision)


def test_task_id_and_clock_boundaries_fail_closed() -> None:
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: FakeController(products),
        visible_products=lambda: (1,),
        clock=lambda: datetime(2026, 10, 29, 15),
        task_id_factory=lambda: "../unsafe",
    )

    response = _agent(coordinator).handle("assemble product 1")

    assert response.kind is OperatorResponseKind.HELD
    assert "unsafe" in response.message


def test_completed_product_pauses_multi_product_task_until_final_check() -> None:
    coordinator, created = _coordinator(visible=(1, 2))
    agent = _agent(coordinator)
    agent.handle("assemble everything on the table")
    controller = created[0]
    controller.completed_products = (1,)
    controller.state_machine.current_record = StateRecord(PackMLState.EXECUTE, None, 2, TIMESTAMP)

    report = agent.handle("status").report

    assert report is not None
    assert report.quality_release is QualityRelease.PENDING
    assert report.execution_authorized is False
    assert report.active_fault == "final recipe and Cosmos Reason 2 checks are pending"

    checked = agent.evaluate_final_build(_request(1)).report

    assert checked is not None
    assert checked.quality_release is QualityRelease.IN_PROGRESS
    assert checked.execution_authorized is True


def test_actual_t09_controller_satisfies_task_port_read_boundary() -> None:
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _build_controller(MockPerception()),
        visible_products=lambda: (1, 3, 4),
        clock=lambda: datetime.now(UTC),
        task_id_factory=lambda: "actual-t09",
    )
    agent = _agent(coordinator)

    accepted = agent.handle("assemble everything on the table")
    report = agent.handle("status")

    assert accepted.kind is OperatorResponseKind.ACCEPTED
    assert report.report is not None
    assert report.report.packml["state"] == "Idle"
    assert report.report.selected_products == (1, 3, 4)
    assert report.report.execution_authorized is True


def test_finished_controller_must_report_all_selected_products() -> None:
    coordinator, created = _coordinator(visible=(1, 2))
    agent = _agent(coordinator)
    agent.handle("assemble everything on the table")
    controller = created[0]
    controller.completed_products = (1,)
    controller.finished = True
    controller.state_machine.current_record = StateRecord(PackMLState.COMPLETE, None, 1, TIMESTAMP)

    response = agent.handle("status")

    assert response.kind is OperatorResponseKind.HELD
    assert "every selected product" in response.message


@pytest.mark.parametrize("record", [
    StateRecord(PackMLState.EXECUTE, None, 1, "not-a-timestamp"), StateRecord(PackMLState.EXECUTE, None, None, TIMESTAMP),
    StateRecord(PackMLState.IDLE, None, 1, TIMESTAMP), StateRecord(PackMLState.EXECUTE, None, 1, "2026-10-29T15:00:01Z")])
def test_malformed_packml_record_is_held(record: StateRecord) -> None:
    coordinator, created = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    created[0].state_machine.current_record = record

    response = agent.handle("status")

    assert response.kind is OperatorResponseKind.HELD
    assert "T09" in response.message or "Idle" in response.message


def test_new_task_is_allowed_only_after_quality_release_with_unique_id() -> None:
    ids = iter(("task-1", "task-2"))
    created: list[FakeController] = []

    def factory(products: tuple[int, ...]) -> FakeController:
        controller = FakeController(products)
        created.append(controller)
        return controller

    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=factory,
        visible_products=lambda: (1,),
        clock=lambda: NOW,
        task_id_factory=lambda: next(ids),
    )
    agent = _agent(coordinator)
    first = agent.handle("assemble product 1")
    _complete(created[0], (1,))
    agent.evaluate_final_build(_request())

    second = agent.handle("assemble product 2")

    assert first.receipt is not None and first.receipt.task_id == "task-1"
    assert second.receipt is not None and second.receipt.task_id == "task-2"
    assert created[1].products == (2,)
