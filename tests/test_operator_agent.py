"""Tests for operator agent.
Authored by Shritej Gunda.
"""

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agent.authority import TrialInstructionEvidence
from agent.cosmos_reason import (
    BuildCheckOutcome,
    CosmosEvidenceSource,
    CosmosReasonResult,
    CosmosReasonUnavailable,
    FinalBuildGate,
    FinalBuildRequest,
    InstalledInstanceLineage,
)
from agent.operator import OperatorAgent, OperatorResponseKind
from agent.task_evidence import FinalEvidenceQuery
from agent.task_port import ControllerSource, QualityRelease, T09TaskCoordinator
from assembly.catalog import load_catalog
from assembly.controller_ports import ControllerEvent
from assembly.planner import ContractPoseUF0, PlanStepRecord
from assembly.state_machine import PackMLState, StateRecord

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)
TIMESTAMP = "2026-10-29T15:00:00Z"
DIGEST = "a" * 64


class FakeStateMachine:
    def __init__(self, state: StateRecord | None = None) -> None:
        self.current_record = state or StateRecord(PackMLState.IDLE, None, None, TIMESTAMP)


class FakeController:
    def __init__(self, products: tuple[int, ...]) -> None:
        self.products = products
        self.requested_products = products
        self.completed_products: tuple[int, ...] = ()
        self.events: tuple[ControllerEvent, ...] = ()
        self.plan_records: tuple[PlanStepRecord, ...] = ()
        self.finished = False
        self.state_machine = FakeStateMachine()


class PassingCosmos:
    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        return CosmosReasonResult(
            request.run_id,
            request.rollout,
            request.task_id,
            request.request_id,
            request.product,
            request.image_capture_id,
            request.image_sha256,
            True,
            "visual build matches",
            "Cosmos-Reason2-2B",
            CosmosEvidenceSource.COSMOS_REASON_2,
            request.observed_at,
        )


class UnavailableCosmos:
    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        raise CosmosReasonUnavailable(f"no result for {request.request_id}")


class FailingCosmos:
    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        return replace(PassingCosmos().check(request), passed=False, explanation="build mismatch")


class EvidenceProvider:
    def __init__(self) -> None:
        self.changes: dict[str, object] = {}
        self.request_id = "request-1"
        self.calls: list[FinalEvidenceQuery] = []

    def capture(self, query: FinalEvidenceQuery) -> FinalBuildRequest:
        self.calls.append(query)
        lineage = tuple(
            InstalledInstanceLineage(f"physical-{query.product}-{index}", object_id)
            for index, object_id in enumerate(query.expected_object_ids, start=1)
        )
        values: dict[str, object] = {
            "run_id": query.run_id,
            "rollout": query.rollout,
            "task_id": query.task_id,
            "request_id": self.request_id,
            "product": query.product,
            "installed_instances": lineage,
            "image_capture_id": f"capture-{self.request_id}",
            "image_sha256": DIGEST,
            "observed_at": NOW,
            "expires_at": NOW + timedelta(minutes=1),
            "evidence_source": "trusted-final-port",
        }
        values.update(self.changes)
        return FinalBuildRequest(**values)  # type: ignore[arg-type]


def _trial(products: tuple[int, ...]) -> TrialInstructionEvidence:
    listed = ", ".join(str(product) for product in products)
    return TrialInstructionEvidence(
        "run-1", 7, "observation-1", f"assemble products {listed}", NOW, NOW + timedelta(minutes=1), "trial-port"
    )


def _coordinator(
    *,
    products: tuple[int, ...] = (1,),
    factory: object | None = None,
    evidence: EvidenceProvider | None = None,
    cosmos: object | None = None,
) -> tuple[T09TaskCoordinator, list[FakeController], EvidenceProvider]:
    created: list[FakeController] = []
    final_evidence = evidence or EvidenceProvider()

    def default_factory(selected: tuple[int, ...]) -> FakeController:
        controller = FakeController(selected)
        created.append(controller)
        return controller

    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=factory or default_factory,  # type: ignore[arg-type]
        trial_instruction=lambda: _trial(products),
        final_evidence=final_evidence,
        final_gate=FinalBuildGate(cosmos or PassingCosmos(), load_catalog()),  # type: ignore[arg-type]
        clock=lambda: NOW,
        task_id_factory=lambda: "task-1",
    )
    return coordinator, created, final_evidence


def _agent(coordinator: T09TaskCoordinator) -> OperatorAgent:
    return OperatorAgent(coordinator)


def _complete(controller: FakeController, products: tuple[int, ...]) -> None:
    controller.completed_products = products
    controller.finished = True
    controller.state_machine.current_record = StateRecord(PackMLState.COMPLETE, None, products[-1], TIMESTAMP)


def _record(product: int = 1) -> PlanStepRecord:
    pose = ContractPoseUF0(400.0, 0.0, -300.0, 0.0)
    return PlanStepRecord(
        f"product-{product}-step-01-attempt-1", product, "part-1", f"M{product}_01", pose, pose, 0.0, 1
    )


def test_status_before_task_is_read_only_idle_report() -> None:
    coordinator, created, _ = _coordinator()
    response = _agent(coordinator).handle("status")
    assert response.kind is OperatorResponseKind.REPORT
    assert response.report is not None and response.report.packml["state"] == "Idle"
    assert response.report.source is ControllerSource.MOCK
    assert created == []


def test_assemble_everything_uses_prompt_products_in_prompt_order() -> None:
    coordinator, created, _ = _coordinator(products=(3, 1, 2))
    response = _agent(coordinator).handle("assemble everything on the table")
    assert response.kind is OperatorResponseKind.ACCEPTED
    assert response.receipt is not None and response.receipt.selected_products == (3, 1, 2)
    assert created[0].products == (3, 1, 2)


def test_explicit_product_must_agree_with_prompt() -> None:
    coordinator, created, _ = _coordinator(products=(4,))
    response = _agent(coordinator).handle("assemble product 4")
    assert response.kind is OperatorResponseKind.ACCEPTED
    assert created[0].products == (4,)


def test_unsafe_language_is_rejected_without_controller_creation() -> None:
    coordinator, created, _ = _coordinator()
    response = _agent(coordinator).handle("move tcp to 400, 0, -300")
    assert response.kind is OperatorResponseKind.REJECTED
    assert created == []


def test_second_task_is_held_until_first_is_quality_released() -> None:
    coordinator, _, _ = _coordinator()
    agent = _agent(coordinator)
    assert agent.handle("assemble product 1").kind is OperatorResponseKind.ACCEPTED
    response = agent.handle("assemble product 2")
    assert response.kind is OperatorResponseKind.HELD
    assert "quality-released" in response.message


def test_live_report_preserves_t09_state_step_event_and_source() -> None:
    coordinator, created, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    controller = created[0]
    controller.state_machine.current_record = StateRecord(PackMLState.HELD, "missing spare", 1, TIMESTAMP)
    controller.events = (ControllerEvent("andon", 1, None, "missing spare"),)
    controller.plan_records = (_record(),)
    response = agent.handle("show cell status")
    assert response.report is not None
    assert response.report.current_step_id == "product-1-step-01-attempt-1"
    assert response.report.event_count == 1 and response.report.active_fault == "missing spare"
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
    coordinator, created, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    created[0].state_machine.current_record = StateRecord(state, "operator hold", 1, TIMESTAMP)
    report = agent.handle("status").report
    assert report is not None and report.execution_authorized is False
    if state is not PackMLState.COMPLETE:
        assert report.active_fault == "operator hold"


def test_mechanical_completion_remains_pending_until_trusted_final_check() -> None:
    coordinator, created, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))
    before = agent.handle("status")
    after = agent.evaluate_final_build(1)
    assert before.report is not None and before.report.quality_release is QualityRelease.PENDING
    assert before.report.execution_authorized is False
    assert after.kind is OperatorResponseKind.FINAL_CHECK
    assert after.report is not None and after.report.quality_release is QualityRelease.RELEASED
    assert after.final_check is not None and after.final_check.outcome is BuildCheckOutcome.DONE
    evidence = after.report.as_dict()["final_checks"][0]  # type: ignore[index]
    assert evidence["task_id"] == "task-1"  # type: ignore[index]
    assert evidence["evidence_source"] == "cosmos_reason_2"  # type: ignore[index]


def test_final_check_before_mechanical_completion_is_held() -> None:
    coordinator, _, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    response = agent.evaluate_final_build(1)
    assert response.kind is OperatorResponseKind.HELD
    assert "mechanically complete" in response.message


def test_wrong_installed_lineage_is_held_before_quality_gate() -> None:
    provider = EvidenceProvider()
    provider.changes["installed_instances"] = (
        InstalledInstanceLineage("physical-1", "M1_01"),
        InstalledInstanceLineage("physical-2", "M1_02"),
    )
    coordinator, created, _ = _coordinator(evidence=provider)
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))
    response = agent.evaluate_final_build(1)
    assert response.kind is OperatorResponseKind.HELD
    assert "lineage" in response.message and response.report is None


def test_reused_final_evidence_cannot_release_after_rebuild() -> None:
    coordinator, created, provider = _coordinator(cosmos=FailingCosmos())
    failing = _agent(coordinator)
    failing.handle("assemble product 1")
    _complete(created[0], (1,))
    first = failing.evaluate_final_build(1)
    replayed = _agent(coordinator).evaluate_final_build(1)
    assert first.final_check is not None and first.final_check.outcome is BuildCheckOutcome.REBUILD
    assert replayed.kind is OperatorResponseKind.HELD
    assert "already used" in replayed.message and provider.calls


def test_cosmos_unavailability_is_a_held_response_not_success() -> None:
    coordinator, created, _ = _coordinator(cosmos=UnavailableCosmos())
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))
    response = agent.evaluate_final_build(1)
    assert response.kind is OperatorResponseKind.HELD
    assert response.report is not None and response.report.quality_release is QualityRelease.BLOCKED
    assert response.final_check is not None
    assert response.final_check.outcome is BuildCheckOutcome.HOLD_ANDON
    assert response.final_check.evidence_source is None


def test_stale_final_evidence_is_held() -> None:
    provider = EvidenceProvider()
    provider.changes.update(observed_at=NOW - timedelta(minutes=2), expires_at=NOW - timedelta(minutes=1))
    coordinator, created, _ = _coordinator(evidence=provider)
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))
    response = agent.evaluate_final_build(1)
    assert response.kind is OperatorResponseKind.HELD
    assert "older than mechanical state" in response.message or "stale" in response.message


def test_prompt_and_factory_failures_are_held() -> None:
    def bad_prompt() -> TrialInstructionEvidence:
        raise RuntimeError("trial observation offline")

    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: FakeController(products),
        trial_instruction=bad_prompt,
        clock=lambda: NOW,
        task_id_factory=lambda: "task-1",
    )
    response = _agent(coordinator).handle("assemble everything on the table")
    assert response.kind is OperatorResponseKind.HELD
    assert "trial instruction evidence failed" in response.message

    def bad_factory(products: tuple[int, ...]) -> FakeController:
        raise RuntimeError(f"cannot build {products}")

    coordinator, _, _ = _coordinator(factory=bad_factory)
    response = _agent(coordinator).handle("assemble product 1")
    assert response.kind is OperatorResponseKind.HELD
    assert "controller creation failed" in response.message


def test_invalid_controller_report_is_held() -> None:
    coordinator, created, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    created[0].finished = 1  # type: ignore[assignment]
    response = agent.handle("status")
    assert response.kind is OperatorResponseKind.HELD
    assert "finished must be a boolean" in response.message


def test_controller_product_drift_from_prompt_is_held() -> None:
    coordinator, created, _ = _coordinator(products=(1, 2))
    agent = _agent(coordinator)
    agent.handle("assemble everything on the table")
    created[0].requested_products = (2, 1)
    response = agent.handle("status")
    assert response.kind is OperatorResponseKind.HELD
    assert "trial instruction" in response.message


def test_released_final_check_cannot_be_overwritten() -> None:
    coordinator, created, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    _complete(created[0], (1,))
    assert agent.evaluate_final_build(1).kind is OperatorResponseKind.FINAL_CHECK
    replayed = agent.evaluate_final_build(1)
    assert replayed.kind is OperatorResponseKind.HELD
    assert "already used" in replayed.message or "overwritten" in replayed.message


def test_task_id_and_clock_boundaries_fail_closed() -> None:
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: FakeController(products),
        trial_instruction=lambda: _trial((1,)),
        clock=lambda: datetime(2026, 10, 29, 15),
        task_id_factory=lambda: "../unsafe",
    )
    response = _agent(coordinator).handle("assemble product 1")
    assert response.kind is OperatorResponseKind.HELD
    assert "timezone-aware" in response.message or "unsafe" in response.message


def test_completed_product_pauses_mock_report_until_final_check() -> None:
    coordinator, created, _ = _coordinator(products=(1, 2))
    agent = _agent(coordinator)
    agent.handle("assemble everything on the table")
    controller = created[0]
    controller.completed_products = (1,)
    controller.state_machine.current_record = StateRecord(PackMLState.EXECUTE, None, 2, TIMESTAMP)
    report = agent.handle("status").report
    assert report is not None and report.quality_release is QualityRelease.PENDING
    assert report.execution_authorized is False
    assert report.active_fault == "final recipe and Cosmos Reason 2 checks are pending"
    checked = agent.evaluate_final_build(1).report
    assert checked is not None and checked.quality_release is QualityRelease.IN_PROGRESS
    assert checked.execution_authorized is True


def test_finished_controller_must_report_all_selected_products() -> None:
    coordinator, created, _ = _coordinator(products=(1, 2))
    agent = _agent(coordinator)
    agent.handle("assemble everything on the table")
    controller = created[0]
    controller.completed_products = (1,)
    controller.finished = True
    controller.state_machine.current_record = StateRecord(PackMLState.COMPLETE, None, 1, TIMESTAMP)
    response = agent.handle("status")
    assert response.kind is OperatorResponseKind.HELD
    assert "every selected product" in response.message


@pytest.mark.parametrize(
    "record",
    [
        StateRecord(PackMLState.EXECUTE, None, 1, "not-a-timestamp"),
        StateRecord(PackMLState.EXECUTE, None, None, TIMESTAMP),
        StateRecord(PackMLState.IDLE, None, 1, TIMESTAMP),
        StateRecord(PackMLState.EXECUTE, None, 1, "2026-10-29T15:00:01Z"),
    ],
)
def test_malformed_packml_record_is_held(record: StateRecord) -> None:
    coordinator, created, _ = _coordinator()
    agent = _agent(coordinator)
    agent.handle("assemble product 1")
    created[0].state_machine.current_record = record
    response = agent.handle("status")
    assert response.kind is OperatorResponseKind.HELD
    assert "T09" in response.message or "Idle" in response.message


def test_new_task_requires_release_and_fresh_prompt_authority() -> None:
    ids = iter(("task-1", "task-2"))
    prompts = iter((_trial((1,)), replace(_trial((2,)), observation_id="observation-2")))
    created: list[FakeController] = []
    evidence = EvidenceProvider()

    def factory(products: tuple[int, ...]) -> FakeController:
        controller = FakeController(products)
        created.append(controller)
        return controller

    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=factory,
        trial_instruction=lambda: next(prompts),
        final_evidence=evidence,
        final_gate=FinalBuildGate(PassingCosmos(), load_catalog()),
        clock=lambda: NOW,
        task_id_factory=lambda: next(ids),
    )
    agent = _agent(coordinator)
    first = agent.handle("assemble product 1")
    _complete(created[0], (1,))
    agent.evaluate_final_build(1)
    evidence.request_id = "request-2"
    second = agent.handle("assemble product 2")
    assert first.receipt is not None and first.receipt.task_id == "task-1"
    assert second.receipt is not None and second.receipt.task_id == "task-2"
    assert created[1].products == (2,)
