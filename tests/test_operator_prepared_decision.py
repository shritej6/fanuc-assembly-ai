from datetime import UTC, datetime, timedelta

import pytest

from agent.authority import TrialInstructionEvidence
from agent.commands import AssemblyScope, TaskRequest
from agent.cosmos_reason import (
    BuildCheckOutcome,
    CosmosEvidenceSource,
    CosmosReasonResult,
    FinalBuildGate,
    FinalBuildRequest,
    InstalledInstanceLineage,
)
from agent.task_evidence import FinalEvidenceQuery
from agent.task_port import ControllerSource, OperatorTaskPortError, QualityRelease, T09TaskCoordinator
from assembly.catalog import load_catalog
from assembly.state_machine import PackMLState, StateRecord

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)


class _StateMachine:
    current_record = StateRecord(PackMLState.COMPLETE, None, 1, NOW.isoformat())


class _CompleteController:
    requested_products = (1,)
    completed_products = (1,)
    events = ()
    plan_records = ()
    finished = True
    state_machine = _StateMachine()


class _EvidenceProvider:
    def __init__(self, expires_at: datetime) -> None:
        self.expires_at = expires_at
        self.request_id = "request-1"
        self.capture_id = "capture-1"

    def capture(self, query: FinalEvidenceQuery) -> FinalBuildRequest:
        lineage = tuple(
            InstalledInstanceLineage(f"physical-{index}", object_id)
            for index, object_id in enumerate(query.expected_object_ids, start=1)
        )
        return FinalBuildRequest(
            query.run_id, query.rollout, query.task_id, self.request_id, query.product,
            lineage, self.capture_id, "a" * 64, NOW, self.expires_at, "trusted-final-port",
        )


class _Cosmos:
    def __init__(self, after_check: object | None = None) -> None:
        self.calls: list[FinalBuildRequest] = []
        self.after_check = after_check

    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        self.calls.append(request)
        if callable(self.after_check):
            self.after_check()
        return CosmosReasonResult(
            request.run_id, request.rollout, request.task_id, request.request_id, request.product,
            request.image_capture_id, request.image_sha256, True, "visual build matches",
            "Cosmos-Reason2-2B", CosmosEvidenceSource.COSMOS_REASON_2, request.observed_at,
        )


def _coordinator(
    cosmos: _Cosmos,
    current: list[datetime],
    *,
    expires_at: datetime = NOW + timedelta(minutes=1),
) -> T09TaskCoordinator:
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _CompleteController(),
        trial_instruction=lambda: TrialInstructionEvidence(
            "run-1", 7, "observation-1", "assemble products 1",
            NOW, NOW + timedelta(minutes=1), "trial-port",
        ),
        final_evidence=_EvidenceProvider(expires_at),
        final_gate=FinalBuildGate(cosmos, load_catalog()),
        clock=lambda: current[0],
        task_id_factory=lambda: "task-1",
    )
    coordinator.submit(TaskRequest(AssemblyScope.EXPLICIT, (1,)))
    return coordinator


def test_coordinator_owns_gate_and_records_verified_decision_atomically() -> None:
    current = [NOW]
    cosmos = _Cosmos()
    coordinator = _coordinator(cosmos, current)

    evaluation = coordinator.evaluate_final_build(1)

    assert len(cosmos.calls) == 1
    assert evaluation.decision.outcome is BuildCheckOutcome.DONE
    assert evaluation.report.quality_release is QualityRelease.RELEASED
    assert evaluation.report.final_checks == (evaluation.decision,)


def test_evidence_expiring_during_cosmos_check_is_not_recorded() -> None:
    current = [NOW]
    cosmos = _Cosmos(lambda: current.__setitem__(0, NOW + timedelta(seconds=2)))
    coordinator = _coordinator(cosmos, current, expires_at=NOW + timedelta(seconds=1))

    with pytest.raises(OperatorTaskPortError, match="stale"):
        coordinator.evaluate_final_build(1)

    assert len(cosmos.calls) == 1
    assert coordinator.report().quality_release is QualityRelease.PENDING


def test_expired_attempt_is_discarded_so_fresh_capture_can_retry() -> None:
    current = [NOW]
    cosmos = _Cosmos(lambda: current.__setitem__(0, NOW + timedelta(seconds=2)))
    provider = _EvidenceProvider(NOW + timedelta(seconds=1))
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _CompleteController(),
        trial_instruction=lambda: TrialInstructionEvidence(
            "run-1", 7, "observation-1", "assemble products 1",
            NOW, NOW + timedelta(minutes=1), "trial-port",
        ),
        final_evidence=provider,
        final_gate=FinalBuildGate(cosmos, load_catalog()),
        clock=lambda: current[0],
        task_id_factory=lambda: "task-1",
    )
    coordinator.submit(TaskRequest(AssemblyScope.EXPLICIT, (1,)))
    with pytest.raises(OperatorTaskPortError, match="stale"):
        coordinator.evaluate_final_build(1)
    provider.request_id = "request-2"
    provider.capture_id = "capture-2"
    provider.expires_at = NOW + timedelta(seconds=3)

    retried = coordinator.evaluate_final_build(1)

    assert len(cosmos.calls) == 2
    assert retried.report.quality_release is QualityRelease.RELEASED


def test_missing_coordinator_gate_holds_quality_release() -> None:
    current = [NOW]
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _CompleteController(),
        trial_instruction=lambda: TrialInstructionEvidence(
            "run-1", 7, "observation-1", "assemble products 1",
            NOW, NOW + timedelta(minutes=1), "trial-port",
        ),
        final_evidence=_EvidenceProvider(NOW + timedelta(minutes=1)),
        clock=lambda: current[0],
        task_id_factory=lambda: "task-1",
    )
    coordinator.submit(TaskRequest(AssemblyScope.EXPLICIT, (1,)))

    with pytest.raises(OperatorTaskPortError, match="gate.*unavailable"):
        coordinator.evaluate_final_build(1)
