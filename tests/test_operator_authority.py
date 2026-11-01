"""Regression tests for prompt, evidence, and physical-runtime authority.
Authored by Shritej Gunda.
"""

from datetime import UTC, datetime, timedelta

import pytest

from agent.authority import TrialInstructionEvidence
from agent.commands import AssemblyScope, TaskRequest
from agent.cosmos_reason import (
    CosmosEvidenceSource,
    CosmosReasonResult,
    FinalBuildGate,
    FinalBuildRequest,
    InstalledInstanceLineage,
)
from agent.operator import OperatorAgent, OperatorResponseKind
from agent.task_evidence import (
    FinalEvidenceQuery,
    TrustedBuildEvidenceError,
    capture_final_build_request,
)
from agent.task_port import (
    ControllerSource,
    FinalBuildEvaluation,
    LiveCellReport,
    OperatorTaskPortError,
    QualityRelease,
    T09TaskCoordinator,
)
from assembly.catalog import load_catalog
from robot.cell.t09_mock_runtime import MockPerception, _build_controller

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)
DIGEST = "a" * 64


def _trial(
    instruction: str = "assemble products 1",
    *,
    captured_at: datetime = NOW,
    expires_at: datetime = NOW + timedelta(minutes=1),
) -> TrialInstructionEvidence:
    return TrialInstructionEvidence(
        "run-1", 7, "observation-1", instruction, captured_at, expires_at, "trusted-trial-port"
    )


def _lineage(product: int = 1) -> tuple[InstalledInstanceLineage, ...]:
    return tuple(
        InstalledInstanceLineage(f"physical-{index}", step.object_id)
        for index, step in enumerate(load_catalog().product(product).steps, start=1)
    )


def _request(**changes: object) -> FinalBuildRequest:
    values = {
        "run_id": "run-1",
        "rollout": 7,
        "task_id": "task-1",
        "request_id": "request-1",
        "product": 1,
        "installed_instances": _lineage(),
        "image_capture_id": "capture-1",
        "image_sha256": DIGEST,
        "observed_at": NOW,
        "expires_at": NOW + timedelta(minutes=1),
        "evidence_source": "trusted-final-port",
    }
    values.update(changes)
    return FinalBuildRequest(**values)  # type: ignore[arg-type]


def _query() -> FinalEvidenceQuery:
    return FinalEvidenceQuery(
        "run-1",
        7,
        "task-1",
        1,
        tuple(step.object_id for step in load_catalog().product(1).steps),
        NOW,
    )


class _EvidenceProvider:
    def __init__(self, request: FinalBuildRequest) -> None:
        self.request = request
        self.queries: list[FinalEvidenceQuery] = []

    def capture(self, query: FinalEvidenceQuery) -> FinalBuildRequest:
        self.queries.append(query)
        return self.request


class _PassingCosmos:
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


class _SpyCosmos(_PassingCosmos):
    def __init__(self) -> None:
        self.calls: list[FinalBuildRequest] = []

    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        self.calls.append(request)
        return super().check(request)


class _TrustedTasks:
    def __init__(self, request: FinalBuildRequest) -> None:
        self.request = request
        self.calls = 0
        self.failure: Exception | None = None

    def submit(self, request: TaskRequest) -> object:
        raise AssertionError(request)

    def report(self) -> object:
        raise AssertionError("not used")

    def evaluate_final_build(self, product: int) -> FinalBuildEvaluation:
        assert product == 1
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        decision = FinalBuildGate(_PassingCosmos(), load_catalog()).evaluate(self.request)
        report = LiveCellReport(
            "task-1", ControllerSource.MOCK, NOW.isoformat(),
            {"state": "Complete", "reason": None, "product": 1, "timestamp": NOW.isoformat()},
            (1,), (1,), None, 0, True, False, QualityRelease.RELEASED, (decision,), None,
        )
        return FinalBuildEvaluation(decision, report)


def _coordinator(provider: object) -> T09TaskCoordinator:
    return T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _build_controller(MockPerception()),
        trial_instruction=provider,  # type: ignore[arg-type]
        clock=lambda: NOW,
    )


def test_cell_source_is_unavailable_without_reviewed_act_gate() -> None:
    with pytest.raises(OperatorTaskPortError, match="reviewed physical act gate"):
        T09TaskCoordinator(
            source=ControllerSource.CELL,
            controller_factory=lambda products: object(),
            trial_instruction=lambda: _trial(),
            clock=lambda: NOW,
        )


def test_explicit_intent_cannot_override_trial_product_authority() -> None:
    coordinator = _coordinator(lambda: _trial())

    with pytest.raises(OperatorTaskPortError, match="trial instruction"):
        coordinator.submit(TaskRequest(AssemblyScope.EXPLICIT, (2,)))


@pytest.mark.parametrize(
    "provider",
    [
        lambda: (1,),
        lambda: _trial(expires_at=NOW - timedelta(seconds=1), captured_at=NOW - timedelta(minutes=1)),
        lambda: _trial(captured_at=NOW + timedelta(seconds=1), expires_at=NOW + timedelta(minutes=1)),
    ],
)
def test_untyped_stale_or_future_prompt_evidence_fails_closed(provider: object) -> None:
    coordinator = _coordinator(provider)

    with pytest.raises(OperatorTaskPortError, match="trial instruction evidence"):
        coordinator.submit(TaskRequest(AssemblyScope.EXPLICIT, (1,)))


def test_missing_prompt_provider_fails_closed() -> None:
    def unavailable() -> TrialInstructionEvidence:
        raise RuntimeError("camera unavailable")

    coordinator = _coordinator(unavailable)

    with pytest.raises(OperatorTaskPortError, match="trial instruction evidence failed"):
        coordinator.submit(TaskRequest(AssemblyScope.EXPLICIT, (1,)))


@pytest.mark.parametrize(
    "changes",
    [
        {"instruction": "assemble products 1\n"},
        {"instruction": "assemble products 1" + "x" * 4096},
        {"source": "caller://untrusted"},
    ],
)
def test_trial_evidence_rejects_control_unbounded_or_unsafe_text(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "run_id": "run-1",
        "rollout": 7,
        "observation_id": "observation-1",
        "instruction": "assemble products 1",
        "captured_at": NOW,
        "expires_at": NOW + timedelta(minutes=1),
        "source": "trusted-trial-port",
    }
    values.update(changes)

    with pytest.raises(ValueError):
        TrialInstructionEvidence(**values)  # type: ignore[arg-type]


def test_caller_cannot_supply_self_asserted_final_build_evidence() -> None:
    request = _request(image_capture_id="capture-self-asserted")
    tasks = _TrustedTasks(_request())
    agent = OperatorAgent(tasks)

    response = agent.evaluate_final_build(request)

    assert response.kind is OperatorResponseKind.REJECTED
    assert "product ID" in response.message
    assert tasks.calls == 0


def test_forged_same_identity_done_cannot_reach_coordinator_gate() -> None:
    spy = _SpyCosmos()
    coordinator = T09TaskCoordinator(
        source=ControllerSource.MOCK,
        controller_factory=lambda products: _build_controller(MockPerception()),
        trial_instruction=lambda: _trial(),
        final_gate=FinalBuildGate(spy, load_catalog()),
        clock=lambda: NOW,
    )
    forged = FinalBuildGate(_PassingCosmos(), load_catalog()).evaluate(_request())

    response = OperatorAgent(coordinator).evaluate_final_build(forged)

    assert response.kind is OperatorResponseKind.REJECTED
    assert spy.calls == []
    assert not hasattr(coordinator, "prepare_final_build")
    assert not hasattr(coordinator, "record_final_check")


def test_arbitrary_image_uri_is_not_a_valid_capture_identity() -> None:
    with pytest.raises(ValueError, match="unsafe"):
        _request(image_capture_id="arbitrary://unverified-old-image")


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "other-run"},
        {"rollout": 8},
        {"task_id": "task-other"},
        {"product": 2},
    ],
)
def test_trusted_provider_must_match_run_rollout_task_and_product(changes: dict[str, object]) -> None:
    provider = _EvidenceProvider(_request(**changes))

    with pytest.raises(TrustedBuildEvidenceError, match="identity does not match"):
        capture_final_build_request(provider, _query(), NOW)


@pytest.mark.parametrize(
    "evidence",
    [
        _request(installed_instances=_lineage()[:-1]),
        _request(
            installed_instances=(
                InstalledInstanceLineage("same-instance", "M1_01"),
                InstalledInstanceLineage("same-instance", "M1_02"),
                *_lineage()[2:],
            )
        ),
        _request(
            installed_instances=(
                InstalledInstanceLineage("physical-1", "M1_02"),
                *_lineage()[1:],
            )
        ),
    ],
)
def test_trusted_provider_rejects_missing_duplicate_or_substituted_lineage(evidence: FinalBuildRequest) -> None:
    with pytest.raises(TrustedBuildEvidenceError, match="lineage"):
        capture_final_build_request(_EvidenceProvider(evidence), _query(), NOW)


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_at": NOW - timedelta(seconds=1)},
        {"observed_at": NOW + timedelta(seconds=1)},
        {"expires_at": NOW - timedelta(microseconds=1), "observed_at": NOW - timedelta(seconds=1)},
    ],
)
def test_trusted_provider_rejects_old_future_or_expired_evidence(changes: dict[str, object]) -> None:
    with pytest.raises(TrustedBuildEvidenceError, match="older|future|stale"):
        capture_final_build_request(_EvidenceProvider(_request(**changes)), _query(), NOW)


def test_operator_gets_final_build_evidence_only_from_task_port() -> None:
    tasks = _TrustedTasks(_request())
    agent = OperatorAgent(tasks)

    response = agent.evaluate_final_build(1)

    assert response.kind is OperatorResponseKind.FINAL_CHECK
    assert response.final_check is not None
    assert response.final_check.request_id == "request-1"
    assert tasks.calls == 1


def test_malformed_trusted_task_evidence_holds_instead_of_blames_operator() -> None:
    tasks = _TrustedTasks(_request())
    tasks.failure = RuntimeError("trusted evidence corrupt")
    agent = OperatorAgent(tasks)

    response = agent.evaluate_final_build(1)

    assert response.kind is OperatorResponseKind.HELD
    assert "evidence" in response.message
    assert tasks.calls == 1


def test_valid_trusted_evidence_path_preserves_capture_and_lineage() -> None:
    request = _request()
    provider = _EvidenceProvider(request)

    captured = capture_final_build_request(provider, _query(), NOW)

    assert captured is request
    assert captured.image_capture_id == "capture-1"
    assert captured.image_sha256 == DIGEST
    assert captured.installed_instances == _lineage()
    assert provider.queries == [_query()]
