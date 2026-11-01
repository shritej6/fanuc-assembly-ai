from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from agent.cosmos_reason import (
    BuildCheckOutcome,
    CosmosEvidenceSource,
    CosmosReasonResult,
    CosmosReasonUnavailable,
    DeterministicCosmosReasonStub,
    FinalBuildCheckError,
    FinalBuildGate,
    FinalBuildRequest,
    InstalledInstanceLineage,
)
from assembly.catalog import load_catalog

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)
DIGEST = "a" * 64


def _lineage(product: int = 1) -> tuple[InstalledInstanceLineage, ...]:
    recipe = load_catalog().product(product)
    return tuple(
        InstalledInstanceLineage(f"physical-{product}-{index}", step.object_id)
        for index, step in enumerate(recipe.steps, start=1)
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
        "evidence_source": "trusted-test-provider",
    }
    values.update(changes)
    return FinalBuildRequest(**values)  # type: ignore[arg-type]


def _result(request: FinalBuildRequest, passed: bool = True, **changes: object) -> CosmosReasonResult:
    values = {
        "run_id": request.run_id,
        "rollout": request.rollout,
        "task_id": request.task_id,
        "request_id": request.request_id,
        "product": request.product,
        "image_capture_id": request.image_capture_id,
        "image_sha256": request.image_sha256,
        "passed": passed,
        "explanation": "visual build matches" if passed else "upper guide is not seated",
        "model_id": "Cosmos-Reason2-2B",
        "evidence_source": CosmosEvidenceSource.COSMOS_REASON_2,
        "observed_at": request.observed_at,
    }
    values.update(changes)
    return CosmosReasonResult(**values)  # type: ignore[arg-type]


class FakeCosmos:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[FinalBuildRequest] = []

    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        self.calls.append(request)
        if isinstance(self.response, Exception):
            raise self.response
        return self.response  # type: ignore[return-value]


def _gate(port: object) -> FinalBuildGate:
    return FinalBuildGate(port, load_catalog())  # type: ignore[arg-type]


def test_both_checks_pass_before_done_and_preserve_evidence_lineage() -> None:
    request = _request()
    port = FakeCosmos(_result(request))

    decision = _gate(port).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.DONE
    assert decision.recipe_passed is True
    assert decision.cosmos_passed is True
    assert decision.evidence_source is CosmosEvidenceSource.COSMOS_REASON_2
    assert decision.model_id == "Cosmos-Reason2-2B"
    assert decision.run_id == request.run_id
    assert decision.rollout == request.rollout
    assert decision.installed_instances == request.installed_instances
    assert decision.image_capture_id == request.image_capture_id
    assert decision.image_sha256 == request.image_sha256
    assert port.calls == [request]


def test_recipe_failure_rebuilds_without_calling_cosmos() -> None:
    request = _request(installed_instances=_lineage()[:-1])
    port = FakeCosmos(RuntimeError("must not be called"))

    decision = _gate(port).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.REBUILD
    assert decision.cosmos_passed is None
    assert decision.evidence_source is None
    assert port.calls == []


def test_cosmos_failure_requests_rebuild() -> None:
    request = _request()

    decision = _gate(FakeCosmos(_result(request, False))).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.REBUILD
    assert decision.recipe_passed is True
    assert decision.cosmos_passed is False


@pytest.mark.parametrize(
    "response",
    [CosmosReasonUnavailable("endpoint offline"), RuntimeError("transport failed"), object()],
)
def test_unavailable_or_malformed_cosmos_holds_andon(response: object) -> None:
    decision = _gate(FakeCosmos(response)).evaluate(_request())

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is None
    assert "Cosmos Reason 2" in decision.detail
    assert "endpoint offline" not in decision.detail
    assert "transport failed" not in decision.detail


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "other-run"},
        {"rollout": 8},
        {"request_id": "other-request"},
        {"task_id": "task-other"},
        {"product": 2},
        {"image_capture_id": "other-capture"},
        {"image_sha256": "b" * 64},
        {"observed_at": NOW + timedelta(seconds=1)},
        {"model_id": "other-model"},
    ],
)
def test_mismatched_or_substituted_evidence_holds(changes: dict[str, object]) -> None:
    request = _request()

    decision = _gate(FakeCosmos(_result(request, **changes))).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is None


def test_stub_can_never_quality_release_product() -> None:
    request = _request()
    gate = _gate(DeterministicCosmosReasonStub({request.request_id: True}))

    decision = gate.evaluate(request)

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is True
    assert decision.evidence_source is CosmosEvidenceSource.STUB
    assert decision.model_id == "deterministic-stub"
    assert decision.detail == "stub evidence cannot release a product"


def test_missing_stub_result_holds() -> None:
    decision = _gate(DeterministicCosmosReasonStub({})).evaluate(_request())

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is None


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": "../unsafe"},
        {"rollout": True},
        {"request_id": "../unsafe"},
        {"product": True},
        {"installed_instances": (object(),)},
        {"image_capture_id": ""},
        {"image_sha256": "not-a-digest"},
        {"evidence_source": "untrusted://caller-label"},
        {"observed_at": datetime(2026, 10, 29, 15)},
        {"expires_at": NOW - timedelta(seconds=1)},
    ],
)
def test_request_boundary_rejects_malformed_evidence(changes: dict[str, object]) -> None:
    with pytest.raises(FinalBuildCheckError):
        _request(**changes)


def test_gate_rejects_wrong_request_type() -> None:
    with pytest.raises(FinalBuildCheckError, match="FinalBuildRequest"):
        _gate(DeterministicCosmosReasonStub({})).evaluate(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("explanation", ["line one\nline two", "x" * 2049])
def test_cosmos_result_rejects_unbounded_or_control_text(explanation: str) -> None:
    request = _request()

    with pytest.raises(FinalBuildCheckError, match="explanation"):
        _result(request, explanation=explanation)


def test_final_request_is_immutable() -> None:
    request = _request()

    with pytest.raises((AttributeError, TypeError)):
        request.run_id = "other"  # type: ignore[misc]

    assert replace(request, request_id="request-2").request_id == "request-2"
