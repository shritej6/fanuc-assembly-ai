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
)

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)


def _request(**changes: object) -> FinalBuildRequest:
    values = {
        "task_id": "task-1",
        "request_id": "run-1-product-1",
        "product": 1,
        "installed_object_ids": ("M1_01", "M1_02", "M1_03", "M1_04"),
        "image_refs": ("eval/run-1/product-1-side.png",),
        "observed_at": NOW,
    }
    values.update(changes)
    return FinalBuildRequest(**values)  # type: ignore[arg-type]


def _result(request: FinalBuildRequest, passed: bool = True, **changes: object) -> CosmosReasonResult:
    values = {
        "task_id": request.task_id,
        "request_id": request.request_id,
        "product": request.product,
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


def test_both_checks_pass_before_done() -> None:
    request = _request()
    port = FakeCosmos(_result(request))

    decision = FinalBuildGate(port).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.DONE
    assert decision.recipe_passed is True
    assert decision.cosmos_passed is True
    assert decision.evidence_source is CosmosEvidenceSource.COSMOS_REASON_2
    assert decision.model_id == "Cosmos-Reason2-2B"
    assert port.calls == [request]


def test_recipe_failure_rebuilds_without_calling_cosmos() -> None:
    request = _request(installed_object_ids=("M1_01", "M1_02", "M1_04"))
    port = FakeCosmos(RuntimeError("must not be called"))

    decision = FinalBuildGate(port).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.REBUILD
    assert decision.cosmos_passed is None
    assert decision.evidence_source is None
    assert port.calls == []


def test_cosmos_failure_requests_rebuild() -> None:
    request = _request()

    decision = FinalBuildGate(FakeCosmos(_result(request, False))).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.REBUILD
    assert decision.recipe_passed is True
    assert decision.cosmos_passed is False


@pytest.mark.parametrize(
    "response",
    [
        CosmosReasonUnavailable("endpoint offline"),
        RuntimeError("transport failed"),
        object(),
    ],
)
def test_unavailable_or_malformed_cosmos_holds_andon(response: object) -> None:
    decision = FinalBuildGate(FakeCosmos(response)).evaluate(_request())

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is None
    assert "Cosmos Reason 2" in decision.detail
    assert "endpoint offline" not in decision.detail
    assert "transport failed" not in decision.detail


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "other"},
        {"task_id": "task-other"},
        {"product": 2},
        {"observed_at": NOW + timedelta(seconds=1)},
        {"model_id": "other-model"},
    ],
)
def test_mismatched_or_substituted_evidence_holds(changes: dict[str, object]) -> None:
    request = _request()
    response = _result(request, **changes)

    decision = FinalBuildGate(FakeCosmos(response)).evaluate(request)

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is None


def test_stub_can_never_quality_release_product() -> None:
    request = _request()
    gate = FinalBuildGate(DeterministicCosmosReasonStub({request.request_id: True}))

    decision = gate.evaluate(request)

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is True
    assert decision.evidence_source is CosmosEvidenceSource.STUB
    assert decision.model_id == "deterministic-stub"
    assert decision.detail == "stub evidence cannot release a product"


def test_missing_stub_result_holds() -> None:
    decision = FinalBuildGate(DeterministicCosmosReasonStub({})).evaluate(_request())

    assert decision.outcome is BuildCheckOutcome.HOLD_ANDON
    assert decision.cosmos_passed is None


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": "../unsafe"},
        {"product": True},
        {"image_refs": ()},
        {"image_refs": ("",)},
        {"observed_at": datetime(2026, 10, 29, 15)},
    ],
)
def test_request_boundary_rejects_malformed_evidence(changes: dict[str, object]) -> None:
    with pytest.raises(FinalBuildCheckError):
        _request(**changes)


def test_gate_rejects_wrong_request_type() -> None:
    with pytest.raises(FinalBuildCheckError, match="FinalBuildRequest"):
        FinalBuildGate(DeterministicCosmosReasonStub({})).evaluate(object())  # type: ignore[arg-type]
