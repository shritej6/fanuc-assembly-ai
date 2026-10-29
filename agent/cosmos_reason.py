"""Poka-yoke final-build gate: deterministic recipe check plus Cosmos Reason 2."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

LOGGER = logging.getLogger(__name__)


class FinalBuildCheckError(ValueError):
    """Final-build evidence is malformed or inconsistent."""


class CosmosReasonUnavailable(RuntimeError):
    """Cosmos Reason 2 did not return usable evidence for this request."""


class BuildCheckOutcome(StrEnum):
    DONE = "done"
    REBUILD = "rebuild"
    HOLD_ANDON = "hold_andon"


class CosmosEvidenceSource(StrEnum):
    COSMOS_REASON_2 = "cosmos_reason_2"
    STUB = "stub"


@dataclass(frozen=True, slots=True)
class FinalBuildRequest:
    task_id: str
    request_id: str
    product: int
    installed_object_ids: tuple[str, ...]
    image_refs: tuple[str, ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        _safe_id(self.task_id, "task_id")
        _safe_id(self.request_id, "request_id")
        _product(self.product)
        _string_tuple(self.installed_object_ids, "installed_object_ids", allow_empty=True)
        _string_tuple(self.image_refs, "image_refs", allow_empty=False)
        object.__setattr__(self, "observed_at", _aware(self.observed_at, "observed_at"))

    @property
    def expected_object_ids(self) -> tuple[str, ...]:
        return tuple(f"M{self.product}_{step:02d}" for step in range(1, 5))


@dataclass(frozen=True, slots=True)
class CosmosReasonResult:
    task_id: str
    request_id: str
    product: int
    passed: bool
    explanation: str
    model_id: str
    evidence_source: CosmosEvidenceSource
    observed_at: datetime

    def __post_init__(self) -> None:
        _safe_id(self.task_id, "Cosmos task_id")
        _safe_id(self.request_id, "request_id")
        _product(self.product)
        if type(self.passed) is not bool:
            raise FinalBuildCheckError("Cosmos passed must be a boolean")
        _nonempty(self.explanation, "Cosmos explanation")
        _nonempty(self.model_id, "Cosmos model_id")
        if not isinstance(self.evidence_source, CosmosEvidenceSource):
            raise FinalBuildCheckError("Cosmos evidence_source is invalid")
        object.__setattr__(self, "observed_at", _aware(self.observed_at, "Cosmos observed_at"))


class CosmosReasonPort(Protocol):
    def check(self, request: FinalBuildRequest) -> CosmosReasonResult: ...


@dataclass(frozen=True, slots=True)
class FinalGateDecision:
    product: int
    task_id: str
    request_id: str
    observed_at: datetime
    outcome: BuildCheckOutcome
    recipe_passed: bool
    cosmos_passed: bool | None
    evidence_source: CosmosEvidenceSource | None
    model_id: str | None
    detail: str


class FinalBuildGate:
    """Require both checks; stub evidence can never quality-release a product."""

    def __init__(self, cosmos: CosmosReasonPort) -> None:
        self._cosmos = cosmos

    def evaluate(self, request: FinalBuildRequest) -> FinalGateDecision:
        if not isinstance(request, FinalBuildRequest):
            raise FinalBuildCheckError("final build request must be a FinalBuildRequest")
        recipe_passed = request.installed_object_ids == request.expected_object_ids
        if not recipe_passed:
            return _decision(
                request, BuildCheckOutcome.REBUILD, False, None, None, None, "installed objects do not match the recipe"
            )
        try:
            result = self._cosmos.check(request)
            validated = _validate_result(request, result)
        except (FinalBuildCheckError, CosmosReasonUnavailable):
            LOGGER.warning("Cosmos Reason 2 returned no valid evidence", extra={"request_id": request.request_id})
            return _unavailable_decision(request, "Cosmos Reason 2 evidence unavailable")
        except Exception:
            LOGGER.exception("unexpected Cosmos Reason 2 port failure", extra={"request_id": request.request_id})
            return _unavailable_decision(request, "Cosmos Reason 2 failed unexpectedly")
        if validated.evidence_source is CosmosEvidenceSource.STUB:
            return _decision(
                request,
                BuildCheckOutcome.HOLD_ANDON,
                True,
                validated.passed,
                validated.evidence_source,
                validated.model_id,
                "stub evidence cannot release a product",
            )
        outcome = BuildCheckOutcome.DONE if validated.passed else BuildCheckOutcome.REBUILD
        return _decision(
            request,
            outcome,
            True,
            validated.passed,
            validated.evidence_source,
            validated.model_id,
            validated.explanation,
        )


class DeterministicCosmosReasonStub:
    """Explicit mock boundary; missing request IDs are unavailable, never guessed."""

    def __init__(self, results: Mapping[str, bool]) -> None:
        if not isinstance(results, Mapping) or any(
            not isinstance(key, str) or type(value) is not bool for key, value in results.items()
        ):
            raise FinalBuildCheckError("stub results must map string request IDs to booleans")
        self._results = dict(results)

    def check(self, request: FinalBuildRequest) -> CosmosReasonResult:
        try:
            passed = self._results[request.request_id]
        except KeyError as error:
            raise CosmosReasonUnavailable(f"no stub result for {request.request_id!r}") from error
        explanation = "configured mock pass" if passed else "configured mock failure"
        return CosmosReasonResult(
            request.task_id,
            request.request_id,
            request.product,
            passed,
            explanation,
            "deterministic-stub",
            CosmosEvidenceSource.STUB,
            request.observed_at,
        )


def _validate_result(request: FinalBuildRequest, result: object) -> CosmosReasonResult:
    if type(result) is not CosmosReasonResult:
        raise FinalBuildCheckError("Cosmos port must return exactly CosmosReasonResult")
    if (
        result.task_id != request.task_id
        or result.request_id != request.request_id
        or result.product != request.product
    ):
        raise FinalBuildCheckError("Cosmos response does not match the request")
    if result.observed_at != request.observed_at:
        raise FinalBuildCheckError("Cosmos response does not describe the current image evidence")
    if result.evidence_source is CosmosEvidenceSource.COSMOS_REASON_2 and result.model_id != "Cosmos-Reason2-2B":
        raise FinalBuildCheckError("production evidence must come from Cosmos Reason 2 (2B)")
    return result


def _decision(
    request: FinalBuildRequest,
    outcome: BuildCheckOutcome,
    recipe_passed: bool,
    cosmos_passed: bool | None,
    evidence_source: CosmosEvidenceSource | None,
    model_id: str | None,
    detail: str,
) -> FinalGateDecision:
    return FinalGateDecision(
        request.product,
        request.task_id,
        request.request_id,
        request.observed_at,
        outcome,
        recipe_passed,
        cosmos_passed,
        evidence_source,
        model_id,
        detail,
    )


def _unavailable_decision(request: FinalBuildRequest, detail: str) -> FinalGateDecision:
    return _decision(request, BuildCheckOutcome.HOLD_ANDON, True, None, None, None, detail)


def _safe_id(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if len(text) > 128 or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in text
    ):
        raise FinalBuildCheckError(f"{field} contains unsafe characters")
    return text


def _string_tuple(values: object, field: str, *, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(values, tuple) or (not allow_empty and not values):
        raise FinalBuildCheckError(f"{field} must be a {'possibly empty' if allow_empty else 'non-empty'} tuple")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise FinalBuildCheckError(f"{field} must contain non-empty strings")
    return values


def _product(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
        raise FinalBuildCheckError("product must be an integer from 1 through 6")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalBuildCheckError(f"{field} must be a non-empty string")
    return value


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise FinalBuildCheckError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)
