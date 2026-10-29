"""Poka-yoke final-build gate: deterministic recipe check plus Cosmos Reason 2."""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

from assembly.catalog import Catalog

LOGGER = logging.getLogger(__name__)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


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
class InstalledInstanceLineage:
    instance_id: str
    object_id: str

    def __post_init__(self) -> None:
        _safe_id(self.instance_id, "instance_id")
        _safe_id(self.object_id, "object_id")


@dataclass(frozen=True, slots=True)
class FinalBuildRequest:
    run_id: str
    rollout: int
    task_id: str
    request_id: str
    product: int
    installed_instances: tuple[InstalledInstanceLineage, ...]
    image_capture_id: str
    image_sha256: str
    observed_at: datetime
    expires_at: datetime
    evidence_source: str

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _rollout(self.rollout)
        _safe_id(self.task_id, "task_id")
        _safe_id(self.request_id, "request_id")
        _product(self.product)
        if not isinstance(self.installed_instances, tuple) or any(
            type(item) is not InstalledInstanceLineage for item in self.installed_instances
        ):
            raise FinalBuildCheckError("installed_instances must contain trusted lineage values")
        _safe_id(self.image_capture_id, "image_capture_id")
        _sha256(self.image_sha256)
        observed = _aware(self.observed_at, "observed_at")
        expires = _aware(self.expires_at, "expires_at")
        if expires < observed:
            raise FinalBuildCheckError("final-build evidence expires before capture")
        object.__setattr__(self, "observed_at", observed)
        object.__setattr__(self, "expires_at", expires)
        _safe_id(self.evidence_source, "evidence_source")

    @property
    def installed_object_ids(self) -> tuple[str, ...]:
        return tuple(item.object_id for item in self.installed_instances)


@dataclass(frozen=True, slots=True)
class CosmosReasonResult:
    run_id: str
    rollout: int
    task_id: str
    request_id: str
    product: int
    image_capture_id: str
    image_sha256: str
    passed: bool
    explanation: str
    model_id: str
    evidence_source: CosmosEvidenceSource
    observed_at: datetime

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "Cosmos run_id")
        _rollout(self.rollout)
        _safe_id(self.task_id, "Cosmos task_id")
        _safe_id(self.request_id, "request_id")
        _product(self.product)
        _safe_id(self.image_capture_id, "Cosmos image_capture_id")
        _sha256(self.image_sha256)
        if type(self.passed) is not bool:
            raise FinalBuildCheckError("Cosmos passed must be a boolean")
        _bounded_text(self.explanation, "Cosmos explanation")
        _safe_id(self.model_id, "Cosmos model_id")
        if not isinstance(self.evidence_source, CosmosEvidenceSource):
            raise FinalBuildCheckError("Cosmos evidence_source is invalid")
        object.__setattr__(self, "observed_at", _aware(self.observed_at, "Cosmos observed_at"))


class CosmosReasonPort(Protocol):
    def check(self, request: FinalBuildRequest) -> CosmosReasonResult: ...


@dataclass(frozen=True, slots=True)
class FinalGateDecision:
    run_id: str
    rollout: int
    product: int
    task_id: str
    request_id: str
    observed_at: datetime
    expires_at: datetime
    installed_instances: tuple[InstalledInstanceLineage, ...]
    image_capture_id: str
    image_sha256: str
    build_evidence_source: str
    outcome: BuildCheckOutcome
    recipe_passed: bool
    cosmos_passed: bool | None
    evidence_source: CosmosEvidenceSource | None
    model_id: str | None
    detail: str


class FinalBuildGate:
    """Require both checks; stub evidence can never quality-release a product."""

    def __init__(self, cosmos: CosmosReasonPort, catalog: Catalog) -> None:
        if not isinstance(catalog, Catalog):
            raise FinalBuildCheckError("final build gate requires a validated Catalog")
        self._cosmos = cosmos
        self._catalog = catalog

    def evaluate(self, request: FinalBuildRequest) -> FinalGateDecision:
        if not isinstance(request, FinalBuildRequest):
            raise FinalBuildCheckError("final build request must be a FinalBuildRequest")
        expected = tuple(step.object_id for step in self._catalog.product(request.product).steps)
        recipe_passed = request.installed_object_ids == expected
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
            request.run_id,
            request.rollout,
            request.task_id,
            request.request_id,
            request.product,
            request.image_capture_id,
            request.image_sha256,
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
        result.run_id != request.run_id
        or result.rollout != request.rollout
        or result.task_id != request.task_id
        or result.request_id != request.request_id
        or result.product != request.product
        or result.image_capture_id != request.image_capture_id
        or result.image_sha256 != request.image_sha256
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
        request.run_id,
        request.rollout,
        request.product,
        request.task_id,
        request.request_id,
        request.observed_at,
        request.expires_at,
        request.installed_instances,
        request.image_capture_id,
        request.image_sha256,
        request.evidence_source,
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


def _rollout(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FinalBuildCheckError("rollout must be a nonnegative integer")
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FinalBuildCheckError("image_sha256 must be 64 lowercase hexadecimal characters")
    return value


def _product(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
        raise FinalBuildCheckError("product must be an integer from 1 through 6")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FinalBuildCheckError(f"{field} must be a non-empty string")
    return value


def _bounded_text(value: object, field: str) -> str:
    text = _nonempty(value, field)
    if len(text) > 2048 or any(unicodedata.category(character).startswith("C") for character in text):
        raise FinalBuildCheckError(f"{field} must be control-free text no longer than 2048 characters")
    return text


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise FinalBuildCheckError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)
