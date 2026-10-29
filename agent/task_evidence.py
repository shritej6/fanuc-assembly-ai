"""Trusted final-build evidence boundary for the operator task coordinator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from agent.cosmos_reason import FinalBuildRequest, FinalGateDecision


class TrustedBuildEvidenceError(ValueError):
    """Final-build evidence is unavailable, malformed, stale, or mismatched."""


@dataclass(frozen=True, slots=True)
class FinalEvidenceQuery:
    run_id: str
    rollout: int
    task_id: str
    product: int
    expected_object_ids: tuple[str, ...]
    mechanical_at: datetime

    def __post_init__(self) -> None:
        _safe_id(self.run_id, "run_id")
        _nonnegative(self.rollout, "rollout")
        _safe_id(self.task_id, "task_id")
        if isinstance(self.product, bool) or not isinstance(self.product, int) or not 1 <= self.product <= 6:
            raise TrustedBuildEvidenceError("product must be an integer from 1 through 6")
        if not isinstance(self.expected_object_ids, tuple) or not self.expected_object_ids:
            raise TrustedBuildEvidenceError("expected_object_ids must be a non-empty tuple")
        for object_id in self.expected_object_ids:
            _safe_id(object_id, "expected object_id")
        object.__setattr__(self, "mechanical_at", _aware(self.mechanical_at, "mechanical_at"))


class FinalBuildEvidenceProvider(Protocol):
    def capture(self, query: FinalEvidenceQuery) -> FinalBuildRequest: ...


def capture_final_build_request(
    provider: FinalBuildEvidenceProvider | None,
    query: FinalEvidenceQuery,
    now: datetime,
) -> FinalBuildRequest:
    """Capture and validate immutable evidence from the trusted provider."""
    if provider is None:
        raise TrustedBuildEvidenceError("trusted final-build evidence provider is unavailable")
    try:
        request = provider.capture(query)
    except TrustedBuildEvidenceError:
        raise
    except Exception as error:
        raise TrustedBuildEvidenceError("trusted final-build evidence capture failed") from error
    if type(request) is not FinalBuildRequest:
        raise TrustedBuildEvidenceError("evidence provider must return exactly FinalBuildRequest")
    _validate_identity(request, query)
    _validate_lineage(request, query.expected_object_ids)
    current = _aware(now, "final evidence clock")
    if request.observed_at < query.mechanical_at or request.observed_at > current:
        raise TrustedBuildEvidenceError("final-build evidence is older than mechanical state or from the future")
    if current > request.expires_at:
        raise TrustedBuildEvidenceError("final-build evidence is stale")
    return request


def require_prepared_decision(decision: FinalGateDecision, request: FinalBuildRequest) -> None:
    """Reject decisions that did not preserve the prepared evidence lineage."""
    expected = (
        request.run_id,
        request.rollout,
        request.task_id,
        request.request_id,
        request.product,
        request.observed_at,
        request.expires_at,
        request.installed_instances,
        request.image_capture_id,
        request.image_sha256,
        request.evidence_source,
    )
    actual = (
        decision.run_id,
        decision.rollout,
        decision.task_id,
        decision.request_id,
        decision.product,
        decision.observed_at,
        decision.expires_at,
        decision.installed_instances,
        decision.image_capture_id,
        decision.image_sha256,
        decision.build_evidence_source,
    )
    if actual != expected:
        raise TrustedBuildEvidenceError("final decision does not preserve prepared evidence lineage")


def _validate_identity(request: FinalBuildRequest, query: FinalEvidenceQuery) -> None:
    expected = (query.run_id, query.rollout, query.task_id, query.product)
    actual = (request.run_id, request.rollout, request.task_id, request.product)
    if actual != expected:
        raise TrustedBuildEvidenceError("final-build run, rollout, task, or product identity does not match")


def _validate_lineage(request: FinalBuildRequest, expected_object_ids: tuple[str, ...]) -> None:
    if request.installed_object_ids != expected_object_ids:
        raise TrustedBuildEvidenceError("installed lineage does not match the catalog recipe")
    instance_ids = tuple(item.instance_id for item in request.installed_instances)
    if len(instance_ids) != len(set(instance_ids)):
        raise TrustedBuildEvidenceError("installed lineage contains duplicate physical instances")


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TrustedBuildEvidenceError(f"{field} must include a timezone")
    return value.astimezone(UTC)


def _safe_id(value: object, field: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not isinstance(value, str) or not value or len(value) > 128:
        raise TrustedBuildEvidenceError(f"{field} must be a safe 1-128 character identifier")
    if any(character not in allowed for character in value):
        raise TrustedBuildEvidenceError(f"{field} contains unsafe characters")
    return value


def _nonnegative(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TrustedBuildEvidenceError(f"{field} must be a nonnegative integer")
    return value
