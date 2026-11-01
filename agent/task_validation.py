"""Fail-closed validation helpers for the operator-to-T09 boundary."""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime

from agent.cosmos_reason import BuildCheckOutcome, CosmosEvidenceSource, FinalGateDecision
from assembly.controller_ports import ControllerEvent
from assembly.planner import PlanStepRecord
from assembly.state_machine import PackMLState, StateRecord

LOGGER = logging.getLogger(__name__)


class OperatorTaskPortError(RuntimeError):
    """Task submission or live controller reporting failed safely."""


def validate_final_decision(decision: FinalGateDecision) -> None:
    product(decision.product)
    safe_id(decision.run_id, "run ID")
    safe_id(decision.task_id, "task ID")
    safe_id(decision.request_id, "request ID")
    safe_id(decision.image_capture_id, "image capture ID")
    _validate_decision_times(decision)
    if not isinstance(decision.outcome, BuildCheckOutcome):
        raise OperatorTaskPortError("final-check outcome is invalid")
    if type(decision.recipe_passed) is not bool:
        raise OperatorTaskPortError("final-check evidence flags are invalid")
    if decision.cosmos_passed is not None and type(decision.cosmos_passed) is not bool:
        raise OperatorTaskPortError("final-check evidence flags are invalid")
    if not _bounded_text(decision.detail):
        raise OperatorTaskPortError("final-check detail must be control-free text no longer than 2048 characters")
    if decision.outcome is BuildCheckOutcome.DONE and not (decision.recipe_passed and decision.cosmos_passed is True):
        raise OperatorTaskPortError("DONE requires both independent checks to pass")
    _validate_cosmos_provenance(decision)


def validate_events(events: tuple[ControllerEvent, ...], selected: tuple[int, ...]) -> None:
    for event in events:
        if not isinstance(event.kind, str) or not event.kind.strip():
            raise OperatorTaskPortError("T09 event kind must be a non-empty string")
        if event.product is not None and event.product not in selected:
            raise OperatorTaskPortError("T09 event product is outside the selected task")
        if event.object_id is not None and (not isinstance(event.object_id, str) or not event.object_id.strip()):
            raise OperatorTaskPortError("T09 event object_id must be a string or null")
        if not isinstance(event.detail, str) or not event.detail.strip():
            raise OperatorTaskPortError("T09 event detail must be a non-empty string")


def validate_records(records: tuple[PlanStepRecord, ...], selected: tuple[int, ...]) -> None:
    for record in records:
        if record.product not in selected:
            raise OperatorTaskPortError("T09 plan record product is outside the selected task")
        safe_id(record.step_id, "step ID")


def validate_state_record(state: StateRecord, selected: tuple[int, ...]) -> None:
    if not isinstance(state.state, PackMLState):
        raise OperatorTaskPortError("T09 state must be a PackMLState")
    if state.reason is not None and not isinstance(state.reason, str):
        raise OperatorTaskPortError("T09 state reason must be a string or null")
    if (state.state is PackMLState.IDLE) != (state.product is None):
        raise OperatorTaskPortError("Idle must have no product and every non-Idle PackML state must own one product")
    if state.product is not None and state.product not in selected:
        raise OperatorTaskPortError("T09 state product is outside the selected task")
    _state_timestamp(state.timestamp)


def products(
    values: object,
    label: str,
    *,
    allow_empty: bool = False,
    preserve_order: bool = False,
) -> tuple[int, ...]:
    if not isinstance(values, tuple) or (not allow_empty and not values):
        qualification = "possibly empty" if allow_empty else "non-empty"
        raise OperatorTaskPortError(f"{label} must be a {qualification} tuple")
    if len(values) != len(set(values)):
        raise OperatorTaskPortError(f"{label} must be unique")
    for value in values:
        product(value)
    return values if preserve_order else tuple(sorted(values))


def product(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
        raise OperatorTaskPortError("product IDs must be integers from 1 through 6")
    return value


def safe_id(value: object, label: str = "task ID") -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not isinstance(value, str) or not value or len(value) > 128:
        raise OperatorTaskPortError(f"{label} must be a non-empty string no longer than 128 characters")
    if any(character not in allowed for character in value):
        raise OperatorTaskPortError(f"{label} contains unsafe characters")
    return value


def clock_value(clock: Callable[[], datetime], label: str) -> datetime:
    try:
        value = clock()
    except Exception as error:
        LOGGER.exception("operator clock failed", extra={"clock_label": label})
        raise OperatorTaskPortError(f"{label} failed; cell held") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise OperatorTaskPortError(f"{label} must return a timezone-aware datetime")
    return value.astimezone(UTC)


def iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def decision_dict(decision: FinalGateDecision) -> dict[str, object]:
    return {
        "run_id": decision.run_id,
        "rollout": decision.rollout,
        "product": decision.product,
        "task_id": decision.task_id,
        "request_id": decision.request_id,
        "observed_at": iso(decision.observed_at),
        "expires_at": iso(decision.expires_at),
        "installed_instances": [
            {"instance_id": item.instance_id, "object_id": item.object_id} for item in decision.installed_instances
        ],
        "image_capture_id": decision.image_capture_id,
        "image_sha256": decision.image_sha256,
        "build_evidence_source": decision.build_evidence_source,
        "outcome": decision.outcome.value,
        "recipe_passed": decision.recipe_passed,
        "cosmos_passed": decision.cosmos_passed,
        "evidence_source": None if decision.evidence_source is None else decision.evidence_source.value,
        "model_id": decision.model_id,
        "detail": decision.detail,
    }


def _validate_decision_times(decision: FinalGateDecision) -> None:
    observed = decision.observed_at
    expires = decision.expires_at
    if not isinstance(observed, datetime) or observed.tzinfo is None or observed.utcoffset() is None:
        raise OperatorTaskPortError("final-check observation time must include a timezone")
    if not isinstance(expires, datetime) or expires.tzinfo is None or expires.utcoffset() is None:
        raise OperatorTaskPortError("final-check expiry time must include a timezone")
    if expires < observed:
        raise OperatorTaskPortError("final-check evidence expires before it was captured")


def _validate_cosmos_provenance(decision: FinalGateDecision) -> None:
    source = decision.evidence_source
    model_id = decision.model_id
    absent = decision.cosmos_passed is None and source is None and model_id is None
    complete = type(decision.cosmos_passed) is bool and isinstance(source, CosmosEvidenceSource) and _text(model_id)
    if not (absent or complete):
        raise OperatorTaskPortError("final-check Cosmos evidence and provenance are inconsistent")
    if source is CosmosEvidenceSource.COSMOS_REASON_2 and model_id != "Cosmos-Reason2-2B":
        raise OperatorTaskPortError("Cosmos evidence must identify Cosmos Reason 2 (2B)")
    if source is CosmosEvidenceSource.STUB and model_id != "deterministic-stub":
        raise OperatorTaskPortError("stub evidence must identify the deterministic stub")
    if decision.outcome is BuildCheckOutcome.DONE and source is not CosmosEvidenceSource.COSMOS_REASON_2:
        raise OperatorTaskPortError("DONE requires Cosmos Reason 2 (2B) provenance")


def _state_timestamp(value: object) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise OperatorTaskPortError("T09 state timestamp must be ISO 8601") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise OperatorTaskPortError("T09 state timestamp must include a timezone")
    return timestamp


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _bounded_text(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and len(value) <= 2048
        and not any(unicodedata.category(character).startswith("C") for character in value)
    )
