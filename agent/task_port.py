"""Typed task boundary between operator intent and the T09 controller runtime."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Protocol
from uuid import uuid4

from agent.commands import AssemblyScope, TaskRequest
from agent.cosmos_reason import BuildCheckOutcome, CosmosEvidenceSource, FinalGateDecision
from assembly.controller_ports import ControllerEvent
from assembly.planner import PlanStepRecord
from assembly.state_machine import PackMLState, StateRecord

LOGGER = logging.getLogger(__name__)


class OperatorTaskPortError(RuntimeError):
    """Task submission or live controller reporting failed safely."""


class ControllerSource(StrEnum):
    MOCK = "mock"
    SIM = "sim"
    REPLAY = "replay"
    CELL = "cell"


class QualityRelease(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    IN_PROGRESS = "in_progress"
    PENDING = "pending"
    BLOCKED = "blocked"
    RELEASED = "released"


class ControllerView(Protocol):
    @property
    def completed_products(self) -> tuple[int, ...]: ...

    @property
    def events(self) -> tuple[ControllerEvent, ...]: ...

    @property
    def plan_records(self) -> tuple[PlanStepRecord, ...]: ...

    @property
    def finished(self) -> bool: ...

    @property
    def state_machine(self) -> object: ...


ControllerFactory = Callable[[tuple[int, ...]], ControllerView]
VisibleProductProvider = Callable[[], tuple[int, ...]]


class OperatorTaskPort(Protocol):
    def submit(self, request: TaskRequest) -> TaskReceipt: ...

    def report(self) -> LiveCellReport: ...

    def record_final_check(self, decision: FinalGateDecision) -> LiveCellReport: ...


@dataclass(frozen=True, slots=True)
class TaskReceipt:
    task_id: str
    source: ControllerSource
    selected_products: tuple[int, ...]
    accepted_at: str


@dataclass(frozen=True, slots=True)
class LiveCellReport:
    task_id: str | None
    source: ControllerSource
    reported_at: str
    packml: dict[str, object]
    selected_products: tuple[int, ...]
    completed_products: tuple[int, ...]
    current_step_id: str | None
    event_count: int
    controller_finished: bool
    execution_authorized: bool
    quality_release: QualityRelease
    final_checks: tuple[FinalGateDecision, ...]
    active_fault: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "source": self.source.value,
            "reported_at": self.reported_at,
            "packml": dict(self.packml),
            "selected_products": list(self.selected_products),
            "completed_products": list(self.completed_products),
            "current_step_id": self.current_step_id,
            "event_count": self.event_count,
            "controller_finished": self.controller_finished,
            "execution_authorized": self.execution_authorized,
            "quality_release": self.quality_release.value,
            "final_checks": [_decision_dict(decision) for decision in self.final_checks],
            "active_fault": self.active_fault,
        }


@dataclass(frozen=True, slots=True)
class _ControllerSnapshot:
    state: StateRecord
    completed: tuple[int, ...]
    events: tuple[ControllerEvent, ...]
    records: tuple[PlanStepRecord, ...]
    finished: bool


class T09TaskCoordinator:
    """Create T09 work from typed requests and expose only read-side status."""

    def __init__(
        self,
        *,
        source: ControllerSource,
        controller_factory: ControllerFactory,
        visible_products: VisibleProductProvider,
        clock: Callable[[], datetime] | None = None,
        task_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(source, ControllerSource):
            raise OperatorTaskPortError("source must be a ControllerSource")
        if not callable(controller_factory) or not callable(visible_products):
            raise OperatorTaskPortError("controller_factory and visible_products must be callable")
        self._source = source
        self._factory = controller_factory
        self._visible_products = visible_products
        self._clock = clock or _utc_now
        self._task_id_factory = task_id_factory or _new_task_id
        self._lock = RLock()
        self._controller: ControllerView | None = None
        self._active_task_id: str | None = None
        self._selected_products: tuple[int, ...] = ()
        self._final_checks: dict[int, FinalGateDecision] = {}
        self._used_task_ids: set[str] = set()

    def submit(self, request: TaskRequest) -> TaskReceipt:
        with self._lock:
            return self._submit(request)

    def _submit(self, request: TaskRequest) -> TaskReceipt:
        if not isinstance(request, TaskRequest):
            raise OperatorTaskPortError("task submission requires a TaskRequest")
        if self._controller is not None and not self._is_quality_released():
            raise OperatorTaskPortError("an active task must be quality-released before another task starts")
        products = self._resolve_products(request)
        task_id = self._next_task_id()
        try:
            controller = self._factory(products)
            _read_controller(controller, products)
        except OperatorTaskPortError:
            raise
        except Exception as error:
            LOGGER.exception("T09 controller creation failed", extra={"products": products})
            raise OperatorTaskPortError("T09 controller creation failed; cell held") from error
        accepted_at = _iso(_clock_value(self._clock, "task acceptance clock"))
        self._controller = controller
        self._active_task_id = task_id
        self._selected_products = products
        self._final_checks.clear()
        self._used_task_ids.add(task_id)
        return TaskReceipt(task_id, self._source, products, accepted_at)

    def report(self) -> LiveCellReport:
        with self._lock:
            return self._report()

    def _report(self) -> LiveCellReport:
        reported_at = _iso(_clock_value(self._clock, "report clock"))
        if self._controller is None:
            return _idle_report(self._source, reported_at)
        snapshot = _read_controller(self._controller, self._selected_products)
        release = _quality_release(snapshot, self._selected_products, self._final_checks)
        final_checks = tuple(self._final_checks[product] for product in sorted(self._final_checks))
        decision_order = tuple(self._final_checks.values())
        return LiveCellReport(
            self._active_task_id,
            self._source,
            reported_at,
            snapshot.state.as_dict(),
            self._selected_products,
            snapshot.completed,
            snapshot.records[-1].step_id if snapshot.records else None,
            len(snapshot.events),
            snapshot.finished,
            _execution_authorized(snapshot.state, release),
            release,
            final_checks,
            _active_fault(snapshot.state, release, decision_order),
        )

    def record_final_check(self, decision: FinalGateDecision) -> LiveCellReport:
        with self._lock:
            return self._record_final_check(decision)

    def _record_final_check(self, decision: FinalGateDecision) -> LiveCellReport:
        if type(decision) is not FinalGateDecision:
            raise OperatorTaskPortError("final check must be exactly FinalGateDecision")
        if decision.task_id != self._active_task_id:
            raise OperatorTaskPortError("final check does not belong to the active task")
        if self._controller is None or decision.product not in self._selected_products:
            raise OperatorTaskPortError("final check does not belong to the active task")
        snapshot = _read_controller(self._controller, self._selected_products)
        if decision.product not in snapshot.completed:
            raise OperatorTaskPortError("a product must be mechanically complete before its final check")
        _validate_final_decision(decision)
        existing = self._final_checks.get(decision.product)
        if existing is not None and existing.outcome is BuildCheckOutcome.DONE:
            raise OperatorTaskPortError("a quality-released product cannot be overwritten")
        self._final_checks.pop(decision.product, None)
        self._final_checks[decision.product] = decision
        try:
            return self._report()
        except Exception:
            self._final_checks.pop(decision.product, None)
            if existing is not None:
                self._final_checks[decision.product] = existing
            raise

    def _resolve_products(self, request: TaskRequest) -> tuple[int, ...]:
        if request.scope is AssemblyScope.EXPLICIT:
            return request.products
        try:
            products = self._visible_products()
        except Exception as error:
            LOGGER.exception("visible-product scan failed")
            raise OperatorTaskPortError("visible-product scan failed; cell held") from error
        return _products(products, "visible products")

    def _next_task_id(self) -> str:
        try:
            value = self._task_id_factory()
        except Exception as error:
            LOGGER.exception("task ID generation failed")
            raise OperatorTaskPortError("task ID generation failed; cell held") from error
        task_id = _safe_id(value)
        if task_id in self._used_task_ids:
            raise OperatorTaskPortError("task ID factory returned a previously used ID")
        return task_id

    def _is_quality_released(self) -> bool:
        if self._controller is None:
            return True
        snapshot = _read_controller(self._controller, self._selected_products)
        release = _quality_release(snapshot, self._selected_products, self._final_checks)
        return release is QualityRelease.RELEASED


def _read_controller(controller: object, selected: tuple[int, ...]) -> _ControllerSnapshot:
    try:
        state = controller.state_machine.current_record
        completed = controller.completed_products
        events = controller.events
        records = controller.plan_records
        finished = controller.finished
    except Exception as error:
        LOGGER.exception("T09 controller status failed")
        raise OperatorTaskPortError("T09 controller status failed; cell held") from error
    if type(state) is not StateRecord:
        raise OperatorTaskPortError("T09 current state must be exactly StateRecord")
    completed = _products(completed, "completed products", allow_empty=True, preserve_order=True)
    if any(product not in selected for product in completed):
        raise OperatorTaskPortError("T09 reported a completed product outside the selected task")
    if not isinstance(events, tuple) or any(type(event) is not ControllerEvent for event in events):
        raise OperatorTaskPortError("T09 events must be a tuple of ControllerEvent")
    if not isinstance(records, tuple) or any(type(record) is not PlanStepRecord for record in records):
        raise OperatorTaskPortError("T09 plan records must be a tuple of PlanStepRecord")
    _validate_events(events, selected)
    _validate_records(records, selected)
    if type(finished) is not bool:
        raise OperatorTaskPortError("T09 finished must be a boolean")
    if finished and (completed != selected or state.state is not PackMLState.COMPLETE):
        raise OperatorTaskPortError("finished T09 task must report every selected product and Complete state")
    _validate_state_record(state, selected)
    return _ControllerSnapshot(state, completed, events, records, finished)


def _quality_release(
    snapshot: _ControllerSnapshot,
    selected: tuple[int, ...],
    decisions: dict[int, FinalGateDecision],
) -> QualityRelease:
    outcomes = {decision.outcome for decision in decisions.values()}
    if BuildCheckOutcome.REBUILD in outcomes or BuildCheckOutcome.HOLD_ANDON in outcomes:
        return QualityRelease.BLOCKED
    released = {product for product, decision in decisions.items() if decision.outcome is BuildCheckOutcome.DONE}
    if set(snapshot.completed) - released:
        return QualityRelease.PENDING
    if not snapshot.finished:
        return QualityRelease.IN_PROGRESS
    return QualityRelease.RELEASED if released == set(selected) else QualityRelease.PENDING


def _active_fault(
    state: StateRecord,
    release: QualityRelease,
    decisions: tuple[FinalGateDecision, ...],
) -> str | None:
    fault_states = {PackMLState.HELD, PackMLState.SUSPENDED, PackMLState.STOPPED, PackMLState.ABORTED}
    if state.state in fault_states:
        return state.reason or f"cell is {state.state.value} without a reason"
    if release is QualityRelease.PENDING:
        return "final recipe and Cosmos Reason 2 checks are pending"
    if release is QualityRelease.BLOCKED:
        blocked = next(decision for decision in reversed(decisions) if decision.outcome is not BuildCheckOutcome.DONE)
        return blocked.detail
    return None


def _execution_authorized(state: StateRecord, release: QualityRelease) -> bool:
    active_states = {PackMLState.IDLE, PackMLState.STARTING, PackMLState.EXECUTE, PackMLState.COMPLETING}
    return release is QualityRelease.IN_PROGRESS and state.state in active_states


def _validate_final_decision(decision: FinalGateDecision) -> None:
    _product(decision.product)
    _safe_id(decision.task_id)
    _safe_id(decision.request_id)
    if not isinstance(decision.observed_at, datetime) or decision.observed_at.tzinfo is None:
        raise OperatorTaskPortError("final-check observation time must include a timezone")
    if not isinstance(decision.outcome, BuildCheckOutcome):
        raise OperatorTaskPortError("final-check outcome is invalid")
    if type(decision.recipe_passed) is not bool:
        raise OperatorTaskPortError("final-check evidence flags are invalid")
    if decision.cosmos_passed is not None and type(decision.cosmos_passed) is not bool:
        raise OperatorTaskPortError("final-check evidence flags are invalid")
    if not isinstance(decision.detail, str) or not decision.detail.strip():
        raise OperatorTaskPortError("final-check detail must be a non-empty string")
    if decision.outcome is BuildCheckOutcome.DONE and not (decision.recipe_passed and decision.cosmos_passed is True):
        raise OperatorTaskPortError("DONE requires both independent checks to pass")
    _validate_cosmos_provenance(decision)
    if decision.evidence_source is CosmosEvidenceSource.COSMOS_REASON_2 and decision.model_id != "Cosmos-Reason2-2B":
        raise OperatorTaskPortError("Cosmos evidence must identify Cosmos Reason 2 (2B)")
    if decision.evidence_source is CosmosEvidenceSource.STUB and decision.model_id != "deterministic-stub":
        raise OperatorTaskPortError("stub evidence must identify the deterministic stub")
    if decision.outcome is BuildCheckOutcome.DONE and (
        decision.evidence_source is not CosmosEvidenceSource.COSMOS_REASON_2 or decision.model_id != "Cosmos-Reason2-2B"
    ):
        raise OperatorTaskPortError("DONE requires Cosmos Reason 2 (2B) provenance")


def _validate_cosmos_provenance(decision: FinalGateDecision) -> None:
    source = decision.evidence_source
    model_id = decision.model_id
    absent = decision.cosmos_passed is None and source is None and model_id is None
    complete = (
        type(decision.cosmos_passed) is bool
        and isinstance(source, CosmosEvidenceSource)
        and isinstance(model_id, str)
        and bool(model_id.strip())
    )
    if not (absent or complete):
        raise OperatorTaskPortError("final-check Cosmos evidence and provenance are inconsistent")


def _validate_events(events: tuple[ControllerEvent, ...], selected: tuple[int, ...]) -> None:
    for event in events:
        if not isinstance(event.kind, str) or not event.kind.strip():
            raise OperatorTaskPortError("T09 event kind must be a non-empty string")
        if event.product is not None and event.product not in selected:
            raise OperatorTaskPortError("T09 event product is outside the selected task")
        if event.object_id is not None and (not isinstance(event.object_id, str) or not event.object_id.strip()):
            raise OperatorTaskPortError("T09 event object_id must be a string or null")
        if not isinstance(event.detail, str) or not event.detail.strip():
            raise OperatorTaskPortError("T09 event detail must be a non-empty string")


def _validate_records(records: tuple[PlanStepRecord, ...], selected: tuple[int, ...]) -> None:
    for record in records:
        if record.product not in selected:
            raise OperatorTaskPortError("T09 plan record product is outside the selected task")
        _safe_id(record.step_id)


def _validate_state_record(state: StateRecord, selected: tuple[int, ...]) -> None:
    if not isinstance(state.state, PackMLState):
        raise OperatorTaskPortError("T09 state must be a PackMLState")
    if state.reason is not None and not isinstance(state.reason, str):
        raise OperatorTaskPortError("T09 state reason must be a string or null")
    if state.product is not None and state.product not in selected:
        raise OperatorTaskPortError("T09 state product is outside the selected task")
    try:
        timestamp = datetime.fromisoformat(state.timestamp)
    except (TypeError, ValueError) as error:
        raise OperatorTaskPortError("T09 state timestamp must be ISO 8601") from error
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise OperatorTaskPortError("T09 state timestamp must include a timezone")


def _products(
    values: object,
    label: str,
    *,
    allow_empty: bool = False,
    preserve_order: bool = False,
) -> tuple[int, ...]:
    if not isinstance(values, tuple) or (not allow_empty and not values):
        raise OperatorTaskPortError(f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} tuple")
    if len(values) != len(set(values)):
        raise OperatorTaskPortError(f"{label} must be unique")
    for value in values:
        _product(value)
    return values if preserve_order else tuple(sorted(values))


def _product(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
        raise OperatorTaskPortError("product IDs must be integers from 1 through 6")
    return value


def _idle_report(source: ControllerSource, timestamp: str) -> LiveCellReport:
    packml = {"state": PackMLState.IDLE.value, "reason": None, "product": None, "timestamp": timestamp}
    return LiveCellReport(
        None,
        source,
        timestamp,
        packml,
        (),
        (),
        None,
        0,
        False,
        False,
        QualityRelease.NOT_APPLICABLE,
        (),
        None,
    )


def _decision_dict(decision: FinalGateDecision) -> dict[str, object]:
    return {
        "product": decision.product,
        "task_id": decision.task_id,
        "request_id": decision.request_id,
        "observed_at": _iso(decision.observed_at),
        "outcome": decision.outcome.value,
        "recipe_passed": decision.recipe_passed,
        "cosmos_passed": decision.cosmos_passed,
        "evidence_source": None if decision.evidence_source is None else decision.evidence_source.value,
        "model_id": decision.model_id,
        "detail": decision.detail,
    }


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise OperatorTaskPortError("task ID must be a non-empty string no longer than 128 characters")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if any(character not in allowed for character in value):
        raise OperatorTaskPortError("task ID contains unsafe characters")
    return value


def _clock_value(clock: Callable[[], datetime], label: str) -> datetime:
    try:
        value = clock()
    except Exception as error:
        LOGGER.exception("operator clock failed", extra={"clock_label": label})
        raise OperatorTaskPortError(f"{label} failed; cell held") from error
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise OperatorTaskPortError(f"{label} must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_task_id() -> str:
    return f"task-{uuid4().hex}"
