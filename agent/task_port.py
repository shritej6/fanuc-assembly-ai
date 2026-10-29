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

from agent.authority import AuthorityEvidenceError, TrialInstructionEvidence, authorized_trial_products
from agent.commands import AssemblyScope, TaskRequest
from agent.cosmos_reason import (
    BuildCheckOutcome,
    FinalBuildCheckError,
    FinalBuildGate,
    FinalBuildRequest,
    FinalGateDecision,
)
from agent.task_evidence import (
    FinalBuildEvidenceProvider,
    FinalEvidenceQuery,
    TrustedBuildEvidenceError,
    capture_final_build_request,
    require_prepared_decision,
)
from agent.task_validation import (
    OperatorTaskPortError,
    clock_value,
    decision_dict,
    iso,
    products,
    safe_id,
    validate_events,
    validate_final_decision,
    validate_records,
    validate_state_record,
)
from agent.task_validation import product as validate_product
from assembly.catalog import Catalog, load_catalog
from assembly.controller_ports import ControllerEvent
from assembly.planner import PlanStepRecord
from assembly.state_machine import PackMLState, StateRecord

LOGGER = logging.getLogger(__name__)


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
    def requested_products(self) -> tuple[int, ...]: ...

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
TrialInstructionProvider = Callable[[], TrialInstructionEvidence]


class OperatorTaskPort(Protocol):
    def submit(self, request: TaskRequest) -> TaskReceipt: ...

    def report(self) -> LiveCellReport: ...

    def evaluate_final_build(self, product: int) -> FinalBuildEvaluation: ...


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
            "task_id": self.task_id, "source": self.source.value,
            "reported_at": self.reported_at,
            "packml": dict(self.packml),
            "selected_products": list(self.selected_products), "completed_products": list(self.completed_products),
            "current_step_id": self.current_step_id,
            "event_count": self.event_count, "controller_finished": self.controller_finished,
            "execution_authorized": self.execution_authorized, "quality_release": self.quality_release.value,
            "final_checks": [decision_dict(decision) for decision in self.final_checks],
            "active_fault": self.active_fault,
        }


@dataclass(frozen=True, slots=True)
class FinalBuildEvaluation:
    decision: FinalGateDecision
    report: LiveCellReport


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
        trial_instruction: TrialInstructionProvider,
        final_evidence: FinalBuildEvidenceProvider | None = None,
        final_gate: FinalBuildGate | None = None,
        catalog: Catalog | None = None,
        clock: Callable[[], datetime] | None = None,
        task_id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(source, ControllerSource):
            raise OperatorTaskPortError("source must be a ControllerSource")
        if source is ControllerSource.CELL:
            raise OperatorTaskPortError("cell source requires a reviewed physical act gate; integration unavailable")
        if not callable(controller_factory) or not callable(trial_instruction):
            raise OperatorTaskPortError("controller_factory and trial_instruction must be callable")
        if catalog is not None and type(catalog) is not Catalog:
            raise OperatorTaskPortError("catalog must be a validated Catalog")
        if final_gate is not None and type(final_gate) is not FinalBuildGate:
            raise OperatorTaskPortError("final_gate must be exactly FinalBuildGate")
        self._source = source
        self._factory = controller_factory
        self._trial_instruction = trial_instruction
        self._final_evidence = final_evidence
        self._final_gate = final_gate
        self._catalog = catalog or load_catalog()
        self._clock = clock or _utc_now
        self._task_id_factory = task_id_factory or _new_task_id
        self._lock = RLock()
        self._controller: ControllerView | None = None
        self._active_task_id: str | None = None
        self._accepted_at: datetime | None = None
        self._selected_products: tuple[int, ...] = ()
        self._final_checks: dict[int, FinalGateDecision] = {}
        self._prepared_final: dict[str, FinalBuildRequest] = {}
        self._used_task_ids: set[str] = set()
        self._trial_evidence: TrialInstructionEvidence | None = None

    def submit(self, request: TaskRequest) -> TaskReceipt:
        with self._lock:
            return self._submit(request)

    def _submit(self, request: TaskRequest) -> TaskReceipt:
        if not isinstance(request, TaskRequest):
            raise OperatorTaskPortError("task submission requires a TaskRequest")
        if self._controller is not None and not self._is_quality_released():
            raise OperatorTaskPortError("an active task must be quality-released before another task starts")
        products, trial_evidence = self._resolve_products(request)
        task_id = self._next_task_id()
        try:
            controller = self._factory(products)
            _read_controller(controller, products)
        except OperatorTaskPortError:
            raise
        except Exception as error:
            LOGGER.exception("T09 controller creation failed", extra={"products": products})
            raise OperatorTaskPortError("T09 controller creation failed; cell held") from error
        accepted_at_value = clock_value(self._clock, "task acceptance clock")
        accepted_at = iso(accepted_at_value)
        self._controller = controller
        self._active_task_id = task_id
        self._accepted_at = accepted_at_value
        self._selected_products = products
        self._trial_evidence = trial_evidence
        self._final_checks.clear()
        self._prepared_final.clear()
        self._used_task_ids.add(task_id)
        return TaskReceipt(task_id, self._source, products, accepted_at)

    def report(self) -> LiveCellReport:
        with self._lock:
            return self._report()

    def _report(self) -> LiveCellReport:
        report_time = clock_value(self._clock, "report clock")
        reported_at = iso(report_time)
        if self._controller is None:
            return _idle_report(self._source, reported_at)
        snapshot = _read_controller(self._controller, self._selected_products)
        if datetime.fromisoformat(snapshot.state.timestamp).astimezone(UTC) > report_time:
            raise OperatorTaskPortError("T09 state timestamp is later than the report clock")
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

    def evaluate_final_build(self, product: int) -> FinalBuildEvaluation:
        with self._lock:
            if self._final_gate is None:
                raise OperatorTaskPortError("final-build gate is unavailable; cell held")
            request = self._prepare_final_build(product)
            try:
                try:
                    decision = self._final_gate.evaluate(request)
                except FinalBuildCheckError as error:
                    raise OperatorTaskPortError("final-build gate rejected trusted evidence; cell held") from error
                report = self._record_final_check(decision)
                return FinalBuildEvaluation(decision, report)
            finally:
                self._prepared_final.pop(request.request_id, None)

    def _prepare_final_build(self, product: int) -> FinalBuildRequest:
        validate_product(product)
        if self._controller is None or self._active_task_id is None or self._trial_evidence is None:
            raise OperatorTaskPortError("no active task can provide final-build evidence")
        if product not in self._selected_products:
            raise OperatorTaskPortError("final-build product is outside the active task")
        snapshot = _read_controller(self._controller, self._selected_products)
        if product not in snapshot.completed:
            raise OperatorTaskPortError("a product must be mechanically complete before evidence capture")
        if any(item.product == product for item in self._prepared_final.values()):
            raise OperatorTaskPortError("a final-build check is already prepared for this product")
        mechanical_at = datetime.fromisoformat(snapshot.state.timestamp).astimezone(UTC)
        expected = tuple(step.object_id for step in self._catalog.product(product).steps)
        query = FinalEvidenceQuery(
            self._trial_evidence.run_id,
            self._trial_evidence.rollout,
            self._active_task_id,
            product,
            expected,
            mechanical_at,
        )
        try:
            request = capture_final_build_request(
                self._final_evidence,
                query,
                clock_value(self._clock, "final evidence clock"),
            )
        except TrustedBuildEvidenceError as error:
            raise OperatorTaskPortError(str(error)) from error
        self._require_unused_evidence(request)
        self._prepared_final[request.request_id] = request
        return request

    def _record_final_check(self, decision: FinalGateDecision) -> LiveCellReport:
        if type(decision) is not FinalGateDecision:
            raise OperatorTaskPortError("final check must be exactly FinalGateDecision")
        validate_final_decision(decision)
        prepared = self._prepared_final.get(decision.request_id)
        if prepared is None:
            raise OperatorTaskPortError("final check was not prepared by the trusted evidence port")
        try:
            require_prepared_decision(decision, prepared)
        except TrustedBuildEvidenceError as error:
            raise OperatorTaskPortError(str(error)) from error
        if decision.task_id != self._active_task_id:
            raise OperatorTaskPortError("final check does not belong to the active task")
        if self._controller is None or decision.product not in self._selected_products:
            raise OperatorTaskPortError("final check does not belong to the active task")
        snapshot = _read_controller(self._controller, self._selected_products)
        state_at = datetime.fromisoformat(snapshot.state.timestamp).astimezone(UTC)
        observed_at = decision.observed_at.astimezone(UTC)
        now = clock_value(self._clock, "final-check clock")
        if self._accepted_at is None or observed_at < max(self._accepted_at, state_at) or observed_at > now:
            raise OperatorTaskPortError("final check evidence predates mechanical state or is from the future")
        if now > decision.expires_at:
            raise OperatorTaskPortError("final check evidence is stale")
        if decision.product not in snapshot.completed:
            raise OperatorTaskPortError("a product must be mechanically complete before its final check")
        existing = self._validated_existing_decision(decision)
        self._final_checks.pop(decision.product, None)
        self._final_checks[decision.product] = decision
        try:
            report = self._report()
        except Exception:
            self._final_checks.pop(decision.product, None)
            if existing is not None:
                self._final_checks[decision.product] = existing
            raise
        self._prepared_final.pop(decision.request_id, None)
        return report

    def _validated_existing_decision(self, decision: FinalGateDecision) -> FinalGateDecision | None:
        existing = self._final_checks.get(decision.product)
        if existing is not None and existing.outcome is BuildCheckOutcome.DONE:
            raise OperatorTaskPortError("a quality-released product cannot be overwritten")
        if existing is not None and (
            decision.request_id == existing.request_id or decision.observed_at <= existing.observed_at
        ):
            raise OperatorTaskPortError("a retried final check requires fresh image evidence and request ID")
        return existing

    def _require_unused_evidence(self, request: FinalBuildRequest) -> None:
        prepared = tuple(self._prepared_final.values())
        decisions = tuple(self._final_checks.values())
        if any(item.request_id == request.request_id for item in prepared) or any(
            item.request_id == request.request_id for item in decisions
        ):
            raise OperatorTaskPortError("final-build request ID was already used")
        lineage = (request.image_capture_id, request.image_sha256)
        if any((item.image_capture_id, item.image_sha256) == lineage for item in (*prepared, *decisions)):
            raise OperatorTaskPortError("final-build image evidence was already used")

    def _resolve_products(self, request: TaskRequest) -> tuple[tuple[int, ...], TrialInstructionEvidence]:
        try:
            evidence = self._trial_instruction()
        except Exception as error:
            LOGGER.exception("trial instruction evidence failed")
            raise OperatorTaskPortError("trial instruction evidence failed; cell held") from error
        try:
            trial_products = authorized_trial_products(evidence, self._catalog, clock_value(self._clock, "trial clock"))
        except AuthorityEvidenceError as error:
            raise OperatorTaskPortError(str(error)) from error
        if request.scope is AssemblyScope.EXPLICIT and request.products != trial_products:
            raise OperatorTaskPortError("operator intent does not match the trial instruction")
        return trial_products, evidence

    def _next_task_id(self) -> str:
        try:
            value = self._task_id_factory()
        except Exception as error:
            LOGGER.exception("task ID generation failed")
            raise OperatorTaskPortError("task ID generation failed; cell held") from error
        task_id = safe_id(value)
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
        requested = controller.requested_products
        completed = controller.completed_products
        events = controller.events
        records = controller.plan_records
        finished = controller.finished
    except Exception as error:
        LOGGER.exception("T09 controller status failed")
        raise OperatorTaskPortError("T09 controller status failed; cell held") from error
    if type(state) is not StateRecord:
        raise OperatorTaskPortError("T09 current state must be exactly StateRecord")
    requested = products(requested, "requested products", allow_empty=True, preserve_order=True)
    completed = products(completed, "completed products", allow_empty=True, preserve_order=True)
    if any(product not in selected for product in completed):
        raise OperatorTaskPortError("T09 reported a completed product outside the selected task")
    if not isinstance(events, tuple) or any(type(event) is not ControllerEvent for event in events):
        raise OperatorTaskPortError("T09 events must be a tuple of ControllerEvent")
    if not isinstance(records, tuple) or any(type(record) is not PlanStepRecord for record in records):
        raise OperatorTaskPortError("T09 plan records must be a tuple of PlanStepRecord")
    validate_events(events, selected)
    validate_records(records, selected)
    if type(finished) is not bool:
        raise OperatorTaskPortError("T09 finished must be a boolean")
    if finished and (completed != selected or state.state is not PackMLState.COMPLETE):
        raise OperatorTaskPortError("finished T09 task must report every selected product and Complete state")
    validate_state_record(state, selected)
    pristine = state.state is PackMLState.IDLE and not completed and not events and not records and not finished
    if not requested and not pristine:
        raise OperatorTaskPortError("T09 must bind requested products before leaving pristine Idle")
    if requested and requested != selected:
        raise OperatorTaskPortError("T09 requested products do not match the trial instruction")
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
    fault_states = {PackMLState.HELD, PackMLState.SUSPENDED, PackMLState.STOPPING, PackMLState.STOPPED,
                    PackMLState.ABORTING, PackMLState.ABORTED, PackMLState.RESETTING}
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


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_task_id() -> str:
    return f"task-{uuid4().hex}"
