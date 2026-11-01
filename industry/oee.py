"""Evidence-backed OEE and Andon metrics derived from an append-only run log.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType

from assembly.state_machine import PackMLState
from industry.events import (
    EscapeEvent,
    EventSource,
    HeartbeatEvent,
    PackMLEvent,
    ProductResult,
    ProductResultEvent,
    RunEvent,
)
from industry.lifecycle import LifecycleEvidenceError, validate_terminal_lifecycle


class DashboardMetricError(ValueError):
    """A dashboard metric input is incomplete or malformed."""


class RagStatus(StrEnum):
    RED = "red"
    AMBER = "amber"
    GREEN = "green"


@dataclass(frozen=True, slots=True)
class DashboardSchedule:
    """Measured inputs required for an OEE reporting window."""

    planned_start: datetime
    planned_end: datetime
    ideal_cycle_time_s: Mapping[int, float]

    def __post_init__(self) -> None:
        start = _aware(self.planned_start, "planned_start")
        end = _aware(self.planned_end, "planned_end")
        if end <= start:
            raise DashboardMetricError("planned_end must be later than planned_start")
        normalized = _ideal_times(self.ideal_cycle_time_s)
        object.__setattr__(self, "planned_start", start)
        object.__setattr__(self, "planned_end", end)
        object.__setattr__(self, "ideal_cycle_time_s", MappingProxyType(normalized))


@dataclass(frozen=True, slots=True)
class DashboardMetric:
    value: float | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.value is None and not self.reason:
            raise DashboardMetricError("an unavailable metric requires a reason")
        if self.value is not None and self.reason is not None:
            raise DashboardMetricError("an available metric cannot have an unavailable reason")
        if self.value is not None and not math.isfinite(self.value):
            raise DashboardMetricError("metric values must be finite")


@dataclass(frozen=True, slots=True)
class StateTransition:
    timestamp: str
    state: str
    product: int | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    source: str
    generated_at: str
    last_event_at: str
    stale: bool
    rag_status: str
    packml_state: str
    active_fault: str | None
    availability: DashboardMetric
    performance: DashboardMetric
    quality: DashboardMetric
    oee: DashboardMetric
    first_pass_yield: DashboardMetric
    final_yield: DashboardMetric
    builds_per_assist: DashboardMetric
    average_cycle_time_s: DashboardMetric
    completed_products: int
    good_products: int
    rejected_products: int
    assists: int
    escapes: int
    recent_transitions: tuple[StateTransition, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source, "generated_at": self.generated_at,
            "last_event_at": self.last_event_at, "stale": self.stale,
            "rag_status": self.rag_status, "packml_state": self.packml_state,
            "active_fault": self.active_fault,
            "metrics": _metric_dict(self),
            "counts": _count_dict(self),
            "recent_transitions": [_transition_dict(transition) for transition in self.recent_transitions],
        }


@dataclass(frozen=True, slots=True)
class _SnapshotFacts:
    source: EventSource
    generated_at: datetime
    last_event_at: datetime
    stale: bool
    effective_end: datetime
    states: tuple[PackMLEvent, ...]
    results: tuple[ProductResultEvent, ...]
    escape_count: int
    metrics: tuple[DashboardMetric, ...]
    current: PackMLEvent | None
    counts: tuple[int, int, int, int]


def compute_dashboard_snapshot(
    events: tuple[RunEvent, ...],
    schedule: DashboardSchedule,
    *,
    now: datetime,
    stale_after_s: float,
) -> DashboardSnapshot:
    """Compute OEE without filling missing measured inputs with assumptions."""
    generated_at = _aware(now, "now")
    stale_limit = _positive_number(stale_after_s, "stale_after_s")
    ordered = _validated_events(events)
    facts = _snapshot_facts(ordered, schedule, generated_at, stale_limit)
    return _make_snapshot(facts)


def _snapshot_facts(
    ordered: tuple[RunEvent, ...],
    schedule: DashboardSchedule,
    generated_at: datetime,
    stale_limit: float,
) -> _SnapshotFacts:
    effective_end = min(generated_at, schedule.planned_end)
    in_window = tuple(event for event in ordered if schedule.planned_start <= event.timestamp < effective_end)
    states = tuple(event for event in ordered if isinstance(event, PackMLEvent))
    results = tuple(event for event in in_window if isinstance(event, ProductResultEvent))
    escapes = tuple(event for event in in_window if isinstance(event, EscapeEvent))
    last = ordered[-1].timestamp
    if last > generated_at:
        raise DashboardMetricError("run log contains an event later than the dashboard clock")
    operating = _operating_seconds(states, schedule.planned_start, effective_end)
    return _SnapshotFacts(
        _single_source(ordered),
        generated_at,
        last,
        (generated_at - last).total_seconds() > stale_limit,
        effective_end,
        states,
        results,
        len(escapes),
        _calculate_metrics(results, schedule, operating, effective_end),
        _current_state(states, generated_at),
        _counts(results),
    )


def _make_snapshot(facts: _SnapshotFacts) -> DashboardSnapshot:
    rag_status = RagStatus.RED if facts.stale else _rag_status(facts.current)
    active_fault = _stale_fault(facts.current) if facts.stale else _active_fault(facts.current)
    return DashboardSnapshot(
        source=facts.source.value, generated_at=_iso(facts.generated_at),
        last_event_at=_iso(facts.last_event_at), stale=facts.stale,
        rag_status=rag_status.value, packml_state=facts.current.state.value if facts.current else "Unknown",
        active_fault=active_fault,
        availability=facts.metrics[0],
        performance=facts.metrics[1],
        quality=facts.metrics[2],
        oee=facts.metrics[3],
        first_pass_yield=facts.metrics[4],
        final_yield=facts.metrics[5],
        builds_per_assist=facts.metrics[6],
        average_cycle_time_s=facts.metrics[7],
        completed_products=facts.counts[0],
        good_products=facts.counts[1],
        rejected_products=facts.counts[2],
        assists=facts.counts[3],
        escapes=facts.escape_count,
        recent_transitions=_recent_transitions(facts.states, facts.generated_at),
    )


def _calculate_metrics(
    results: tuple[ProductResultEvent, ...],
    schedule: DashboardSchedule,
    operating_s: float | None,
    effective_end: datetime,
) -> tuple[DashboardMetric, ...]:
    planned_s = max(0.0, (effective_end - schedule.planned_start).total_seconds())
    availability = _ratio(operating_s, planned_s, "PackML history does not cover the window")
    performance = _performance(results, schedule.ideal_cycle_time_s, operating_s)
    completed, good, _, assists = _counts(results)
    quality = _ratio(good, completed, "no completed products in the window")
    oee = _product_metric(availability, performance, quality)
    first_pass = sum(event.result is ProductResult.GOOD_FIRST_PASS for event in results)
    first_pass_yield = _ratio(first_pass, completed, "no completed products in the window")
    final_yield = _ratio(good, completed, "no completed products in the window")
    builds_per_assist = _ratio(good, assists, "no assists recorded in the window")
    average_cycle = _average_cycle_time(results)
    return (
        availability,
        performance,
        quality,
        oee,
        first_pass_yield,
        final_yield,
        builds_per_assist,
        average_cycle,
    )


def _operating_seconds(
    states: tuple[PackMLEvent, ...],
    start: datetime,
    end: datetime,
) -> float | None:
    if end <= start:
        return 0.0
    ordered = tuple(sorted(states, key=lambda event: event.timestamp))
    covering = [event for event in ordered if event.timestamp <= start]
    if not covering:
        return None
    current = covering[-1]
    cursor = start
    operating = 0.0
    for event in (event for event in ordered if start < event.timestamp < end):
        if current.state is PackMLState.EXECUTE:
            operating += (event.timestamp - cursor).total_seconds()
        current, cursor = event, event.timestamp
    if current.state is PackMLState.EXECUTE:
        operating += (end - cursor).total_seconds()
    return operating


def _performance(
    results: tuple[ProductResultEvent, ...],
    ideals: Mapping[int, float],
    operating_s: float | None,
) -> DashboardMetric:
    if operating_s is None:
        return DashboardMetric(None, "PackML history does not cover the window")
    if operating_s <= 0.0:
        return DashboardMetric(None, "no Execute time in the window")
    missing = sorted({event.product for event in results if event.product not in ideals})
    if missing:
        return DashboardMetric(None, f"missing measured ideal cycle time for products {missing}")
    if not results:
        return DashboardMetric(None, "no completed products in the window")
    ideal_seconds = sum(ideals[event.product] for event in results)
    return DashboardMetric(ideal_seconds / operating_s)


def _average_cycle_time(results: tuple[ProductResultEvent, ...]) -> DashboardMetric:
    measured = [event.cycle_time_s for event in results if event.cycle_time_s is not None]
    if len(measured) != len(results):
        missing = len(results) - len(measured)
        return DashboardMetric(None, f"{missing} of {len(results)} completed products lack measured cycle time")
    if not measured:
        return DashboardMetric(None, "no measured cycle times in the window")
    return DashboardMetric(sum(measured) / len(measured))


def _product_metric(*metrics: DashboardMetric) -> DashboardMetric:
    unavailable = [metric.reason for metric in metrics if metric.value is None]
    if unavailable:
        return DashboardMetric(None, "; ".join(reason for reason in unavailable if reason))
    value = math.prod(metric.value for metric in metrics if metric.value is not None)
    return DashboardMetric(value)


def _ratio(numerator: float | None, denominator: float, reason: str) -> DashboardMetric:
    if numerator is None or denominator <= 0:
        return DashboardMetric(None, reason)
    return DashboardMetric(float(numerator) / float(denominator))


def _counts(results: tuple[ProductResultEvent, ...]) -> tuple[int, int, int, int]:
    good_results = {ProductResult.GOOD_FIRST_PASS, ProductResult.GOOD_AFTER_REBUILD}
    good = sum(event.result in good_results for event in results)
    completed = len(results)
    return completed, good, completed - good, sum(event.assists for event in results)


def _current_state(states: tuple[PackMLEvent, ...], at: datetime) -> PackMLEvent | None:
    candidates = [event for event in states if event.timestamp <= at]
    return candidates[-1] if candidates else None


def _rag_status(event: PackMLEvent | None) -> RagStatus:
    if event is None:
        return RagStatus.RED
    red = {
        PackMLState.HELD,
        PackMLState.STOPPING,
        PackMLState.STOPPED,
        PackMLState.ABORTING,
        PackMLState.ABORTED,
    }
    green = {PackMLState.EXECUTE, PackMLState.COMPLETE}
    return RagStatus.RED if event.state in red else RagStatus.GREEN if event.state in green else RagStatus.AMBER


def _active_fault(event: PackMLEvent | None) -> str | None:
    if event is None:
        return "No PackML state available"
    fault_states = {
        PackMLState.HELD,
        PackMLState.SUSPENDED,
        PackMLState.STOPPING,
        PackMLState.STOPPED,
        PackMLState.ABORTING,
        PackMLState.ABORTED,
    }
    if event.state not in fault_states:
        return None
    return event.reason or f"Cell is {event.state.value} without a reason"


def _stale_fault(event: PackMLEvent | None) -> str:
    prior = _active_fault(event)
    return "Run-log data is stale" if prior is None else f"Run-log data is stale; last fault: {prior}"


def _recent_transitions(states: tuple[PackMLEvent, ...], at: datetime) -> tuple[StateTransition, ...]:
    visible = sorted((event for event in states if event.timestamp <= at), key=lambda event: event.timestamp)[-10:]
    return tuple(StateTransition(_iso(event.timestamp), event.state.value, event.product, event.reason) for event in visible)


def _validated_events(events: tuple[RunEvent, ...]) -> tuple[RunEvent, ...]:
    if not isinstance(events, tuple) or not events:
        raise DashboardMetricError("dashboard requires a non-empty tuple of run events")
    if any(not isinstance(event, (PackMLEvent, ProductResultEvent, EscapeEvent, HeartbeatEvent)) for event in events):
        raise DashboardMetricError("run log contains an unsupported event")
    ordered = tuple(sorted(events, key=lambda event: event.timestamp))
    if len({(event.timestamp, type(event).__name__, repr(event)) for event in ordered}) != len(ordered):
        raise DashboardMetricError("run log contains duplicate events")
    terminals = [event for event in ordered if isinstance(event, ProductResultEvent)]
    identities = {(event.run_id, event.rollout, event.product) for event in terminals}
    if len(identities) != len(terminals):
        raise DashboardMetricError("run log contains more than one terminal result for an identity")
    for index, event in enumerate(ordered):
        if isinstance(event, ProductResultEvent):
            try:
                validate_terminal_lifecycle(ordered[:index], event)
            except LifecycleEvidenceError as error:
                raise DashboardMetricError(str(error)) from error
    return ordered


def _single_source(events: tuple[RunEvent, ...]) -> EventSource:
    sources = {event.source for event in events}
    if len(sources) != 1:
        raise DashboardMetricError("one dashboard window cannot mix mock, sim, replay, and cell sources")
    if len({event.run_id for event in events}) != 1:
        raise DashboardMetricError("one dashboard window cannot mix run identities")
    return next(iter(sources))


def _ideal_times(values: Mapping[int, float]) -> dict[int, float]:
    if not isinstance(values, Mapping):
        raise DashboardMetricError("ideal_cycle_time_s must be a mapping")
    normalized: dict[int, float] = {}
    for product, value in values.items():
        _product(product)
        normalized[product] = _positive_number(value, f"ideal cycle time for product {product}")
    return normalized


def _product(value: object, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
        raise DashboardMetricError(f"product must be an integer from 1 through 6, got {value!r}")


def _nonnegative_integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DashboardMetricError(f"{field} must be a nonnegative integer")
    return value


def _positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DashboardMetricError(f"{field} must be a positive finite number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise DashboardMetricError(f"{field} must be a positive finite number")
    return number


def _aware(value: object, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise DashboardMetricError(f"{field} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _metric_dict(snapshot: DashboardSnapshot) -> dict[str, dict[str, object]]:
    names = (
        "availability", "performance", "quality", "oee",
        "first_pass_yield", "final_yield", "builds_per_assist", "average_cycle_time_s",
    )
    return {name: _metric_value(getattr(snapshot, name)) for name in names}


def _metric_value(metric: DashboardMetric) -> dict[str, object]:
    return {"value": metric.value, "unavailable_reason": metric.reason}


def _count_dict(snapshot: DashboardSnapshot) -> dict[str, int]:
    return {
        "completed_products": snapshot.completed_products, "good_products": snapshot.good_products,
        "rejected_products": snapshot.rejected_products, "assists": snapshot.assists, "escapes": snapshot.escapes,
    }


def _transition_dict(transition: StateTransition) -> dict[str, object]:
    return {"timestamp": transition.timestamp, "state": transition.state,
            "product": transition.product, "reason": transition.reason}
