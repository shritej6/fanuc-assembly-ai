from datetime import UTC, datetime, timedelta

import pytest

from assembly.state_machine import PackMLState
from industry.oee import (
    DashboardMetric,
    DashboardMetricError,
    DashboardSchedule,
    EscapeEvent,
    EventSource,
    HeartbeatEvent,
    PackMLEvent,
    ProductResult,
    ProductResultEvent,
    compute_dashboard_snapshot,
)

RUN_ID = "run-2026-10-29"
ROLLOUT = 3


def _time(seconds: int) -> datetime:
    return datetime(2026, 10, 29, 15, tzinfo=UTC) + timedelta(seconds=seconds)


def _state(
    seconds: int,
    state: PackMLState,
    reason: str | None = None,
    *,
    product: int = 1,
    rollout: int = ROLLOUT,
) -> PackMLEvent:
    return PackMLEvent(_time(seconds), EventSource.MOCK, RUN_ID, rollout, state, reason, product)


def _result(
    seconds: int,
    result: ProductResult,
    *,
    product: int = 1,
    rollout: int = ROLLOUT,
    assists: int = 0,
    cycle_time_s: float | None = 30.0,
) -> ProductResultEvent:
    return ProductResultEvent(
        _time(seconds), EventSource.MOCK, RUN_ID, rollout, product, result, assists, cycle_time_s
    )


def _terminal_events(
    seconds: int,
    result: ProductResult,
    **result_fields: object,
) -> tuple[PackMLEvent | ProductResultEvent, ...]:
    terminal = {
        ProductResult.GOOD_FIRST_PASS: PackMLState.COMPLETE,
        ProductResult.GOOD_AFTER_REBUILD: PackMLState.COMPLETE,
        ProductResult.FAILED: PackMLState.ABORTED,
        ProductResult.BLOCKED: PackMLState.HELD,
    }[result]
    event = _result(seconds, result, **result_fields)  # type: ignore[arg-type]
    reason = "operator assistance active" if terminal is PackMLState.HELD else None
    state = _state(seconds, terminal, reason, product=event.product, rollout=event.rollout)
    resumed = _state(seconds, PackMLState.EXECUTE, product=event.product, rollout=event.rollout)
    return state, event, resumed


def _schedule(**ideals: float) -> DashboardSchedule:
    return DashboardSchedule(_time(0), _time(100), {int(product): value for product, value in ideals.items()})


def test_computes_evidence_backed_oee_and_andon_snapshot() -> None:
    events = (
        _state(0, PackMLState.EXECUTE),
        *_terminal_events(30, ProductResult.GOOD_FIRST_PASS, assists=2, cycle_time_s=25.0),
        _state(60, PackMLState.HELD, "restock M1_03"),
        _state(80, PackMLState.EXECUTE),
        *_terminal_events(85, ProductResult.FAILED, rollout=4, cycle_time_s=35.0),
        EscapeEvent(_time(90), EventSource.MOCK, RUN_ID, ROLLOUT, 1, "fault injection was not caught"),
    )

    snapshot = compute_dashboard_snapshot(events, _schedule(**{"1": 20.0}), now=_time(100), stale_after_s=20)

    assert snapshot.availability.value == pytest.approx(0.8)
    assert snapshot.performance.value == pytest.approx(0.5)
    assert snapshot.quality.value == pytest.approx(0.5)
    assert snapshot.oee.value == pytest.approx(0.2)
    assert snapshot.first_pass_yield.value == pytest.approx(0.5)
    assert snapshot.final_yield.value == pytest.approx(0.5)
    assert snapshot.builds_per_assist.value == pytest.approx(0.5)
    assert snapshot.average_cycle_time_s.value == pytest.approx(30.0)
    assert snapshot.completed_products == 2
    assert snapshot.good_products == 1
    assert snapshot.rejected_products == 1
    assert snapshot.assists == 2
    assert snapshot.escapes == 1
    assert snapshot.packml_state == "Execute"
    assert snapshot.rag_status == "green"
    assert snapshot.active_fault is None
    assert snapshot.stale is False


def test_held_state_is_red_and_exposes_reason() -> None:
    events = (_state(0, PackMLState.EXECUTE), _state(20, PackMLState.HELD, "operator assist required"))

    snapshot = compute_dashboard_snapshot(events, _schedule(), now=_time(25), stale_after_s=10)

    assert snapshot.rag_status == "red"
    assert snapshot.packml_state == "Held"
    assert snapshot.active_fault == "operator assist required"


def test_same_timestamp_uses_last_packml_transition() -> None:
    events = (
        _state(0, PackMLState.EXECUTE),
        _state(20, PackMLState.EXECUTE),
        _state(20, PackMLState.HELD, "operator assist"),
    )

    snapshot = compute_dashboard_snapshot(events, _schedule(), now=_time(21), stale_after_s=10)

    assert snapshot.packml_state == "Held"
    assert snapshot.rag_status == "red"
    assert snapshot.active_fault == "operator assist"


def test_suspended_state_surfaces_spare_reason() -> None:
    events = (_state(0, PackMLState.EXECUTE), _state(20, PackMLState.SUSPENDED, "waiting for M2_03 spare"))

    snapshot = compute_dashboard_snapshot(events, _schedule(), now=_time(21), stale_after_s=10)

    assert snapshot.rag_status == "amber"
    assert snapshot.active_fault == "waiting for M2_03 spare"


def test_missing_measurements_stay_unavailable() -> None:
    events = (_state(0, PackMLState.EXECUTE), *_terminal_events(50, ProductResult.GOOD_FIRST_PASS, cycle_time_s=None))

    snapshot = compute_dashboard_snapshot(events, _schedule(), now=_time(100), stale_after_s=60)

    assert snapshot.performance.value is None
    assert snapshot.performance.reason == "missing measured ideal cycle time for products [1]"
    assert snapshot.oee.value is None
    assert "missing measured ideal cycle" in (snapshot.oee.reason or "")
    assert snapshot.builds_per_assist.value is None
    assert snapshot.average_cycle_time_s.value is None


def test_average_cycle_time_requires_complete_measurement_coverage() -> None:
    events = (
        _state(0, PackMLState.EXECUTE),
        *_terminal_events(20, ProductResult.GOOD_FIRST_PASS, cycle_time_s=10.0),
        *_terminal_events(30, ProductResult.GOOD_FIRST_PASS, rollout=4, cycle_time_s=None),
    )

    snapshot = compute_dashboard_snapshot(events, _schedule(**{"1": 10.0}), now=_time(40), stale_after_s=20)

    assert snapshot.average_cycle_time_s.value is None
    assert snapshot.average_cycle_time_s.reason == "1 of 2 completed products lack measured cycle time"


def test_adjacent_windows_do_not_double_count_boundary_result() -> None:
    boundary_result = _result(50, ProductResult.GOOD_FIRST_PASS)
    lifecycle = _state(50, PackMLState.COMPLETE)
    events = (_state(0, PackMLState.EXECUTE), lifecycle, boundary_result)
    first = DashboardSchedule(_time(0), _time(50), {1: 20.0})
    second = DashboardSchedule(_time(50), _time(100), {1: 20.0})

    left = compute_dashboard_snapshot(events, first, now=_time(100), stale_after_s=100)
    right = compute_dashboard_snapshot(events, second, now=_time(100), stale_after_s=100)

    assert left.completed_products == 0
    assert right.completed_products == 1


def test_live_status_keeps_advancing_after_reporting_window_closes() -> None:
    schedule = DashboardSchedule(_time(0), _time(50), {})
    events = (_state(0, PackMLState.EXECUTE), _state(60, PackMLState.HELD, "cell interlock"))

    snapshot = compute_dashboard_snapshot(events, schedule, now=_time(70), stale_after_s=20)

    assert snapshot.packml_state == "Held"
    assert snapshot.rag_status == "red"
    assert snapshot.active_fault == "cell interlock"
    assert snapshot.recent_transitions[-1].state == "Held"


def test_packml_history_must_cover_planned_window() -> None:
    events = (_state(10, PackMLState.EXECUTE), *_terminal_events(20, ProductResult.GOOD_FIRST_PASS))

    snapshot = compute_dashboard_snapshot(events, _schedule(**{"1": 20.0}), now=_time(50), stale_after_s=60)

    assert snapshot.availability.value is None
    assert snapshot.availability.reason == "PackML history does not cover the window"
    assert snapshot.performance.value is None


def test_mixed_sources_are_rejected() -> None:
    cell_state = PackMLEvent(_time(1), EventSource.CELL, RUN_ID, ROLLOUT, PackMLState.EXECUTE, None, 1)

    with pytest.raises(DashboardMetricError, match="cannot mix"):
        compute_dashboard_snapshot((_state(0, PackMLState.EXECUTE), cell_state), _schedule(), now=_time(2), stale_after_s=1)


def test_mixed_run_identities_are_rejected() -> None:
    other_run = PackMLEvent(
        _time(1), EventSource.MOCK, "other-run", ROLLOUT, PackMLState.EXECUTE, None, 1
    )

    with pytest.raises(DashboardMetricError, match="cannot mix run identities"):
        compute_dashboard_snapshot((_state(0, PackMLState.EXECUTE), other_run), _schedule(), now=_time(2), stale_after_s=1)


def test_future_log_event_is_rejected() -> None:
    with pytest.raises(DashboardMetricError, match="later than the dashboard clock"):
        compute_dashboard_snapshot((_state(10, PackMLState.EXECUTE),), _schedule(), now=_time(9), stale_after_s=1)


def test_duplicate_events_are_rejected() -> None:
    event = _state(0, PackMLState.EXECUTE)

    with pytest.raises(DashboardMetricError, match="duplicate"):
        compute_dashboard_snapshot((event, event), _schedule(), now=_time(1), stale_after_s=1)


def test_duplicate_terminal_identity_is_rejected_even_when_timestamp_changes() -> None:
    results = (
        _result(10, ProductResult.GOOD_FIRST_PASS),
        _result(20, ProductResult.GOOD_AFTER_REBUILD),
    )

    with pytest.raises(DashboardMetricError, match="terminal result"):
        compute_dashboard_snapshot(results, _schedule(), now=_time(30), stale_after_s=20)


@pytest.mark.parametrize(
    "events",
    [
        (_result(10, ProductResult.GOOD_FIRST_PASS),),
        (_result(10, ProductResult.GOOD_FIRST_PASS), _state(20, PackMLState.COMPLETE)),
        (_state(5, PackMLState.COMPLETE, rollout=4), _result(10, ProductResult.GOOD_FIRST_PASS)),
        (
            _state(5, PackMLState.COMPLETE),
            _state(7, PackMLState.ABORTED),
            _result(10, ProductResult.GOOD_FIRST_PASS),
        ),
        (_state(5, PackMLState.STOPPED), _result(10, ProductResult.FAILED)),
        (_state(5, PackMLState.EXECUTE), _result(10, ProductResult.BLOCKED)),
    ],
)
def test_terminal_results_require_matching_earlier_active_lifecycle(
    events: tuple[PackMLEvent | ProductResultEvent, ...],
) -> None:
    with pytest.raises(DashboardMetricError, match="lifecycle"):
        compute_dashboard_snapshot(events, _schedule(), now=_time(30), stale_after_s=20)


def test_blocked_terminal_accepts_active_held_andon_lifecycle() -> None:
    events = (
        _state(0, PackMLState.EXECUTE),
        _state(10, PackMLState.HELD, "operator assistance active"),
        _result(10, ProductResult.BLOCKED),
    )

    snapshot = compute_dashboard_snapshot(events, _schedule(), now=_time(11), stale_after_s=20)

    assert snapshot.packml_state == "Held"
    assert snapshot.rejected_products == 1


def test_stale_source_forces_fail_closed_red_status() -> None:
    snapshot = compute_dashboard_snapshot(
        (_state(0, PackMLState.EXECUTE),),
        _schedule(),
        now=_time(100),
        stale_after_s=10,
    )

    assert snapshot.stale is True
    assert snapshot.rag_status == "red"
    assert snapshot.active_fault == "Run-log data is stale"


def test_periodic_heartbeat_keeps_a_long_healthy_execute_live() -> None:
    events = (_state(0, PackMLState.EXECUTE), HeartbeatEvent(_time(55), EventSource.MOCK, RUN_ID, ROLLOUT))

    snapshot = compute_dashboard_snapshot(events, _schedule(), now=_time(60), stale_after_s=10)

    assert snapshot.stale is False
    assert snapshot.rag_status == "green"
    assert snapshot.packml_state == "Execute"


def test_stale_source_preserves_last_recorded_fault_context() -> None:
    snapshot = compute_dashboard_snapshot(
        (_state(0, PackMLState.HELD, "missing spare"),),
        _schedule(),
        now=_time(100),
        stale_after_s=10,
    )

    assert snapshot.active_fault == "Run-log data is stale; last fault: missing spare"


@pytest.mark.parametrize(
    ("start", "end", "message"),
    [(_time(1), _time(1), "later"), (_time(2), _time(1), "later")],
)
def test_schedule_requires_forward_window(start: datetime, end: datetime, message: str) -> None:
    with pytest.raises(DashboardMetricError, match=message):
        DashboardSchedule(start, end, {})


def test_metric_requires_exactly_one_of_value_or_reason() -> None:
    with pytest.raises(DashboardMetricError, match="requires a reason"):
        DashboardMetric(None)
    with pytest.raises(DashboardMetricError, match="cannot have"):
        DashboardMetric(1.0, "not allowed")


def test_snapshot_dictionary_is_json_ready() -> None:
    snapshot = compute_dashboard_snapshot(
        (_state(0, PackMLState.EXECUTE),),
        _schedule(),
        now=_time(1),
        stale_after_s=10,
    )

    payload = snapshot.as_dict()

    assert payload["source"] == "mock"
    assert payload["recent_transitions"] == [
        {"timestamp": "2026-10-29T15:00:00Z", "state": "Execute", "product": 1, "reason": None}
    ]
