"""Behavioral tests for deterministic PLC telemetry and command validation."""

import json
from datetime import UTC, datetime, timedelta
from threading import Event, Lock, Thread

import pytest

from assembly.catalog import load_catalog
from assembly.controller_ports import ControllerEvent
from assembly.inventory import Inventory
from assembly.planner import ContractPoseUF0, PlanStepRecord
from assembly.state_machine import PackMLState, PackMLStateMachine
from robot.opcua_model import (
    CommandQueueFullError,
    CommandStateError,
    CommandValidationError,
    ControllerSnapshotMailbox,
    OpcUaModelError,
    SafeCommandBoundary,
    build_controller_snapshot,
)


class _ControllerSource:
    def __init__(self) -> None:
        self.state_machine = PackMLStateMachine()
        self.inventory = Inventory(load_catalog())
        self._completed_products: tuple[int, ...] = ()
        self._events: tuple[ControllerEvent, ...] = ()
        self._plan_records: tuple[PlanStepRecord, ...] = ()

    @property
    def completed_products(self) -> tuple[int, ...]:
        return self._completed_products

    @property
    def events(self) -> tuple[ControllerEvent, ...]:
        return self._events

    @property
    def plan_records(self) -> tuple[PlanStepRecord, ...]:
        return self._plan_records


def _executing_source() -> _ControllerSource:
    source = _ControllerSource()
    source.state_machine.start(4)
    source.state_machine.transition(PackMLState.EXECUTE)
    source.inventory.add("table-M4_01", "M4_01")
    source.inventory.add("spare-M4_02", "M4_02", zone="spare")
    source.inventory.add("bad-M4_03", "M4_03")
    source.inventory.reject("bad-M4_03")
    source._completed_products = (1, 2)
    source._events = (
        ControllerEvent("andon", 4, None, "operator check"),
        ControllerEvent("retry", 4, "M4_01", "attempt 2"),
        ControllerEvent("andon", 4, None, "same active episode"),
        ControllerEvent("resume", 4, None, "operator cleared fault"),
        ControllerEvent("andon", 4, None, "spare needed"),
    )
    pose = ContractPoseUF0(200.0, 0.0, -300.0, 90.0)
    source._plan_records = (
        PlanStepRecord("product-4-step-01-attempt-1", 4, "table-M4_01", "M4_01", pose, pose, 0.0, 1),
    )
    return source


def test_snapshot_exposes_packml_counts_inventory_and_current_step() -> None:
    from assembly.inventory import InventoryZone

    snapshot = build_controller_snapshot(_executing_source())

    assert snapshot.state is PackMLState.EXECUTE
    assert snapshot.current_product == 4
    assert snapshot.current_step == "product-4-step-01-attempt-1"
    assert snapshot.alarm_active is False
    assert snapshot.active_fault == ""
    assert snapshot.good_count == 2
    assert snapshot.reject_count == 1
    assert snapshot.assist_count == 2
    assert snapshot.inventory_count(InventoryZone.TABLE) == 1
    assert snapshot.inventory_count(InventoryZone.SPARE) == 1
    assert snapshot.inventory_count(InventoryZone.REJECT) == 1


def test_snapshot_mailbox_changes_only_when_the_controller_owner_updates_it() -> None:
    source = _executing_source()
    mailbox = ControllerSnapshotMailbox.from_source(source)

    source.state_machine.hold("camera evidence disagrees")

    assert mailbox.read().state is PackMLState.EXECUTE
    mailbox.update_from(source)
    assert mailbox.read().state is PackMLState.HELD


def test_snapshot_mailbox_rejects_updates_from_a_non_owner_thread() -> None:
    source = _executing_source()
    mailbox = ControllerSnapshotMailbox.from_source(source)
    source.state_machine.hold("camera evidence disagrees")
    errors: list[Exception] = []

    def update() -> None:
        try:
            mailbox.update_from(source)
        except OpcUaModelError as error:
            errors.append(error)

    worker = Thread(target=update)
    worker.start()
    worker.join(timeout=2.0)

    assert len(errors) == 1
    assert isinstance(errors[0], OpcUaModelError)
    assert "controller-owning thread" in str(errors[0])
    assert mailbox.read().state is PackMLState.EXECUTE


def test_snapshot_inventory_json_is_stable_and_contains_no_motion_commands() -> None:
    snapshot = build_controller_snapshot(_executing_source())
    records = json.loads(snapshot.inventory_json)

    assert [record["instance_id"] for record in records] == ["bad-M4_03", "spare-M4_02", "table-M4_01"]
    assert records[0] == {
        "condition": "defective",
        "instance_id": "bad-M4_03",
        "object_id": "M4_03",
        "reserved_for": None,
        "zone": "reject",
    }
    assert "tcp" not in snapshot.inventory_json.lower()
    assert "joint" not in snapshot.inventory_json.lower()


def test_alarm_uses_only_the_live_packml_fault_reason() -> None:
    source = _executing_source()
    source.state_machine.hold("camera evidence disagrees")

    snapshot = build_controller_snapshot(source)

    assert snapshot.alarm_active is True
    assert snapshot.active_fault == "camera evidence disagrees"
    assert snapshot.current_step == "product-4-step-01-attempt-1"


def test_assist_count_tracks_andon_episodes_instead_of_duplicate_events() -> None:
    source = _executing_source()
    source._events = (
        ControllerEvent("andon", 4, None, "camera disagreement"),
        ControllerEvent("retry", 4, "M4_01", "attempt 2"),
        ControllerEvent("andon", 4, None, "same active episode"),
        ControllerEvent("resume", 4, None, "operator cleared fault"),
        ControllerEvent("andon", 4, None, "spare needed"),
        ControllerEvent("andon", 4, None, "same spare episode"),
        ControllerEvent("complete", 4, None, "finished"),
        ControllerEvent("andon", 5, None, "next product check"),
        ControllerEvent("abort", 5, None, "cycle aborted"),
    )

    assert build_controller_snapshot(source).assist_count == 3


def test_command_boundary_accepts_only_a_named_state_valid_intent() -> None:
    now = datetime(2026, 10, 29, 16, 0, tzinfo=UTC)
    boundary = SafeCommandBoundary(clock=lambda: now)

    command = boundary.submit("Hold", "hmi-42", "inspect camera disagreement", state=PackMLState.EXECUTE)

    assert command.name.value == "Hold"
    assert command.command_id == "hmi-42"
    assert command.reason == "inspect camera disagreement"
    assert command.submitted_at == "2026-10-29T16:00:00Z"
    assert not hasattr(command, "tcp")
    assert not hasattr(command, "joints")
    assert boundary.pending == (command,)


@pytest.mark.parametrize(
    ("name", "state"),
    [
        ("Hold", PackMLState.IDLE),
        ("Resume", PackMLState.EXECUTE),
        ("Stop", PackMLState.COMPLETE),
        ("Reset", PackMLState.EXECUTE),
        ("Abort", PackMLState.IDLE),
    ],
)
def test_command_boundary_rejects_invalid_packml_transitions(name: str, state: PackMLState) -> None:
    boundary = SafeCommandBoundary()

    with pytest.raises(CommandStateError, match=f"{name} is not allowed"):
        boundary.submit(name, f"cmd-{name}", "operator request", state=state)

    assert boundary.pending == ()


@pytest.mark.parametrize(
    ("name", "command_id", "reason"),
    [
        ("MoveJoints", "cmd-1", "operator request"),
        ("Hold", "has spaces", "operator request"),
        ("Hold", "cmd-1", ""),
        ("Hold", "cmd-1", "line\nbreak"),
        ("Hold", "x" * 65, "operator request"),
    ],
)
def test_command_boundary_rejects_malformed_or_motion_like_input(
    name: str,
    command_id: str,
    reason: str,
) -> None:
    boundary = SafeCommandBoundary()

    with pytest.raises(CommandValidationError):
        boundary.submit(name, command_id, reason, state=PackMLState.EXECUTE)

    assert boundary.pending == ()


def test_command_ids_are_replay_protected_and_queue_is_bounded() -> None:
    boundary = SafeCommandBoundary(max_pending=1, replay_history=2)
    boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)

    with pytest.raises(CommandQueueFullError):
        boundary.submit("Stop", "cmd-2", "operator request", state=PackMLState.EXECUTE)
    boundary.pop(state=PackMLState.EXECUTE)
    with pytest.raises(CommandValidationError, match="already submitted"):
        boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)


def test_stale_commands_expire_without_releasing_replay_protection() -> None:
    now = datetime(2026, 10, 29, 16, 0, tzinfo=UTC)
    current = [now]
    monotonic_time = [100.0]
    boundary = SafeCommandBoundary(
        max_pending=1,
        replay_history=2,
        pending_ttl_s=5.0,
        clock=lambda: current[0],
        monotonic_clock=lambda: monotonic_time[0],
    )
    boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)

    current[0] = now + timedelta(seconds=6)
    monotonic_time[0] = 106.0

    assert boundary.pending == ()
    assert boundary.pop(state=PackMLState.EXECUTE) is None
    boundary.submit("Stop", "cmd-2", "operator request", state=PackMLState.EXECUTE)
    with pytest.raises(CommandValidationError, match="already submitted"):
        boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)


def test_wall_clock_rollback_does_not_extend_monotonic_command_ttl() -> None:
    submitted_at = datetime(2026, 10, 29, 16, 0, tzinfo=UTC)
    wall_time = [submitted_at]
    monotonic_time = [100.0]
    boundary = SafeCommandBoundary(
        pending_ttl_s=5.0,
        clock=lambda: wall_time[0],
        monotonic_clock=lambda: monotonic_time[0],
    )
    command = boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)

    wall_time[0] = submitted_at - timedelta(days=365)
    monotonic_time[0] = 106.0

    assert command.submitted_at == "2026-10-29T16:00:00Z"
    assert boundary.pending == ()


@pytest.mark.parametrize("monotonic_value", [True, float("inf"), float("nan"), "100"])
def test_command_deadline_clock_must_return_finite_seconds(monotonic_value: object) -> None:
    boundary = SafeCommandBoundary(monotonic_clock=lambda: monotonic_value)  # type: ignore[arg-type,return-value]

    with pytest.raises(CommandValidationError, match="monotonic"):
        boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)

    assert boundary.pending == ()


@pytest.mark.parametrize("pending_ttl_s", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_command_expiry_must_be_a_finite_positive_duration(pending_ttl_s: object) -> None:
    with pytest.raises(CommandValidationError, match="pending_ttl_s"):
        SafeCommandBoundary(pending_ttl_s=pending_ttl_s)  # type: ignore[arg-type]


def test_command_capacity_is_atomic_across_submitter_and_consumer_threads() -> None:
    now = datetime(2026, 10, 29, 16, 0, tzinfo=UTC)
    first_after_capacity = Event()
    release_first = Event()
    competing_submit = Event()
    call_lock = Lock()
    calls = 0

    def clock() -> datetime:
        nonlocal calls
        with call_lock:
            calls += 1
            current_call = calls
        if current_call == 1:
            first_after_capacity.set()
            assert release_first.wait(timeout=2.0)
        elif current_call == 2:
            competing_submit.set()
        return now

    boundary = SafeCommandBoundary(max_pending=1, replay_history=2, clock=clock)
    queued: list[str] = []
    rejected: list[str] = []

    def submit(command_id: str) -> None:
        try:
            boundary.submit("Hold", command_id, "operator request", state=PackMLState.EXECUTE)
            queued.append(command_id)
        except CommandQueueFullError:
            rejected.append(command_id)

    first = Thread(target=submit, args=("cmd-1",))
    second = Thread(target=submit, args=("cmd-2",))
    first.start()
    assert first_after_capacity.wait(timeout=2.0)
    second.start()
    competing_submit.wait(timeout=0.1)
    release_first.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert len(boundary.pending) == 1
    assert len(queued) == 1
    assert len(rejected) == 1


def test_command_is_revalidated_and_dropped_if_state_changes_before_consumption() -> None:
    boundary = SafeCommandBoundary()
    boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)

    with pytest.raises(CommandStateError):
        boundary.pop(state=PackMLState.HELD)

    assert boundary.pending == ()


def test_naive_command_clock_fails_without_enqueuing() -> None:
    boundary = SafeCommandBoundary(clock=lambda: datetime(2026, 10, 29, 16, 0))

    with pytest.raises(CommandValidationError, match="timezone-aware"):
        boundary.submit("Hold", "cmd-1", "operator request", state=PackMLState.EXECUTE)

    assert boundary.pending == ()


@pytest.mark.parametrize(
    ("max_pending", "replay_history"),
    [
        (True, 4096),
        (1.5, 4096),
        (1, False),
        (1, 1.5),
        (0, 4096),
        (65_536, 65_536),
        (2, 1),
    ],
)
def test_command_capacity_must_fit_exact_uint16_pending_tag(
    max_pending: object,
    replay_history: object,
) -> None:
    with pytest.raises(CommandValidationError, match="UInt16"):
        SafeCommandBoundary(max_pending=max_pending, replay_history=replay_history)  # type: ignore[arg-type]
