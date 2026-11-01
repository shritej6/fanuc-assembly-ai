"""Deterministic PLC telemetry and a fail-closed PackML command boundary."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter, OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from numbers import Number
from threading import RLock, get_ident
from time import monotonic
from typing import Protocol

from assembly.controller_ports import ControllerEvent
from assembly.inventory import Inventory, InventoryItem, InventoryZone
from assembly.metrics import count_assist_episodes
from assembly.planner import PlanStepRecord
from assembly.state_machine import PackMLState, PackMLStateMachine


class OpcUaModelError(ValueError):
    """Base class for invalid telemetry or remote command input."""


class CommandValidationError(OpcUaModelError):
    """A command is malformed, replayed, or unsafe for the live state."""


class CommandQueueFullError(CommandValidationError):
    """The bounded command inbox has no remaining capacity."""


class CommandStateError(CommandValidationError):
    """A command is not valid for the current PackML state."""


class ControllerStatusSource(Protocol):
    """Public T09 interfaces consumed by the PLC telemetry adapter."""

    state_machine: PackMLStateMachine
    inventory: Inventory

    @property
    def completed_products(self) -> tuple[int, ...]: ...

    @property
    def events(self) -> tuple[ControllerEvent, ...]: ...

    @property
    def plan_records(self) -> tuple[PlanStepRecord, ...]: ...


_ALARM_STATES = frozenset(
    {
        PackMLState.HELD,
        PackMLState.SUSPENDED,
        PackMLState.STOPPING,
        PackMLState.STOPPED,
        PackMLState.ABORTING,
        PackMLState.ABORTED,
    }
)


@dataclass(frozen=True, slots=True)
class ControllerSnapshot:
    """One coherent, PLC-shaped view of controller-owned runtime state."""

    state: PackMLState
    state_timestamp: str
    current_product: int | None
    current_step: str
    alarm_active: bool
    active_fault: str
    good_count: int
    reject_count: int
    assist_count: int
    inventory_counts: tuple[tuple[InventoryZone, int], ...]
    inventory_json: str

    def inventory_count(self, zone: InventoryZone) -> int:
        return dict(self.inventory_counts)[zone]


class ControllerSnapshotMailbox:
    """Transfer immutable snapshots from the controller owner to OPC UA."""

    def __init__(self, snapshot: ControllerSnapshot) -> None:
        if type(snapshot) is not ControllerSnapshot:
            raise OpcUaModelError("snapshot mailbox requires a ControllerSnapshot")
        self._snapshot = snapshot
        self._lock = RLock()
        self._owner_thread_id = get_ident()

    @classmethod
    def from_source(cls, source: ControllerStatusSource) -> ControllerSnapshotMailbox:
        return cls(build_controller_snapshot(source))

    def update_from(self, source: ControllerStatusSource) -> ControllerSnapshot:
        if get_ident() != self._owner_thread_id:
            raise OpcUaModelError("snapshot mailbox updates require the controller-owning thread")
        snapshot = build_controller_snapshot(source)
        with self._lock:
            self._snapshot = snapshot
        return snapshot

    def read(self) -> ControllerSnapshot:
        with self._lock:
            return self._snapshot


def build_controller_snapshot(source: ControllerStatusSource) -> ControllerSnapshot:
    """Read only public T09 state; never invoke policy, adapters, or motion."""
    record = source.state_machine.current_record
    items = tuple(source.inventory.items())
    counts = Counter(item.zone for item in items)
    inventory_counts = tuple((zone, counts[zone]) for zone in InventoryZone)
    alarm_active = record.state in _ALARM_STATES
    return ControllerSnapshot(
        state=record.state,
        state_timestamp=record.timestamp,
        current_product=record.product,
        current_step=_current_step(source, record.product),
        alarm_active=alarm_active,
        active_fault=record.reason if alarm_active and record.reason is not None else "",
        good_count=len(source.completed_products),
        reject_count=counts[InventoryZone.REJECT],
        assist_count=count_assist_episodes(source.events),
        inventory_counts=inventory_counts,
        inventory_json=_inventory_json(items),
    )


def _current_step(source: ControllerStatusSource, product: int | None) -> str:
    if product is None or source.state_machine.state in {PackMLState.IDLE, PackMLState.COMPLETE}:
        return ""
    for record in reversed(source.plan_records):
        if record.product == product:
            return record.step_id
    return ""


def _inventory_json(items: tuple[InventoryItem, ...]) -> str:
    records = []
    for item in sorted(items, key=lambda value: value.instance_id):
        records.append(
            {
                "condition": item.condition.value,
                "instance_id": item.instance_id,
                "object_id": item.object_id,
                "reserved_for": item.reserved_for,
                "zone": item.zone.value,
            }
        )
    return json.dumps(records, separators=(",", ":"), sort_keys=True)


class CommandName(StrEnum):
    """Non-motion controller requests that may be queued from a PLC or HMI."""

    HOLD = "Hold"
    RESUME = "Resume"
    STOP = "Stop"
    ABORT = "Abort"
    RESET = "Reset"


_COMMAND_STATES = {
    CommandName.HOLD: frozenset({PackMLState.EXECUTE}),
    CommandName.RESUME: frozenset({PackMLState.HELD, PackMLState.SUSPENDED}),
    CommandName.STOP: frozenset(
        {
            PackMLState.STARTING,
            PackMLState.EXECUTE,
            PackMLState.HELD,
            PackMLState.SUSPENDED,
            PackMLState.COMPLETING,
        }
    ),
    CommandName.ABORT: frozenset(
        {
            PackMLState.STARTING,
            PackMLState.EXECUTE,
            PackMLState.HELD,
            PackMLState.SUSPENDED,
            PackMLState.COMPLETING,
            PackMLState.COMPLETE,
            PackMLState.STOPPING,
            PackMLState.STOPPED,
            PackMLState.RESETTING,
        }
    ),
    CommandName.RESET: frozenset({PackMLState.COMPLETE, PackMLState.STOPPED, PackMLState.ABORTED}),
}
_COMMAND_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")


@dataclass(frozen=True, slots=True)
class ControllerCommand:
    """A validated intent; deliberately contains no joints or TCP target."""

    name: CommandName
    command_id: str
    reason: str
    submitted_at: str


def _utc_now() -> datetime:
    return datetime.now(UTC)


class SafeCommandBoundary:
    """Bound, deduplicate, and state-check remote controller requests."""

    def __init__(
        self,
        *,
        max_pending: int = 32,
        replay_history: int = 4096,
        pending_ttl_s: float = 30.0,
        clock: Callable[[], datetime] | None = None,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        valid_capacities = type(max_pending) is int and type(replay_history) is int
        if not valid_capacities or not 1 <= max_pending <= min(replay_history, 65_535):
            raise CommandValidationError(
                "max_pending must be a positive UInt16-sized integer no larger than replay_history"
            )
        if (
            isinstance(pending_ttl_s, bool)
            or not isinstance(pending_ttl_s, Number)
            or not math.isfinite(float(pending_ttl_s))
            or float(pending_ttl_s) <= 0.0
        ):
            raise CommandValidationError("pending_ttl_s must be a finite positive duration")
        self._max_pending = max_pending
        self._replay_history = replay_history
        self._pending_ttl_s = float(pending_ttl_s)
        self._clock = clock or _utc_now
        self._monotonic_clock = monotonic_clock or monotonic
        self._pending: deque[ControllerCommand] = deque()
        self._deadlines: dict[str, float] = {}
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = RLock()

    @property
    def pending(self) -> tuple[ControllerCommand, ...]:
        with self._lock:
            self._expire_stale()
            return tuple(self._pending)

    def submit(
        self,
        name: CommandName | str,
        command_id: str,
        reason: str,
        *,
        state: PackMLState,
    ) -> ControllerCommand:
        normalized = _command_name(name)
        normalized_id = _command_id(command_id)
        normalized_reason = _reason(reason)
        _require_allowed(normalized, state)
        with self._lock:
            if normalized_id in self._seen:
                raise CommandValidationError(f"command_id {normalized_id!r} was already submitted")
            self._expire_stale()
            if len(self._pending) >= self._max_pending:
                raise CommandQueueFullError("command queue is full")
            wall_time = _aware_utc(self._clock())
            deadline_clock = _monotonic_seconds(self._monotonic_clock())
            command = ControllerCommand(normalized, normalized_id, normalized_reason, _timestamp(wall_time))
            self._pending.append(command)
            self._deadlines[normalized_id] = deadline_clock + self._pending_ttl_s
            self._remember(normalized_id)
            return command

    def pop(self, *, state: PackMLState) -> ControllerCommand | None:
        """Return the oldest request only if it is still valid at consumption time."""
        with self._lock:
            self._expire_stale()
            if not self._pending:
                return None
            command = self._pending.popleft()
            self._deadlines.pop(command.command_id, None)
            _require_allowed(command.name, state)
            return command

    def _expire_stale(self) -> None:
        if not self._pending:
            return
        now = _monotonic_seconds(self._monotonic_clock())
        retained: deque[ControllerCommand] = deque()
        for command in self._pending:
            if self._deadlines[command.command_id] > now:
                retained.append(command)
            else:
                self._deadlines.pop(command.command_id, None)
        self._pending = retained

    def _remember(self, command_id: str) -> None:
        self._seen[command_id] = None
        while len(self._seen) > self._replay_history:
            self._seen.popitem(last=False)


def _command_name(value: object) -> CommandName:
    try:
        return CommandName(value)
    except (TypeError, ValueError) as error:
        raise CommandValidationError(f"unsupported controller command {value!r}") from error


def _command_id(value: object) -> str:
    if not isinstance(value, str) or _COMMAND_ID.fullmatch(value) is None:
        raise CommandValidationError("command_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,63}")
    return value


def _reason(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 240:
        raise CommandValidationError("reason must contain 1 through 240 visible characters")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise CommandValidationError("reason must not contain control characters")
    return value.strip()


def _require_allowed(name: CommandName, state: PackMLState) -> None:
    if not isinstance(state, PackMLState) or state not in _COMMAND_STATES[name]:
        state_name = state.value if isinstance(state, PackMLState) else repr(state)
        raise CommandStateError(f"{name.value} is not allowed while PackML is {state_name}")


def _timestamp(value: object) -> str:
    return _aware_utc(value).isoformat().replace("+00:00", "Z")


def _aware_utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise CommandValidationError("command clock must return a timezone-aware datetime")
    return value.astimezone(UTC)


def _monotonic_seconds(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Number) or not math.isfinite(float(value)):
        raise CommandValidationError("command monotonic clock must return finite seconds")
    return float(value)
