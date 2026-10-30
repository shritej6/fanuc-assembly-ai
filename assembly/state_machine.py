"""Fail-closed PackML state tracking for one active product."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol


class StateMachineError(Exception):
    """Base class for fail-closed state-machine failures."""


class UnknownStateError(StateMachineError):
    def __init__(self, state: object) -> None:
        self.state = state
        super().__init__(f"Unknown PackML state {state!r}")


class InvalidTransitionError(StateMachineError):
    def __init__(self, current: "PackMLState", target: "PackMLState") -> None:
        self.current = current
        self.target = target
        super().__init__(f"Invalid PackML transition {current.value} -> {target.value}")


class InvalidProductError(StateMachineError):
    def __init__(self, product: object) -> None:
        self.product = product
        super().__init__(f"Product must be an integer from 1 through 6, got {product!r}")


class InvalidReasonError(StateMachineError):
    def __init__(self, reason: object) -> None:
        self.reason = reason
        super().__init__(f"PackML reason must be a string or null, got {reason!r}")


class InvalidTimestampError(StateMachineError):
    def __init__(self, timestamp: object) -> None:
        self.timestamp = timestamp
        super().__init__("PackML timestamps require a timezone-aware datetime")


class ProductOwnershipError(StateMachineError):
    def __init__(self, owner: int, requested: int) -> None:
        self.owner = owner
        self.requested = requested
        super().__init__(f"Product {owner} owns the cell; cannot start product {requested}")


class ReservationReleaseError(StateMachineError):
    def __init__(self, product: int, error: Exception) -> None:
        self.product = product
        super().__init__(f"Cannot release reservations for product {product}: {error}")


class ReservationReleaser(Protocol):
    def release_reservations(self, product: int | None = None) -> object: ...


class PackMLState(StrEnum):
    """The exact ISA-TR88 subset shared in the PackML contract."""

    IDLE = "Idle"
    STARTING = "Starting"
    EXECUTE = "Execute"
    HELD = "Held"
    SUSPENDED = "Suspended"
    COMPLETING = "Completing"
    COMPLETE = "Complete"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    ABORTING = "Aborting"
    ABORTED = "Aborted"
    RESETTING = "Resetting"


_ALLOWED_TRANSITIONS = {
    PackMLState.IDLE: frozenset({PackMLState.STARTING, PackMLState.HELD}),
    PackMLState.STARTING: frozenset({PackMLState.EXECUTE, PackMLState.STOPPING, PackMLState.ABORTING}),
    PackMLState.EXECUTE: frozenset(
        {
            PackMLState.HELD,
            PackMLState.SUSPENDED,
            PackMLState.COMPLETING,
            PackMLState.STOPPING,
            PackMLState.ABORTING,
        }
    ),
    PackMLState.HELD: frozenset(
        {PackMLState.EXECUTE, PackMLState.RESETTING, PackMLState.STOPPING, PackMLState.ABORTING}
    ),
    PackMLState.SUSPENDED: frozenset(
        {PackMLState.EXECUTE, PackMLState.HELD, PackMLState.STOPPING, PackMLState.ABORTING}
    ),
    PackMLState.COMPLETING: frozenset({PackMLState.COMPLETE, PackMLState.STOPPING, PackMLState.ABORTING}),
    PackMLState.COMPLETE: frozenset({PackMLState.RESETTING, PackMLState.ABORTING}),
    PackMLState.STOPPING: frozenset({PackMLState.STOPPED, PackMLState.ABORTING}),
    PackMLState.STOPPED: frozenset({PackMLState.RESETTING, PackMLState.ABORTING}),
    PackMLState.ABORTING: frozenset({PackMLState.ABORTED}),
    PackMLState.ABORTED: frozenset({PackMLState.RESETTING}),
    PackMLState.RESETTING: frozenset({PackMLState.IDLE, PackMLState.ABORTING}),
}


@dataclass(frozen=True, slots=True)
class StateRecord:
    """One JSON-contract-shaped state observation."""

    state: PackMLState
    reason: str | None
    product: int | None
    timestamp: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "reason": self.reason,
            "product": self.product,
            "timestamp": self.timestamp,
        }


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _timestamp(value: object) -> str:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise InvalidTimestampError(value)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


class PackMLStateMachine:
    """Record PackML state while enforcing transition and product ownership rules."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] | None = None,
        reservation_releaser: ReservationReleaser | None = None,
    ) -> None:
        self._clock = clock or _utc_now
        self._reservation_releaser = reservation_releaser
        self._current_product: int | None = None
        self._reservations_released = True
        initial = StateRecord(PackMLState.IDLE, None, None, _timestamp(self._clock()))
        self._history = [initial]

    @property
    def current_record(self) -> StateRecord:
        return self._history[-1]

    @property
    def state(self) -> PackMLState:
        return self.current_record.state

    @property
    def current_product(self) -> int | None:
        return self._current_product

    @property
    def history(self) -> tuple[StateRecord, ...]:
        return tuple(self._history)

    def transition(
        self,
        target: PackMLState | str,
        *,
        reason: str | None = None,
        product: int | None = None,
    ) -> StateRecord:
        normalized = self._normalize_state(target)
        self._validate_reason(reason)
        requested_product = self._validate_ownership(normalized, product)
        if normalized not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidTransitionError(self.state, normalized)
        timestamp = _timestamp(self._clock())
        self._release_if_needed(normalized)
        if normalized is PackMLState.STARTING:
            self._current_product = requested_product
            self._reservations_released = False
        record_product = None if normalized is PackMLState.IDLE else self._current_product
        record = StateRecord(normalized, reason, record_product, timestamp)
        self._history.append(record)
        if normalized is PackMLState.IDLE:
            self._current_product = None
        return record

    def start(self, product: int) -> StateRecord:
        self._validate_product(product)
        if self._current_product is not None and self._current_product != product:
            raise ProductOwnershipError(self._current_product, product)
        return self.transition(PackMLState.STARTING, product=product)

    def hold(self, reason: str) -> StateRecord:
        return self.transition(PackMLState.HELD, reason=reason)

    def suspend(self, reason: str) -> StateRecord:
        return self.transition(PackMLState.SUSPENDED, reason=reason)

    def resume(self) -> StateRecord:
        return self.transition(PackMLState.EXECUTE)

    def complete(self) -> StateRecord:
        return self.transition(PackMLState.COMPLETING)

    def stop(self, reason: str) -> StateRecord:
        return self.transition(PackMLState.STOPPING, reason=reason)

    def abort(self, reason: str) -> StateRecord:
        return self.transition(PackMLState.ABORTING, reason=reason)

    def reset(self, reason: str | None = None) -> StateRecord:
        return self.transition(PackMLState.RESETTING, reason=reason)

    def _normalize_state(self, target: PackMLState | str) -> PackMLState:
        try:
            return PackMLState(target)
        except (TypeError, ValueError) as error:
            raise UnknownStateError(target) from error

    def _validate_ownership(self, target: PackMLState, product: int | None) -> int | None:
        if target is PackMLState.STARTING:
            self._validate_product(product)
        if product is not None and self._current_product not in (None, product):
            raise ProductOwnershipError(self._current_product, product)
        return product if target is PackMLState.STARTING else self._current_product

    def _release_if_needed(self, target: PackMLState) -> None:
        release_states = {PackMLState.STOPPING, PackMLState.ABORTING, PackMLState.RESETTING}
        if target not in release_states or self._reservations_released:
            return
        product = self._current_product
        if product is None or self._reservation_releaser is None:
            self._reservations_released = True
            return
        try:
            self._reservation_releaser.release_reservations(product)
        except Exception as error:
            raise ReservationReleaseError(product, error) from error
        self._reservations_released = True

    @staticmethod
    def _validate_product(product: object) -> None:
        if isinstance(product, bool) or not isinstance(product, int) or not 1 <= product <= 6:
            raise InvalidProductError(product)

    @staticmethod
    def _validate_reason(reason: object) -> None:
        if reason is not None and not isinstance(reason, str):
            raise InvalidReasonError(reason)
