"""Identity-scoped run events shared by the OEE calculator and JSONL boundary.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TypeAlias

from assembly.state_machine import PackMLState

_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class RunEventError(ValueError):
    """A run event is incomplete or malformed."""


class EventSource(StrEnum):
    MOCK = "mock"
    SIM = "sim"
    REPLAY = "replay"
    CELL = "cell"


class ProductResult(StrEnum):
    GOOD_FIRST_PASS = "good_first_pass"
    GOOD_AFTER_REBUILD = "good_after_rebuild"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PackMLEvent:
    timestamp: datetime
    source: EventSource
    run_id: str
    rollout: int
    state: PackMLState
    reason: str | None
    product: int | None

    def __post_init__(self) -> None:
        _common(self.timestamp, self.source, self.run_id, self.rollout)
        if not isinstance(self.state, PackMLState):
            raise RunEventError("PackML state must be a PackMLState")
        _product(self.product, optional=True)
        if self.reason is not None and not isinstance(self.reason, str):
            raise RunEventError("PackML reason must be a string or null")


@dataclass(frozen=True, slots=True)
class ProductResultEvent:
    timestamp: datetime
    source: EventSource
    run_id: str
    rollout: int
    product: int
    result: ProductResult
    assists: int
    cycle_time_s: float | None

    def __post_init__(self) -> None:
        _common(self.timestamp, self.source, self.run_id, self.rollout)
        _product(self.product)
        if not isinstance(self.result, ProductResult):
            raise RunEventError("product result must be a ProductResult")
        _nonnegative_integer(self.assists, "assists")
        if self.cycle_time_s is not None:
            _positive_number(self.cycle_time_s, "cycle_time_s")


@dataclass(frozen=True, slots=True)
class EscapeEvent:
    timestamp: datetime
    source: EventSource
    run_id: str
    rollout: int
    product: int
    detail: str

    def __post_init__(self) -> None:
        _common(self.timestamp, self.source, self.run_id, self.rollout)
        _product(self.product)
        if not isinstance(self.detail, str) or not self.detail.strip():
            raise RunEventError("escape detail must be a non-empty string")


@dataclass(frozen=True, slots=True)
class HeartbeatEvent:
    timestamp: datetime
    source: EventSource
    run_id: str
    rollout: int

    def __post_init__(self) -> None:
        _common(self.timestamp, self.source, self.run_id, self.rollout)


RunEvent: TypeAlias = PackMLEvent | ProductResultEvent | EscapeEvent | HeartbeatEvent


def validate_scope(run_id: object, rollout: object) -> tuple[str, int]:
    """Validate stable run/rollout identity without interpreting certificate evidence."""
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise RunEventError("run_id must be a safe 1-128 character identifier")
    if isinstance(rollout, bool) or not isinstance(rollout, int) or rollout < 0:
        raise RunEventError("rollout must be a nonnegative integer")
    return run_id, rollout


def _common(timestamp: object, source: object, run_id: object, rollout: object) -> None:
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise RunEventError("event timestamp must be a timezone-aware datetime")
    if not isinstance(source, EventSource):
        raise RunEventError("event source must be an EventSource")
    validate_scope(run_id, rollout)


def _product(value: object, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
        raise RunEventError(f"product must be an integer from 1 through 6, got {value!r}")


def _nonnegative_integer(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunEventError(f"{field} must be a nonnegative integer")


def _positive_number(value: object, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunEventError(f"{field} must be a positive finite number")
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise RunEventError(f"{field} must be a positive finite number")
