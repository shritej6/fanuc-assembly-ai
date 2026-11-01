"""Strict JSONL boundary for the read-only OEE dashboard.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from assembly.birth_certificate import BirthCertificateError, validate_birth_certificate
from assembly.state_machine import PackMLState, PackMLStateMachine, StateRecord
from industry.certificate_evidence import CertificateEvidenceError, validate_certificate_evidence
from industry.events import RunEventError, validate_scope
from industry.lifecycle import LifecycleEvidenceError, validate_terminal_lifecycle
from industry.oee import (
    DashboardMetricError,
    EscapeEvent,
    EventSource,
    HeartbeatEvent,
    PackMLEvent,
    ProductResult,
    ProductResultEvent,
    RunEvent,
)

_TOP_LEVEL_KEYS = frozenset({"version", "source", "event", "timestamp", "run_id", "rollout", "payload"})
_MAX_LOG_BYTES = 16 * 1024 * 1024


class RunLogError(DashboardMetricError):
    """The dashboard run log is unavailable or violates its strict format."""


class RunLogWriter:
    """Append validated events atomically to one bounded local run log."""

    def __init__(self, path: Path, *, max_bytes: int = _MAX_LOG_BYTES) -> None:
        if not isinstance(path, Path):
            raise RunLogError("run-log path must be a pathlib.Path")
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or not 1 <= max_bytes <= _MAX_LOG_BYTES:
            raise RunLogError(f"max_bytes must be an integer from 1 through {_MAX_LOG_BYTES}")
        self.path = path
        self.max_bytes = max_bytes

    def read(self) -> tuple[RunEvent, ...]:
        return _load_path(self.path, allow_missing=True, allow_empty=True)

    def append(self, event: RunEvent) -> None:
        _require_event(event)
        encoded = (encode_run_event(event) + "\n").encode()
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor = os.open(self.path, _append_flags(), 0o600)
        except OSError as error:
            raise RunLogError(f"cannot open run log {self.path}: {error}") from error
        try:
            self._append_locked(descriptor, event, encoded)
        finally:
            os.close(descriptor)

    def _append_locked(self, descriptor: int, event: RunEvent, encoded: bytes) -> None:
        locked = False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            locked = True
            size = os.fstat(descriptor).st_size
            if size > self.max_bytes:
                raise RunLogError(f"run log already exceeds configured {self.max_bytes}-byte limit")
            existing = _read_descriptor(descriptor, size)
            _validate_append(existing, event)
            if size + len(encoded) > self.max_bytes:
                raise RunLogError(f"run log would exceed configured {self.max_bytes}-byte limit")
            _write_exact(descriptor, encoded, size)
        except OSError as error:
            raise RunLogError(f"cannot append run log {self.path}: {error}") from error
        finally:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)


class ControllerRunLogRecorder:
    """Bridge public controller PackML history and terminal outcomes into JSONL."""

    def __init__(self, writer: RunLogWriter, source: EventSource, run_id: str, rollout: int) -> None:
        if not isinstance(writer, RunLogWriter):
            raise RunLogError("writer must be a RunLogWriter")
        if not isinstance(source, EventSource):
            raise RunLogError("source must be an EventSource")
        try:
            identity = validate_scope(run_id, rollout)
        except RunEventError as error:
            raise RunLogError(str(error)) from error
        self.writer = writer
        self.source = source
        self.run_id, self.rollout = identity

    def sync_packml(self, machine: PackMLStateMachine) -> int:
        if not isinstance(machine, PackMLStateMachine):
            raise RunLogError("machine must be a PackMLStateMachine")
        history = tuple(_state_event(record, self.source, self.run_id, self.rollout) for record in machine.history)
        existing = self.writer.read()
        _require_log_identity(existing, self.source, self.run_id)
        recorded = tuple(
            event
            for event in existing
            if isinstance(event, PackMLEvent) and event.run_id == self.run_id and event.rollout == self.rollout
        )
        if len(recorded) > len(history) or recorded != history[: len(recorded)]:
            raise RunLogError("recorded PackML history does not match the controller history prefix")
        for event in history[len(recorded) :]:
            self.writer.append(event)
        return len(history) - len(recorded)

    def record_product_result(self, record: Mapping[str, object], *, recorded_at: datetime) -> ProductResultEvent:
        try:
            certificate = validate_birth_certificate(_mapping(record, "birth certificate"))
            validate_certificate_evidence(certificate)
        except (BirthCertificateError, CertificateEvidenceError) as error:
            raise RunLogError(f"invalid birth certificate: {error}") from error
        self._require_certificate_identity(certificate)
        event = _product_event(
            recorded_at,
            self.source,
            self.run_id,
            self.rollout,
            _terminal_payload(certificate),
        )
        self.writer.append(event)
        return event

    def _require_certificate_identity(self, certificate: Mapping[str, object]) -> None:
        identity = (certificate.get("run_id"), certificate.get("rollout"))
        if identity != (self.run_id, self.rollout):
            raise RunLogError("birth certificate run_id/rollout does not match recorder identity")

    def record_escape(self, product: int, detail: str, *, recorded_at: datetime) -> EscapeEvent:
        event = EscapeEvent(recorded_at, self.source, self.run_id, self.rollout, product, detail)
        self.writer.append(event)
        return event

    def record_heartbeat(self, *, recorded_at: datetime) -> HeartbeatEvent:
        event = HeartbeatEvent(recorded_at, self.source, self.run_id, self.rollout)
        self.writer.append(event)
        return event


def load_run_log(path: Path) -> tuple[RunEvent, ...]:
    """Read a bounded append-only log without accepting partial event shapes."""
    return _load_path(Path(path), allow_missing=False, allow_empty=False)


def encode_run_event(event: RunEvent) -> str:
    """Serialize one validated event for an append-only JSONL writer."""
    _require_event(event)
    record = {
        "version": 1,
        "source": event.source.value,
        "event": _event_name(event),
        "timestamp": _timestamp_text(event.timestamp),
        "run_id": event.run_id,
        "rollout": event.rollout,
        "payload": _event_payload(event),
    }
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _parse_line(line: str, number: int) -> RunEvent:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise RunLogError(f"line {number} is not valid JSON: {error.msg}") from error
    try:
        return _parse_record(record)
    except (RunLogError, DashboardMetricError, RunEventError) as error:
        raise RunLogError(f"line {number}: {error}") from error


def _parse_record(record: object) -> RunEvent:
    mapping = _mapping(record, "event record")
    _exact_keys(mapping, _TOP_LEVEL_KEYS, "event record")
    if type(mapping["version"]) is not int or mapping["version"] != 1:
        raise RunLogError("version must be 1")
    source = _enum(EventSource, mapping["source"], "source")
    timestamp = _timestamp(mapping["timestamp"])
    run_id, rollout = _scope(mapping["run_id"], mapping["rollout"])
    payload = _mapping(mapping["payload"], "payload")
    event_type = mapping["event"]
    if event_type == "packml":
        return _packml_event(timestamp, source, run_id, rollout, payload)
    if event_type == "product_result":
        return _product_event(timestamp, source, run_id, rollout, payload)
    if event_type == "escape":
        return _escape_event(timestamp, source, run_id, rollout, payload)
    if event_type == "heartbeat":
        _exact_keys(payload, frozenset(), "heartbeat payload")
        return HeartbeatEvent(timestamp, source, run_id, rollout)
    raise RunLogError(f"unsupported event type {event_type!r}")


def _packml_event(
    timestamp: datetime,
    source: EventSource,
    run_id: str,
    rollout: int,
    payload: Mapping[str, object],
) -> PackMLEvent:
    _exact_keys(payload, frozenset({"state", "reason", "product"}), "PackML payload")
    state = _enum(PackMLState, payload["state"], "PackML state")
    reason = payload["reason"]
    if reason is not None and not isinstance(reason, str):
        raise RunLogError("PackML reason must be a string or null")
    return PackMLEvent(timestamp, source, run_id, rollout, state, reason, _optional_product(payload["product"]))


def _product_event(
    timestamp: datetime,
    source: EventSource,
    run_id: str,
    rollout: int,
    payload: Mapping[str, object],
) -> ProductResultEvent:
    keys = frozenset({"product", "result", "assists", "cycle_time_s"})
    _exact_keys(payload, keys, "product-result payload")
    cycle_time = payload["cycle_time_s"]
    if cycle_time is not None and (isinstance(cycle_time, bool) or not isinstance(cycle_time, (int, float))):
        raise RunLogError("cycle_time_s must be a number or null")
    return ProductResultEvent(
        timestamp,
        source,
        run_id,
        rollout,
        _product(payload["product"]),
        _enum(ProductResult, payload["result"], "product result"),
        _nonnegative_integer(payload["assists"], "assists"),
        float(cycle_time) if cycle_time is not None else None,
    )


def _escape_event(
    timestamp: datetime,
    source: EventSource,
    run_id: str,
    rollout: int,
    payload: Mapping[str, object],
) -> EscapeEvent:
    _exact_keys(payload, frozenset({"product", "detail"}), "escape payload")
    detail = payload["detail"]
    if not isinstance(detail, str):
        raise RunLogError("escape detail must be a string")
    return EscapeEvent(timestamp, source, run_id, rollout, _product(payload["product"]), detail)


def _event_name(event: RunEvent) -> str:
    if isinstance(event, PackMLEvent):
        return "packml"
    if isinstance(event, ProductResultEvent):
        return "product_result"
    if isinstance(event, EscapeEvent):
        return "escape"
    if isinstance(event, HeartbeatEvent):
        return "heartbeat"
    raise RunLogError(f"unsupported event {type(event).__name__}")


def _event_payload(event: RunEvent) -> dict[str, object]:
    if isinstance(event, PackMLEvent):
        return {"state": event.state.value, "reason": event.reason, "product": event.product}
    if isinstance(event, ProductResultEvent):
        return {
            "product": event.product,
            "result": event.result.value,
            "assists": event.assists,
            "cycle_time_s": event.cycle_time_s,
        }
    if isinstance(event, EscapeEvent):
        return {"product": event.product, "detail": event.detail}
    if isinstance(event, HeartbeatEvent):
        return {}
    raise RunLogError(f"unsupported event {type(event).__name__}")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise RunLogError(f"{label} must be an object with string keys")
    return value


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RunLogError(f"{label} keys mismatch; missing={missing}, extra={extra}")


def _timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise RunLogError("timestamp must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise RunLogError("timestamp must be an ISO 8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RunLogError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _timestamp_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _scope(run_id: object, rollout: object) -> tuple[str, int]:
    try:
        return validate_scope(run_id, rollout)
    except RunEventError as error:
        raise RunLogError(str(error)) from error


def _enum(enum_type: type[Any], value: object, label: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as error:
        raise RunLogError(f"invalid {label} {value!r}") from error


def _product(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 6:
        raise RunLogError("product must be an integer from 1 through 6")
    return value


def _optional_product(value: object) -> int | None:
    return None if value is None else _product(value)


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunLogError(f"{label} must be a nonnegative integer")
    return value


def _parse_text(content: str, *, allow_empty: bool = False) -> tuple[RunEvent, ...]:
    lines = content.splitlines()
    events = tuple(_parse_line(line, number) for number, line in enumerate(lines, start=1) if line.strip())
    if not events and not allow_empty:
        raise RunLogError("run log contains no events")
    _validate_sequence(events)
    return events


def _append_flags() -> int:
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    return flags


def _read_descriptor(descriptor: int, size: int, *, allow_empty: bool = True) -> tuple[RunEvent, ...]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    raw = os.read(descriptor, size)
    if len(raw) != size:
        raise RunLogError("run-log read was incomplete")
    try:
        content = raw.decode("utf-8")
    except UnicodeError as error:
        raise RunLogError("run log is not valid UTF-8") from error
    return _parse_text(content, allow_empty=allow_empty)


def _load_path(path: Path, *, allow_missing: bool, allow_empty: bool) -> tuple[RunEvent, ...]:
    try:
        descriptor = os.open(path, _read_flags())
    except FileNotFoundError as error:
        if allow_missing:
            return ()
        raise RunLogError(f"cannot open run log {path}: {error}") from error
    except OSError as error:
        raise RunLogError(f"cannot open run log {path}: {error}") from error
    locked = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        locked = True
        size = os.fstat(descriptor).st_size
        if size > _MAX_LOG_BYTES:
            raise RunLogError(f"run log exceeds {_MAX_LOG_BYTES} bytes")
        return _read_descriptor(descriptor, size, allow_empty=allow_empty)
    except OSError as error:
        raise RunLogError(f"cannot read run log {path}: {error}") from error
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _validate_append(existing: tuple[RunEvent, ...], event: RunEvent) -> None:
    _require_event(event)
    _require_log_identity(existing, event.source, event.run_id)
    if existing and event.timestamp < existing[-1].timestamp:
        raise RunLogError("run-log events must preserve timestamp order")
    if event in existing:
        raise RunLogError("run log already contains this event")
    if isinstance(event, ProductResultEvent):
        _require_unique_terminal(existing, event)
        _require_terminal_lifecycle(existing, event)


def _require_log_identity(events: tuple[RunEvent, ...], source: EventSource, run_id: str) -> None:
    sources = {event.source for event in events}
    if sources and sources != {source}:
        raise RunLogError("run log source does not match the recorder source")
    run_ids = {event.run_id for event in events}
    if run_ids and run_ids != {run_id}:
        raise RunLogError("run log run_id does not match the recorder identity")


def _validate_sequence(events: tuple[RunEvent, ...]) -> None:
    prefix: list[RunEvent] = []
    for event in events:
        _validate_append(tuple(prefix), event)
        prefix.append(event)


def _require_unique_terminal(existing: tuple[RunEvent, ...], event: ProductResultEvent) -> None:
    identity = (event.run_id, event.rollout, event.product)
    terminals = (
        item
        for item in existing
        if isinstance(item, ProductResultEvent) and (item.run_id, item.rollout, item.product) == identity
    )
    if next(terminals, None) is not None:
        raise RunLogError(f"terminal result already exists for identity {identity!r}")


def _require_terminal_lifecycle(existing: tuple[RunEvent, ...], event: ProductResultEvent) -> None:
    try:
        validate_terminal_lifecycle(existing, event)
    except LifecycleEvidenceError as error:
        raise RunLogError(str(error)) from error


def _require_event(event: object) -> None:
    if not isinstance(event, (PackMLEvent, ProductResultEvent, EscapeEvent, HeartbeatEvent)):
        raise RunLogError(f"unsupported event {type(event).__name__}")


def _read_flags() -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    return flags | getattr(os, "O_NOFOLLOW", 0)


def _write_exact(descriptor: int, data: bytes, original_size: int) -> None:
    offset = 0
    try:
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("write returned no progress")
            offset += written
        os.fsync(descriptor)
    except OSError as error:
        try:
            os.ftruncate(descriptor, original_size)
            os.fsync(descriptor)
        except OSError as rollback_error:
            detail = f"run-log append failed; rollback durability is uncertain: {rollback_error}"
            raise RunLogError(detail) from error
        raise RunLogError(f"run-log append failed and was rolled back: {error}") from error


def _state_event(record: object, source: EventSource, run_id: str, rollout: int) -> PackMLEvent:
    if not isinstance(record, StateRecord):
        raise RunLogError("PackML history must contain StateRecord values")
    return PackMLEvent(_timestamp(record.timestamp), source, run_id, rollout, record.state, record.reason, record.product)


def _terminal_payload(certificate: Mapping[str, object]) -> Mapping[str, object]:
    required = frozenset({"product", "result", "assists", "cycle_time_s"})
    missing = required - certificate.keys()
    if missing:
        raise RunLogError(f"validated birth certificate is missing dashboard fields {sorted(missing)}")
    return {name: certificate[name] for name in required}
