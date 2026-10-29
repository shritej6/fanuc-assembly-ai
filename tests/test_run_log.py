import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from assembly.state_machine import PackMLState, PackMLStateMachine
from industry.oee import EscapeEvent, EventSource, HeartbeatEvent, PackMLEvent, ProductResult, ProductResultEvent
from industry.run_log import (
    ControllerRunLogRecorder,
    RunLogError,
    RunLogWriter,
    encode_run_event,
    load_run_log,
)

NOW = datetime(2026, 10, 29, 15, tzinfo=UTC)


def _events() -> tuple[PackMLEvent | ProductResultEvent | EscapeEvent | HeartbeatEvent, ...]:
    return (
        PackMLEvent(NOW, EventSource.REPLAY, PackMLState.EXECUTE, None, 1),
        ProductResultEvent(NOW, EventSource.REPLAY, 1, ProductResult.GOOD_AFTER_REBUILD, 1, 42.5),
        EscapeEvent(NOW, EventSource.REPLAY, 1, "evaluation escape"),
        HeartbeatEvent(NOW, EventSource.REPLAY),
    )


def _write(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_round_trips_all_supported_event_types(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _write(path, [encode_run_event(event) for event in _events()])

    assert load_run_log(path) == _events()


def test_encoder_uses_stable_minimal_shape() -> None:
    record = json.loads(encode_run_event(_events()[0]))

    assert record == {
        "version": 1,
        "source": "replay",
        "event": "packml",
        "timestamp": "2026-10-29T15:00:00Z",
        "payload": {"state": "Execute", "reason": None, "product": 1},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(extra=True),
        lambda record: record.update(version=2),
        lambda record: record.update(version=True),
        lambda record: record.update(source="unknown"),
        lambda record: record.update(event="other"),
        lambda record: record.update(timestamp="2026-10-29T15:00:00"),
        lambda record: record["payload"].update(extra=True),
    ],
)
def test_rejects_unknown_or_partial_shapes(tmp_path: Path, mutation: object) -> None:
    record = json.loads(encode_run_event(_events()[0]))
    mutation(record)  # type: ignore[operator]
    path = tmp_path / "bad.jsonl"
    _write(path, [json.dumps(record)])

    with pytest.raises(RunLogError, match="line 1"):
        load_run_log(path)


def test_rejects_invalid_json_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    _write(path, [encode_run_event(_events()[0]), "{"])

    with pytest.raises(RunLogError, match="line 2"):
        load_run_log(path)


def test_rejects_empty_log(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(RunLogError, match="no events"):
        load_run_log(path)


def test_missing_log_is_contextual() -> None:
    with pytest.raises(RunLogError, match="cannot open"):
        load_run_log(Path("does-not-exist.jsonl"))


def test_encoder_and_writer_reject_unsupported_runtime_event(tmp_path: Path) -> None:
    unsupported = object()

    with pytest.raises(RunLogError, match="unsupported event object"):
        encode_run_event(unsupported)  # type: ignore[arg-type]
    with pytest.raises(RunLogError, match="unsupported event object"):
        RunLogWriter(tmp_path / "run.jsonl").append(unsupported)  # type: ignore[arg-type]


def test_writer_read_treats_only_a_missing_or_empty_log_as_no_events(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")

    assert writer.read() == ()
    writer.path.touch()
    assert writer.read() == ()
    writer.path.unlink()
    writer.path.mkdir()
    with pytest.raises(RunLogError, match="cannot read"):
        writer.read()


def test_controller_recorder_produces_live_log_from_t09_and_terminal_record(tmp_path: Path) -> None:
    times = iter(datetime(2026, 10, 29, 15, 0, second=second, tzinfo=UTC) for second in range(5))
    machine = PackMLStateMachine(clock=lambda: next(times))
    machine.start(1)
    machine.transition(PackMLState.EXECUTE)
    writer = RunLogWriter(tmp_path / "run.jsonl")
    recorder = ControllerRunLogRecorder(writer, EventSource.MOCK)

    assert recorder.sync_packml(machine) == 3
    assert recorder.sync_packml(machine) == 0
    machine.hold("operator assist")
    assert recorder.sync_packml(machine) == 1
    result = recorder.record_product_result(
        {"product": 1, "result": "good_after_rebuild", "assists": 1, "cycle_time_s": 4.0},
        recorded_at=NOW,
    )

    events = load_run_log(writer.path)
    assert len(events) == 5
    assert events[-1] == result
    assert isinstance(events[-2], PackMLEvent)
    assert events[-2].state is PackMLState.HELD
    assert ControllerRunLogRecorder(writer, EventSource.MOCK).sync_packml(machine) == 0


def test_controller_recorder_rejects_divergent_packml_history(tmp_path: Path) -> None:
    machine = PackMLStateMachine(clock=lambda: NOW)
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.append(PackMLEvent(NOW, EventSource.MOCK, PackMLState.EXECUTE, None, None))

    with pytest.raises(RunLogError, match="history prefix"):
        ControllerRunLogRecorder(writer, EventSource.MOCK).sync_packml(machine)


def test_writer_rejects_duplicates_mixed_sources_and_size_overflow(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = RunLogWriter(path)
    event = PackMLEvent(NOW, EventSource.MOCK, PackMLState.EXECUTE, None, 1)
    writer.append(event)

    with pytest.raises(RunLogError, match="already contains"):
        writer.append(event)
    with pytest.raises(RunLogError, match="source"):
        writer.append(PackMLEvent(NOW, EventSource.CELL, PackMLState.EXECUTE, None, 1))
    with pytest.raises(RunLogError, match="exceed configured"):
        RunLogWriter(tmp_path / "tiny.jsonl", max_bytes=8).append(event)


def test_recorder_writes_explicit_escape(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")
    recorder = ControllerRunLogRecorder(writer, EventSource.REPLAY)

    event = recorder.record_escape(2, "injected wrong part passed final QC", recorded_at=NOW)

    assert load_run_log(writer.path) == (event,)


def test_recorder_writes_explicit_heartbeat(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")

    event = ControllerRunLogRecorder(writer, EventSource.MOCK).record_heartbeat(recorded_at=NOW)

    assert load_run_log(writer.path) == (event,)


def test_failed_partial_append_rolls_back_to_last_valid_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")
    original = PackMLEvent(NOW, EventSource.MOCK, PackMLState.EXECUTE, None, 1)
    writer.append(original)
    original_bytes = writer.path.read_bytes()
    real_write = os.write
    calls = 0

    def fail_after_one_byte(descriptor: int, data: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, data[:1])
        raise OSError("injected disk failure")

    monkeypatch.setattr(os, "write", fail_after_one_byte)
    with pytest.raises(RunLogError, match="rolled back"):
        writer.append(HeartbeatEvent(NOW, EventSource.MOCK))

    assert writer.path.read_bytes() == original_bytes
    assert load_run_log(writer.path) == (original,)
