import json
import os
from collections import Counter
from datetime import UTC, datetime, timedelta
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
RUN_ID = "run-2026-10-29"
ROLLOUT = 3


def _certificate(
    *,
    product: int = 1,
    rollout: int = ROLLOUT,
    result: str = "good_first_pass",
    cycle_time_s: float | None = 4.0,
) -> dict[str, object]:
    return {
        "run_id": RUN_ID,
        "rollout": rollout,
        "product": product,
        "base_instance": f"base-{product}",
        "result": result,
        "steps": [],
        "rebuilds": 0,
        "assists": 0,
        "cycle_time_s": cycle_time_s,
        "final_checks": {"recipe": result.startswith("good_"), "cosmos_reason": None},
    }


def _events() -> tuple[PackMLEvent | ProductResultEvent | EscapeEvent | HeartbeatEvent, ...]:
    return (
        PackMLEvent(NOW, EventSource.REPLAY, RUN_ID, ROLLOUT, PackMLState.COMPLETE, None, 1),
        ProductResultEvent(
            NOW, EventSource.REPLAY, RUN_ID, ROLLOUT, 1, ProductResult.GOOD_AFTER_REBUILD, 1, 42.5
        ),
        EscapeEvent(NOW, EventSource.REPLAY, RUN_ID, ROLLOUT, 1, "evaluation escape"),
        HeartbeatEvent(NOW, EventSource.REPLAY, RUN_ID, ROLLOUT),
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
        "run_id": RUN_ID,
        "rollout": ROLLOUT,
        "payload": {"state": "Complete", "reason": None, "product": 1},
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.update(extra=True),
        lambda record: record.update(version=2),
        lambda record: record.update(version=True),
        lambda record: record.update(source="unknown"),
        lambda record: record.update(run_id=""),
        lambda record: record.update(rollout=-1),
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
    times = iter(datetime(2026, 10, 29, 15, 0, second=second, tzinfo=UTC) for second in range(8))
    machine = PackMLStateMachine(clock=lambda: next(times))
    machine.start(1)
    machine.transition(PackMLState.EXECUTE)
    machine.complete()
    machine.transition(PackMLState.COMPLETE)
    writer = RunLogWriter(tmp_path / "run.jsonl")
    recorder = ControllerRunLogRecorder(writer, EventSource.MOCK, RUN_ID, ROLLOUT)

    assert recorder.sync_packml(machine) == 5
    assert recorder.sync_packml(machine) == 0
    result = recorder.record_product_result(_certificate(), recorded_at=NOW.replace(second=7))

    events = load_run_log(writer.path)
    assert len(events) == 6
    assert events[-1] == result
    assert isinstance(events[-2], PackMLEvent)
    assert events[-2].state is PackMLState.COMPLETE
    assert ControllerRunLogRecorder(writer, EventSource.MOCK, RUN_ID, ROLLOUT).sync_packml(machine) == 0


def test_controller_recorder_rejects_divergent_packml_history(tmp_path: Path) -> None:
    machine = PackMLStateMachine(clock=lambda: NOW)
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.append(PackMLEvent(NOW, EventSource.MOCK, RUN_ID, ROLLOUT, PackMLState.EXECUTE, None, None))

    with pytest.raises(RunLogError, match="history prefix"):
        ControllerRunLogRecorder(writer, EventSource.MOCK, RUN_ID, ROLLOUT).sync_packml(machine)


def test_writer_rejects_duplicates_mixed_sources_and_size_overflow(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    writer = RunLogWriter(path)
    event = PackMLEvent(NOW, EventSource.MOCK, RUN_ID, ROLLOUT, PackMLState.EXECUTE, None, 1)
    writer.append(event)

    with pytest.raises(RunLogError, match="already contains"):
        writer.append(event)
    with pytest.raises(RunLogError, match="source"):
        writer.append(PackMLEvent(NOW, EventSource.CELL, RUN_ID, ROLLOUT, PackMLState.EXECUTE, None, 1))
    with pytest.raises(RunLogError, match="exceed configured"):
        RunLogWriter(tmp_path / "tiny.jsonl", max_bytes=8).append(event)


def test_recorder_writes_explicit_escape(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")
    recorder = ControllerRunLogRecorder(writer, EventSource.REPLAY, RUN_ID, ROLLOUT)

    event = recorder.record_escape(2, "injected wrong part passed final QC", recorded_at=NOW)

    assert load_run_log(writer.path) == (event,)


def test_recorder_writes_explicit_heartbeat(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")

    event = ControllerRunLogRecorder(writer, EventSource.MOCK, RUN_ID, ROLLOUT).record_heartbeat(recorded_at=NOW)

    assert load_run_log(writer.path) == (event,)


def test_failed_partial_append_rolls_back_to_last_valid_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")
    original = PackMLEvent(NOW, EventSource.MOCK, RUN_ID, ROLLOUT, PackMLState.EXECUTE, None, 1)
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
        writer.append(HeartbeatEvent(NOW, EventSource.MOCK, RUN_ID, ROLLOUT))

    assert writer.path.read_bytes() == original_bytes
    assert load_run_log(writer.path) == (original,)


def test_recorder_scopes_new_t09_history_after_actual_controller_reset(tmp_path: Path) -> None:
    from tests.test_controller import ScriptedPerception, _controller, _observed_products

    controller = _controller(ScriptedPerception(_observed_products((1, 2)), set(), Counter(), set()))
    writer = RunLogWriter(tmp_path / "run.jsonl")
    controller.state_machine.start(1)
    controller.state_machine.transition(PackMLState.EXECUTE)
    first = ControllerRunLogRecorder(writer, EventSource.MOCK, RUN_ID, 0)
    assert first.sync_packml(controller.state_machine) == 3

    controller.reset()
    controller.state_machine.start(2)
    controller.state_machine.transition(PackMLState.EXECUTE)
    second = ControllerRunLogRecorder(writer, EventSource.MOCK, RUN_ID, 1)
    assert second.sync_packml(controller.state_machine) == 3
    assert [event.rollout for event in load_run_log(writer.path)] == [0, 0, 0, 1, 1, 1]


def test_terminal_result_is_unique_when_retry_changes_timestamp(tmp_path: Path) -> None:
    machine = _complete_machine(1)
    recorder = ControllerRunLogRecorder(RunLogWriter(tmp_path / "run.jsonl"), EventSource.MOCK, RUN_ID, ROLLOUT)
    recorder.sync_packml(machine)
    recorder.record_product_result(_certificate(), recorded_at=NOW.replace(second=6))

    with pytest.raises(RunLogError, match="terminal result"):
        recorder.record_product_result(_certificate(), recorded_at=NOW.replace(second=7))


def test_terminal_result_rejects_invalid_t15_certificate(tmp_path: Path) -> None:
    recorder = ControllerRunLogRecorder(RunLogWriter(tmp_path / "run.jsonl"), EventSource.MOCK, RUN_ID, ROLLOUT)

    with pytest.raises(RunLogError, match="invalid birth certificate"):
        recorder.record_product_result({"product": 1}, recorded_at=NOW)


def test_terminal_result_rejects_wrong_product_lifecycle(tmp_path: Path) -> None:
    recorder = ControllerRunLogRecorder(RunLogWriter(tmp_path / "run.jsonl"), EventSource.MOCK, RUN_ID, ROLLOUT)
    recorder.sync_packml(_complete_machine(1))

    with pytest.raises(RunLogError, match="PackML lifecycle"):
        recorder.record_product_result(_certificate(product=2), recorded_at=NOW.replace(second=6))


def test_terminal_result_requires_packml_lifecycle(tmp_path: Path) -> None:
    recorder = ControllerRunLogRecorder(RunLogWriter(tmp_path / "run.jsonl"), EventSource.MOCK, RUN_ID, ROLLOUT)

    with pytest.raises(RunLogError, match="PackML lifecycle"):
        recorder.record_product_result(_certificate(), recorded_at=NOW)


def test_writer_rejects_out_of_order_evidence(tmp_path: Path) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.append(HeartbeatEvent(NOW, EventSource.MOCK, RUN_ID, ROLLOUT))

    with pytest.raises(RunLogError, match="timestamp order"):
        writer.append(HeartbeatEvent(NOW.replace(second=0) - timedelta(seconds=1), EventSource.MOCK, RUN_ID, ROLLOUT))


def test_fsync_failure_rolls_back_to_last_durable_event(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")
    original = HeartbeatEvent(NOW, EventSource.MOCK, RUN_ID, ROLLOUT)
    writer.append(original)
    real_fsync = os.fsync
    calls = 0

    def fail_commit_then_sync_rollback(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", fail_commit_then_sync_rollback)
    with pytest.raises(RunLogError, match="rolled back"):
        writer.append(HeartbeatEvent(NOW.replace(second=1), EventSource.MOCK, RUN_ID, ROLLOUT))

    assert load_run_log(writer.path) == (original,)


def test_double_fsync_failure_reports_uncertain_rollback_durability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = RunLogWriter(tmp_path / "run.jsonl")
    writer.append(HeartbeatEvent(NOW, EventSource.MOCK, RUN_ID, ROLLOUT))

    def fail_fsync(descriptor: int) -> None:
        del descriptor
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(RunLogError, match="rollback durability is uncertain"):
        writer.append(HeartbeatEvent(NOW.replace(second=1), EventSource.MOCK, RUN_ID, ROLLOUT))


def _complete_machine(product: int) -> PackMLStateMachine:
    times = iter(NOW.replace(second=second) for second in range(6))
    machine = PackMLStateMachine(clock=lambda: next(times))
    machine.start(product)
    machine.transition(PackMLState.EXECUTE)
    machine.complete()
    machine.transition(PackMLState.COMPLETE)
    return machine
