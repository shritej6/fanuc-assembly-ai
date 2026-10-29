# Live OEE and Andon dashboard

T16 provides a read-only operations interface backed by an append-only JSONL run log. It never sends robot commands,
and it keeps `mock`, `sim`, `replay`, and `cell` sources visibly distinct.

## Metrics

- Availability = PackML `Execute` seconds / elapsed planned-production seconds.
- Performance = sum of measured ideal cycle seconds for completed products / PackML `Execute` seconds.
- Quality = good products / completed products.
- OEE = Availability × Performance × Quality.
- First-pass yield = `good_first_pass` / completed products.
- Final yield = (`good_first_pass` + `good_after_rebuild`) / completed products.
- Builds per assist = good products / recorded assists.

Missing inputs produce an unavailable metric and an explanation. Performance therefore requires a measured ideal cycle
time for every product represented in the reporting window. Values are not capped or filled with targets.

## Run-log format

Each line has exactly `version`, `source`, `event`, `timestamp`, and `payload`. Supported sources are `mock`, `sim`,
`replay`, and `cell`; one reporting window cannot mix them.

```json
{"version":1,"source":"mock","event":"packml","timestamp":"2026-10-29T15:00:00Z","payload":{"state":"Execute","reason":null,"product":1}}
{"version":1,"source":"mock","event":"product_result","timestamp":"2026-10-29T15:01:00Z","payload":{"product":1,"result":"good_first_pass","assists":0,"cycle_time_s":60.0}}
{"version":1,"source":"mock","event":"escape","timestamp":"2026-10-29T15:01:01Z","payload":{"product":1,"detail":"fault-injection evaluation escape"}}
{"version":1,"source":"mock","event":"heartbeat","timestamp":"2026-10-29T15:01:05Z","payload":{}}
```

The first PackML event must be at or before the planned window start for Availability to be available. Product results
use the four values in `contracts/birth_certificate.schema.json`. An escape is an explicit evaluation observation; the
dashboard never infers one from a rejected product.

## Record a live run

`ControllerRunLogRecorder` is the producer boundary between T09 and the dashboard. It synchronizes public
`PackMLStateMachine.history`, accepts a validated T15 birth-certificate record at product completion, and records an
explicit evaluation escape. The writer is local, append-only, file-locked, bounded to 16 MiB by default, source-pure,
replay-resistant for exact events, and synced to disk before returning.

```python
from pathlib import Path

from industry.oee import EventSource
from industry.run_log import ControllerRunLogRecorder, RunLogWriter

recorder = ControllerRunLogRecorder(RunLogWriter(Path("eval/runs/run-001.jsonl")), EventSource.CELL)
recorder.sync_packml(controller.state_machine)
recorder.record_product_result(validated_birth_certificate, recorded_at=finished_at)
recorder.record_heartbeat(recorded_at=observed_at)
```

Call `sync_packml` after each state transition and once during startup. On restart it verifies that recorded PackML
events are an exact prefix of the controller history before appending anything; a divergent history fails closed.
Schedule `record_heartbeat` periodically while the controller process is healthy, at an interval shorter than
`--stale-after-s`. A heartbeat is liveness evidence only; it never changes PackML state or any OEE numerator.

## Run locally

The following is an example invocation. Replace every time and ideal cycle with values measured for the run being shown.

```bash
uv run python -m industry.dashboard \
  --log eval/runs/demo.jsonl \
  --planned-start 2026-10-29T15:00:00Z \
  --planned-end 2026-10-29T16:00:00Z \
  --ideal-cycle 1=60.0 \
  --stale-after-s 15
```

Open <http://127.0.0.1:8080>. The browser polls the read-only snapshot endpoint every two seconds. A malformed, missing,
mixed-source, or stale log is displayed conspicuously and never converted to a green cell status. Reporting windows
use half-open `[planned_start, effective_end)` bounds, so adjacent windows never count the same product twice.
