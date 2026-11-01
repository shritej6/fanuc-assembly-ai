# Live OEE and Andon dashboard

Authored by Shritej Gunda.

The dashboard provides a read-only operations interface backed by an append-only JSONL run log. It never sends robot commands,
and it keeps `mock`, `sim`, `replay`, and `cell` sources visibly distinct.

## Metrics

- Availability = PackML `Execute` seconds / elapsed planned-production seconds.
- Performance = sum of measured ideal cycle seconds for completed products / PackML `Execute` seconds.
- Quality = good products / completed products.
- OEE = Availability x Performance x Quality.
- First-pass yield = `good_first_pass` / completed products.
- Final yield = (`good_first_pass` + `good_after_rebuild`) / completed products.
- Builds per assist = good products / recorded assists.

Missing inputs produce an unavailable metric and an explanation. Performance therefore requires a measured ideal cycle
time for every product represented in the reporting window. Values are not capped or filled with targets.

## Run-log format

Each line has exactly `version`, `source`, `event`, `timestamp`, `run_id`, `rollout`, and `payload`. Supported sources
are `mock`, `sim`, `replay`, and `cell`; one log cannot mix sources or run IDs. Rollout identity lets one log retain
multiple controller histories even though the official controller creates a new state machine on every reset.

```json
{"version":1,"source":"mock","event":"packml","timestamp":"2026-10-29T15:00:00Z","run_id":"run-001","rollout":0,"payload":{"state":"Execute","reason":null,"product":1}}
{"version":1,"source":"mock","event":"product_result","timestamp":"2026-10-29T15:01:00Z","run_id":"run-001","rollout":0,"payload":{"product":1,"result":"good_first_pass","assists":0,"cycle_time_s":60.0}}
{"version":1,"source":"mock","event":"escape","timestamp":"2026-10-29T15:01:01Z","run_id":"run-001","rollout":0,"payload":{"product":1,"detail":"fault-injection evaluation escape"}}
{"version":1,"source":"mock","event":"heartbeat","timestamp":"2026-10-29T15:01:05Z","run_id":"run-001","rollout":0,"payload":{}}
```

The first PackML event must be at or before the planned window start for Availability to be available. Product results
use the four values in `contracts/birth_certificate.schema.json`. An escape is an explicit evaluation observation; the
dashboard never infers one from a rejected product.

## Record a live run

`ControllerRunLogRecorder` is the producer boundary between the controller and the dashboard. It synchronizes public
`PackMLStateMachine.history`, validates terminal records with the birth certificate module's public validator plus its semantic result rules, and
records an explicit evaluation escape. The writer is local, append-only, file-locked, bounded to 16 MiB by default,
source/run-pure, timestamp-ordered,
and synced to disk before returning. A `(run_id, rollout, product)` terminal identity can be recorded only once and
must follow the matching PackML terminal state. Good outcomes require the non-null base, four unique recipe steps,
passing seated checks, positive attempts, and both final checks. First-pass records cannot contain recovery evidence;
rebuilt records must contain it. Failed records require `Aborted` (`Stopped` is not terminal failure evidence), while
blocked records require the matching held/Andon lifecycle. The OEE API independently rejects duplicate terminal
identities and requires the latest earlier matching lifecycle, including for direct callers that bypass the JSONL
writer. A good result cannot reuse an earlier `Complete` after a later abort. Every passed step also requires embedded
inspection identity and passing `checks_agree`, `verdict`, and `action` evidence.

```python
from pathlib import Path

from industry.oee import EventSource
from industry.run_log import ControllerRunLogRecorder, RunLogWriter

writer = RunLogWriter(Path("eval/runs/run-001.jsonl"))
rollout = 0
recorder = ControllerRunLogRecorder(writer, EventSource.CELL, "run-001", rollout)
recorder.sync_packml(controller.state_machine)
recorder.record_product_result(validated_birth_certificate, recorded_at=finished_at)
recorder.record_heartbeat(recorded_at=observed_at)
```

Create a new recorder with the next rollout number after every actual `controller.reset()`. Call `sync_packml` after
each state transition and once during startup. On restart it verifies that PackML events for that rollout are an exact
prefix of the controller history before appending anything; a divergent history fails closed. Schedule
`record_heartbeat` periodically while the controller process is healthy, at an interval shorter than `--stale-after-s`.
A heartbeat is liveness evidence only; it never changes PackML state or any OEE numerator. Append or initial fsync
failure rolls back to the prior byte length and syncs that rollback; if the rollback sync also fails, the writer reports
that durability is uncertain rather than claiming success.

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
