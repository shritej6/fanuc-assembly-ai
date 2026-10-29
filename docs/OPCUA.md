# OPC UA integration tags (T19)

`robot.opcua.OpcUaCellServer` publishes the T09 controller's public state through `asyncua`. It is an
integration-ready PLC/HMI boundary, not a direct Rockwell integration and not a second robot-control path.

## Safety boundary

- The default endpoint is loopback-only: `opc.tcp://127.0.0.1:4840/fanuc/`.
- Non-loopback endpoints are rejected. A remote deployment needs certificates, authentication, network policy, and
  a separate security review.
- Telemetry variables are read-only to OPC UA clients.
- `Commands.SubmitCommand` accepts only `Hold`, `Resume`, `Stop`, `Abort`, and `Reset`, with a unique command ID and
  operator reason. The request must be valid for the live PackML state.
- Valid requests are reported as **queued**, never accepted or executed. They enter a bounded, replay-protected,
  thread-safe queue and expire after the configured pending TTL (30 seconds by default). Expiry does not release the
  command ID for replay.
- This branch deliberately provides no controller consumer. A future controller-owned consumer must call `pop()`,
  revalidate the live PackML state, apply the non-motion intent, and publish its own acknowledgement. Until that
  integration exists, a queued command cannot change the controller or robot.
- There are no joint, TCP, gripper, or arbitrary motion fields in the command interface.

## Address space

All application nodes use namespace URI `urn:team-v:fanuc:assembly-cell` and stable string node IDs under
`TeamV.Cell`.

| Group | Tags |
|---|---|
| `PackML` | `State`, `StateTimestamp`, `CurrentProduct`, `CurrentStep`, `AlarmActive`, `ActiveFault` |
| `Counts` | `Good`, `Reject`, `Assist` |
| `Inventory` | `Total`, one count per T09 inventory zone, and deterministic `ItemsJson` |
| `Commands` | `PendingCount`, `LastQueuedId`, `LastQueuedName`, `LastError`, and `SubmitCommand` |
| `Health` | `Healthy`, `LastPublishError`, `SnapshotSequence` |

`CurrentProduct` is `0` when no product owns the cell. `SnapshotSequence` is a seqlock: odd means an update is in
progress, and clients should accept a snapshot only when the same even value is observed before and after reading.

## Thread ownership and embedding

The synchronous controller owns all mutable T09 state. It creates and updates a
`ControllerSnapshotMailbox` from its own thread after a controller cycle. The async OPC UA task receives only the
mailbox and reads immutable `ControllerSnapshot` values under a lock; it never reads the live controller across
threads. `SafeCommandBoundary` separately protects its pending queue because the OPC UA task submits while a future
controller-owned consumer pops.

```python
import asyncio

from robot.opcua import OpcUaCellServer
from robot.opcua_model import ControllerSnapshotMailbox

# Run these two lines while the controller-owning thread has a coherent state.
snapshots = ControllerSnapshotMailbox.from_source(controller)
gateway = OpcUaCellServer(snapshots)


def after_controller_cycle() -> None:
    snapshots.update_from(controller)


async def serve() -> None:
    stop_event = asyncio.Event()
    await gateway.run(stop_event)
```

The controller object used to update the mailbox must expose the public T09 interfaces described by
`robot.opcua_model.ControllerStatusSource`. A snapshot-build or OPC UA write failure marks health false and stops
`run()`; a process supervisor may restart the loop only after resolving the fault. Tests use deterministic model stubs
and a loopback OPC UA client; they do not connect to the real cell.

Run the focused verification with:

```shell
uv run pytest tests/test_opcua_model.py tests/test_opcua_server.py -q
```
