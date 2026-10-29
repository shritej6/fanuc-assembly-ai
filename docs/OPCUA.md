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
- Accepted requests enter a bounded queue. They do not call `policy.act()`, a robot adapter, or the official FANUC
  client. The controller integration must consume and revalidate the request before changing PackML state.
- There are no joint, TCP, gripper, or arbitrary motion fields in the command interface.

## Address space

All application nodes use namespace URI `urn:team-v:fanuc:assembly-cell` and stable string node IDs under
`TeamV.Cell`.

| Group | Tags |
|---|---|
| `PackML` | `State`, `StateTimestamp`, `CurrentProduct`, `CurrentStep`, `AlarmActive`, `ActiveFault` |
| `Counts` | `Good`, `Reject`, `Assist` |
| `Inventory` | `Total`, one count per T09 inventory zone, and deterministic `ItemsJson` |
| `Commands` | `PendingCount`, `LastAcceptedId`, `LastAcceptedName`, `LastError`, and `SubmitCommand` |
| `Health` | `Healthy`, `LastPublishError`, `SnapshotSequence` |

`CurrentProduct` is `0` when no product owns the cell. `SnapshotSequence` is a seqlock: odd means an update is in
progress, and clients should accept a snapshot only when the same even value is observed before and after reading.

## Embedding in the controller process

```python
import asyncio

from robot.opcua import OpcUaCellServer


async def serve(controller) -> None:
    stop_event = asyncio.Event()
    gateway = OpcUaCellServer(controller)
    await gateway.run(stop_event)
```

The `controller` object must expose the public T09 interfaces described by
`robot.opcua_model.ControllerStatusSource`. Tests use deterministic model stubs and a loopback OPC UA client; they do
not connect to the real cell.

Run the focused verification with:

```shell
uv run pytest tests/test_opcua_model.py tests/test_opcua_server.py -q
```
