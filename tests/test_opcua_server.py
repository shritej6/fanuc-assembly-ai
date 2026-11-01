"""Loopback integration tests for the asyncua PLC-style address space.
Authored by Shritej Gunda.
"""

import asyncio
import json
import socket

import pytest
from asyncua import Client
from asyncua.ua.uaerrors import UaStatusCodeError

from assembly.state_machine import PackMLState
from robot.opcua import (
    OpcUaCellServer,
    OpcUaConfigurationError,
    OpcUaPublishError,
    OpcUaServerConfig,
    OpcUaServerError,
)
from robot.opcua_model import ControllerSnapshotMailbox
from tests.test_opcua_model import _executing_source


def _free_loopback_endpoint() -> str:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    return f"opc.tcp://127.0.0.1:{port}/fanuc/"


def _node(client: Client, index: int, path: str):
    return client.get_node(f"ns={index};s={OpcUaCellServer.string_node_id(path)}")


def test_asyncua_server_publishes_read_only_tags_and_queues_validated_command() -> None:
    asyncio.run(_exercise_server())


async def _exercise_server() -> None:
    source = _executing_source()
    config = OpcUaServerConfig(endpoint=_free_loopback_endpoint())
    gateway = OpcUaCellServer(ControllerSnapshotMailbox.from_source(source), config=config)
    await gateway.start()
    try:
        await gateway.publish_once()
        async with Client(config.endpoint) as client:
            index = await client.get_namespace_index(config.namespace_uri)
            await _assert_initial_snapshot(client, index)
            response = await _submit(client, index, "Hold", "plc-1", "operator camera review")
            assert json.loads(response) == {"command_id": "plc-1", "error": "", "queued": True}
            assert "accepted" not in json.loads(response)
            assert source.state_machine.state is PackMLState.EXECUTE
            assert await _node(client, index, "Commands.PendingCount").read_value() == 1
            assert await _node(client, index, "Commands.LastQueuedName").read_value() == "Hold"
            rejected = await _submit(client, index, "MoveJoints", "plc-2", "raw robot request")
            assert json.loads(rejected)["queued"] is False
            assert await _node(client, index, "Commands.PendingCount").read_value() == 1
            assert gateway.commands.pop(state=PackMLState.EXECUTE) is not None
            await gateway.publish_once()
            assert await _node(client, index, "Commands.PendingCount").read_value() == 0
            await _assert_state_update_and_read_only(client, index, gateway, source)
    finally:
        await gateway.stop()


async def _assert_initial_snapshot(client: Client, index: int) -> None:
    assert await _node(client, index, "PackML.State").read_value() == "Execute"
    assert await _node(client, index, "PackML.CurrentProduct").read_value() == 4
    assert await _node(client, index, "PackML.CurrentStep").read_value() == "product-4-step-01-attempt-1"
    assert await _node(client, index, "Counts.Good").read_value() == 2
    assert await _node(client, index, "Counts.Reject").read_value() == 1
    assert await _node(client, index, "Counts.Assist").read_value() == 2
    assert await _node(client, index, "Inventory.Table").read_value() == 1
    assert await _node(client, index, "Inventory.Spare").read_value() == 1
    assert await _node(client, index, "Inventory.Reject").read_value() == 1
    assert await _node(client, index, "Health.Healthy").read_value() is True
    assert await _node(client, index, "Health.SnapshotSequence").read_value() == 2


async def _submit(client: Client, index: int, name: str, command_id: str, reason: str) -> str:
    commands = _node(client, index, "Commands")
    method = _node(client, index, "Commands.SubmitCommand")
    return await commands.call_method(method, name, command_id, reason)


async def _assert_state_update_and_read_only(client: Client, index: int, gateway, source) -> None:
    source.state_machine.hold("camera evidence disagrees")
    gateway.snapshots.update_from(source)
    await gateway.publish_once()
    assert await _node(client, index, "PackML.State").read_value() == "Held"
    assert await _node(client, index, "PackML.AlarmActive").read_value() is True
    assert await _node(client, index, "PackML.ActiveFault").read_value() == "camera evidence disagrees"
    assert await _node(client, index, "Health.SnapshotSequence").read_value() == 6
    with pytest.raises(UaStatusCodeError):
        await _node(client, index, "PackML.State").write_value("Idle")


@pytest.mark.parametrize(
    "endpoint",
    [
        "opc.tcp://0.0.0.0:4840/fanuc/",
        "opc.tcp://192.0.2.10:4840/fanuc/",
        "opc.tcp://user:secret@127.0.0.1:4840/fanuc/",
        "http://127.0.0.1:4840/fanuc/",
        "opc.tcp://127.0.0.1/fanuc/",
    ],
)
def test_configuration_rejects_remote_insecure_or_malformed_endpoints(endpoint: str) -> None:
    with pytest.raises(OpcUaConfigurationError):
        OpcUaServerConfig(endpoint=endpoint)


def test_publish_requires_initialized_address_space() -> None:
    mailbox = ControllerSnapshotMailbox.from_source(_executing_source())
    gateway = OpcUaCellServer(mailbox, config=OpcUaServerConfig(endpoint=_free_loopback_endpoint()))

    with pytest.raises(OpcUaServerError, match="initialize"):
        asyncio.run(gateway.publish_once())


def test_seqlock_recovers_even_parity_after_partial_publish_failure() -> None:
    asyncio.run(_exercise_publish_recovery())


class _FailingNode:
    async def write_value(self, _value, _variant_type=None) -> None:
        raise RuntimeError("injected node write failure")


async def _exercise_publish_recovery() -> None:
    gateway = OpcUaCellServer(
        ControllerSnapshotMailbox.from_source(_executing_source()),
        config=OpcUaServerConfig(endpoint=_free_loopback_endpoint()),
    )
    await gateway.initialize()
    state_node = gateway._nodes["PackML.State"]
    gateway._nodes["PackML.State"] = _FailingNode()  # type: ignore[assignment]
    with pytest.raises(OpcUaPublishError, match="injected node write failure"):
        await gateway.publish_once()
    assert await gateway._nodes["Health.SnapshotSequence"].read_value() == 1

    gateway._nodes["PackML.State"] = state_node
    await gateway.publish_once()
    sequence = await gateway._nodes["Health.SnapshotSequence"].read_value()
    assert sequence == 4
    assert sequence % 2 == 0
