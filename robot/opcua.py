"""Loopback-only asyncua publisher for PLC-style FANUC cell telemetry.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
from dataclasses import dataclass
from numbers import Number
from urllib.parse import urlsplit

from asyncua import Server, ua
from asyncua.common.methods import uamethod
from asyncua.common.node import Node

from robot.opcua_model import (
    CommandValidationError,
    ControllerSnapshot,
    ControllerSnapshotMailbox,
    SafeCommandBoundary,
)


class OpcUaServerError(RuntimeError):
    """Base class for local integration-server failures."""


class OpcUaConfigurationError(OpcUaServerError):
    """The server configuration would expose an unsafe endpoint."""


class OpcUaPublishError(OpcUaServerError):
    """A coherent controller snapshot could not be published."""


@dataclass(frozen=True, slots=True)
class OpcUaServerConfig:
    """A local integration endpoint; remote deployment requires a security review."""

    endpoint: str = "opc.tcp://127.0.0.1:4840/fanuc/"
    namespace_uri: str = "urn:team-v:fanuc:assembly-cell"
    publish_period_s: float = 0.25

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint)
        valid_uri = (
            isinstance(self.namespace_uri, str)
            and 4 < len(self.namespace_uri) <= 256
            and self.namespace_uri.startswith("urn:")
            and not any(character.isspace() for character in self.namespace_uri)
        )
        if not valid_uri:
            raise OpcUaConfigurationError("namespace_uri must be a non-empty URN")
        period = self.publish_period_s
        if isinstance(period, bool) or not isinstance(period, Number) or not math.isfinite(float(period)):
            raise OpcUaConfigurationError("publish_period_s must be finite")
        if not 0.05 <= float(period) <= 60.0:
            raise OpcUaConfigurationError("publish_period_s must be between 0.05 and 60 seconds")


def _validate_endpoint(endpoint: object) -> None:
    if not isinstance(endpoint, str):
        raise OpcUaConfigurationError("endpoint must be an opc.tcp URL")
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError as error:
        raise OpcUaConfigurationError("endpoint contains an invalid port") from error
    if parsed.scheme != "opc.tcp" or parsed.hostname is None or port is None:
        raise OpcUaConfigurationError("endpoint must include opc.tcp, a loopback host, and a port")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise OpcUaConfigurationError("endpoint must not contain credentials, query, or fragment")
    if not _is_loopback(parsed.hostname):
        raise OpcUaConfigurationError(
            "T19 permits loopback endpoints only; remote OPC UA needs certificates and review"
        )


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class OpcUaCellServer:
    """Publish controller state and queue validated non-motion requests through asyncua."""

    NODE_PREFIX = "TeamV.Cell"

    def __init__(
        self,
        snapshots: ControllerSnapshotMailbox,
        *,
        config: OpcUaServerConfig | None = None,
        commands: SafeCommandBoundary | None = None,
    ) -> None:
        if type(snapshots) is not ControllerSnapshotMailbox:
            raise OpcUaConfigurationError("OPC UA requires a controller-owned snapshot mailbox")
        self.snapshots = snapshots
        self.config = config or OpcUaServerConfig()
        self.commands = commands or SafeCommandBoundary()
        self.server = Server()
        self._nodes: dict[str, Node] = {}
        self._initialized = False
        self._started = False
        self._sequence = 0

    @classmethod
    def string_node_id(cls, path: str) -> str:
        return f"{cls.NODE_PREFIX}.{path}"

    async def initialize(self) -> None:
        if self._initialized:
            return
        snapshot = self.snapshots.read()
        await self.server.init()
        self.server.set_endpoint(self.config.endpoint)
        self.server.set_server_name("Team V FANUC Assembly Cell")
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        self.server.set_identity_tokens([ua.AnonymousIdentityToken])
        index = await self.server.register_namespace(self.config.namespace_uri)
        await self._create_address_space(index, snapshot)
        self._initialized = True

    async def start(self) -> None:
        await self.initialize()
        if not self._started:
            await self.server.start()
            self._started = True

    async def stop(self) -> None:
        if self._started:
            await self.server.stop()
            self._started = False

    async def run(self, stop_event: asyncio.Event) -> None:
        """Publish until explicitly stopped; any publish failure stops the endpoint."""
        await self.start()
        try:
            while not stop_event.is_set():
                await self.publish_once()
                await _wait_or_timeout(stop_event, self.config.publish_period_s)
        finally:
            await self.stop()

    async def publish_once(self) -> ControllerSnapshot:
        self._require_initialized()
        try:
            snapshot = self.snapshots.read()
        except Exception as error:
            await self._mark_unhealthy(error)
            raise OpcUaPublishError(f"controller snapshot failed: {error}") from error
        try:
            await self._write_snapshot(snapshot)
        except Exception as error:
            await self._mark_unhealthy(error)
            raise OpcUaPublishError(f"OPC UA snapshot write failed: {error}") from error
        return snapshot

    async def _create_address_space(self, index: int, snapshot: ControllerSnapshot) -> None:
        cell = await self.server.nodes.objects.add_object(
            ua.NodeId(self.NODE_PREFIX, index),
            ua.QualifiedName("Cell", index),
        )
        groups = await self._create_groups(cell, index)
        for path, value, variant_type in _tag_specs(snapshot, self.commands):
            group_name, browse_name = path.split(".", 1)
            await self._add_variable(groups[group_name], index, path, browse_name, value, variant_type)
        await self._add_submit_method(groups["Commands"], index)

    async def _create_groups(self, cell: Node, index: int) -> dict[str, Node]:
        groups: dict[str, Node] = {}
        for name in ("PackML", "Counts", "Inventory", "Commands", "Health"):
            groups[name] = await cell.add_object(
                ua.NodeId(self.string_node_id(name), index),
                ua.QualifiedName(name, index),
            )
        return groups

    async def _add_variable(
        self,
        parent: Node,
        index: int,
        path: str,
        browse_name: str,
        value: object,
        variant_type: ua.VariantType,
    ) -> None:
        node = await parent.add_variable(
            ua.NodeId(self.string_node_id(path), index),
            ua.QualifiedName(browse_name, index),
            value,
            variant_type,
        )
        self._nodes[path] = node

    async def _add_submit_method(self, commands: Node, index: int) -> None:
        @uamethod
        async def submit(_parent: object, name: str, command_id: str, reason: str) -> str:
            return await self._submit_command(name, command_id, reason)

        await commands.add_method(
            ua.NodeId(self.string_node_id("Commands.SubmitCommand"), index),
            ua.QualifiedName("SubmitCommand", index),
            submit,
            [ua.VariantType.String, ua.VariantType.String, ua.VariantType.String],
            [ua.VariantType.String],
        )

    async def _submit_command(self, name: str, command_id: str, reason: str) -> str:
        try:
            command = self.commands.submit(
                name,
                command_id,
                reason,
                state=self.snapshots.read().state,
            )
        except CommandValidationError as error:
            await self._write_command_status(error=str(error))
            return _command_response(False, command_id, error=str(error))
        await self._write_command_status(command_id=command.command_id, command_name=command.name.value)
        return _command_response(True, command.command_id)

    async def _write_command_status(
        self,
        *,
        command_id: str | None = None,
        command_name: str | None = None,
        error: str = "",
    ) -> None:
        await self._nodes["Commands.PendingCount"].write_value(len(self.commands.pending), ua.VariantType.UInt16)
        if command_id is not None and command_name is not None:
            await self._nodes["Commands.LastQueuedId"].write_value(command_id, ua.VariantType.String)
            await self._nodes["Commands.LastQueuedName"].write_value(command_name, ua.VariantType.String)
        await self._nodes["Commands.LastError"].write_value(error[:240], ua.VariantType.String)

    async def _write_snapshot(self, snapshot: ControllerSnapshot) -> None:
        self._sequence = _next_odd_sequence(self._sequence)
        await self._nodes["Health.SnapshotSequence"].write_value(self._sequence, ua.VariantType.UInt64)
        for path, value in _snapshot_values(snapshot).items():
            await self._nodes[path].write_value(value, _variant_type(path))
        await self._nodes["Commands.PendingCount"].write_value(len(self.commands.pending), ua.VariantType.UInt16)
        await self._nodes["Health.Healthy"].write_value(True, ua.VariantType.Boolean)
        await self._nodes["Health.LastPublishError"].write_value("", ua.VariantType.String)
        self._sequence += 1
        await self._nodes["Health.SnapshotSequence"].write_value(self._sequence, ua.VariantType.UInt64)

    async def _mark_unhealthy(self, error: Exception) -> None:
        if not self._initialized:
            return
        message = f"{type(error).__name__}: {error}"[:240]
        await self._nodes["Health.Healthy"].write_value(False, ua.VariantType.Boolean)
        await self._nodes["Health.LastPublishError"].write_value(message, ua.VariantType.String)

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise OpcUaServerError("initialize the OPC UA server before publishing")


async def _wait_or_timeout(stop_event: asyncio.Event, period_s: float) -> None:
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=period_s)
    except TimeoutError:
        return


def _tag_specs(
    snapshot: ControllerSnapshot,
    commands: SafeCommandBoundary,
) -> tuple[tuple[str, object, ua.VariantType], ...]:
    values = _snapshot_values(snapshot)
    specs = [(path, value, _variant_type(path)) for path, value in values.items()]
    specs.extend(
        [
            ("Commands.PendingCount", len(commands.pending), ua.VariantType.UInt16),
            ("Commands.LastQueuedId", "", ua.VariantType.String),
            ("Commands.LastQueuedName", "", ua.VariantType.String),
            ("Commands.LastError", "", ua.VariantType.String),
            ("Health.Healthy", True, ua.VariantType.Boolean),
            ("Health.LastPublishError", "", ua.VariantType.String),
            ("Health.SnapshotSequence", 0, ua.VariantType.UInt64),
        ]
    )
    return tuple(specs)


def _snapshot_values(snapshot: ControllerSnapshot) -> dict[str, object]:
    values: dict[str, object] = {
        "PackML.State": snapshot.state.value,
        "PackML.StateTimestamp": snapshot.state_timestamp,
        "PackML.CurrentProduct": snapshot.current_product or 0,
        "PackML.CurrentStep": snapshot.current_step,
        "PackML.AlarmActive": snapshot.alarm_active,
        "PackML.ActiveFault": snapshot.active_fault,
        "Counts.Good": snapshot.good_count,
        "Counts.Reject": snapshot.reject_count,
        "Counts.Assist": snapshot.assist_count,
        "Inventory.Total": sum(count for _, count in snapshot.inventory_counts),
        "Inventory.ItemsJson": snapshot.inventory_json,
    }
    for zone, count in snapshot.inventory_counts:
        values[f"Inventory.{zone.value.title()}"] = count
    return values


def _variant_type(path: str) -> ua.VariantType:
    if path == "PackML.CurrentProduct":
        return ua.VariantType.Int16
    if path == "PackML.AlarmActive":
        return ua.VariantType.Boolean
    if path.startswith(("Counts.", "Inventory.")) and path != "Inventory.ItemsJson":
        return ua.VariantType.UInt32
    return ua.VariantType.String


def _next_odd_sequence(current: int) -> int:
    return current + 1 if current % 2 == 0 else current + 2


def _command_response(queued: bool, command_id: object, *, error: str = "") -> str:
    safe_id = command_id if isinstance(command_id, str) and len(command_id) <= 64 else ""
    payload = {"queued": queued, "command_id": safe_id, "error": error[:240]}
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)
