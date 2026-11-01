"""Protocol-boundary tests that do not require pandas or a live ZMQ server.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_learn.groot.attested_server import ATTESTATION_PROTOCOL, AttestedProxy
from policy_learn.groot.groot_client import GrootTcpClient, PolicyServerError, groot_to_rows, pack, unpack
from policy_learn.groot.tcp_convert import tcp_to_eef9d


class _FakeUpstream:
    def __init__(self, reply: bytes) -> None:
        self.reply = reply
        self.sent: list[bytes] = []
        self.recv_count = 0

    def send(self, request: bytes) -> None:
        self.sent.append(request)

    def recv(self) -> bytes:
        self.recv_count += 1
        return self.reply


def test_proxy_serves_attestation_locally_and_forwards_model_bytes_unchanged() -> None:
    upstream_reply = pack({"action": [1, 2, 3]})
    upstream = _FakeUpstream(upstream_reply)
    envelope = {
        "protocol": ATTESTATION_PROTOCOL,
        "attestation": {"checkpoint_sha256": "a" * 64},
    }
    proxy = AttestedProxy(upstream, envelope)

    assert unpack(proxy.handle(pack({"endpoint": "describe"}))) == envelope
    assert upstream.sent == []
    assert upstream.recv_count == 0

    inference_request = pack({"endpoint": "get_action", "data": {"opaque": b"payload"}})
    assert proxy.handle(inference_request) == upstream_reply
    assert upstream.sent == [inference_request]
    assert upstream.recv_count == 1


def test_client_describe_accepts_only_the_versioned_attestation_envelope() -> None:
    client = object.__new__(GrootTcpClient)
    valid = {
        "protocol": ATTESTATION_PROTOCOL,
        "attestation": {"checkpoint_sha256": "a" * 64},
    }
    client.call = lambda endpoint: valid

    assert client.describe() == valid["attestation"]

    invalid = (
        {},
        {"protocol": "wrong", "attestation": {}},
        {"protocol": ATTESTATION_PROTOCOL, "attestation": []},
    )
    for reply in invalid:
        client.call = lambda endpoint, reply=reply: reply
        with pytest.raises(PolicyServerError, match="versioned attestation"):
            client.describe()


def test_groot_rows_preserve_ambiguous_or_nonfinite_gripper_for_the_guard() -> None:
    current_tcp = np.array([400.0, 0.0, -300.0, -180.0, 0.0, 0.0])
    tcp_rows = np.tile(current_tcp, (3, 1))
    action = {
        "eef_9d": tcp_to_eef9d(tcp_rows)[None],
        "gripper": np.array([0.49, np.nan, 0.51])[None, :, None],
    }

    rows = np.asarray(groot_to_rows(action, current_tcp))

    assert rows[0, 6] == pytest.approx(0.49)
    assert np.isnan(rows[1, 6])
    assert rows[2, 6] == pytest.approx(0.51)
