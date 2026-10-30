"""Software-only proof of the CELL policy boundary against the official mock."""

from __future__ import annotations

import threading
import time
from contextlib import AbstractContextManager
from http.server import ThreadingHTTPServer
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest
import zmq

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.state_machine import PackMLState
from policy_learn.deploy.attestation import REQUIRED_CELL_SKILLS
from policy_learn.groot.groot_client import ATTESTATION_PROTOCOL, pack, unpack
from policy_learn.groot.tcp_convert import tcp_to_eef9d
from robot.cell.policy import Policy

CELL_DIR = Path(__file__).resolve().parents[2] / "robot" / "cell"
INITIAL_TCP = [400.0, 0.0, -200.0, -180.0, 0.0, 0.0]
HAS_CELL_CODEC = find_spec("cv2") is not None


class _FakeGrootServer(AbstractContextManager["_FakeGrootServer"]):
    """Small PolicyServer-compatible REP service returning one unsafe row."""

    def __init__(self) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.requests: list[dict[str, Any]] = []
        self.error: Exception | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def url(self) -> str:
        return f"groot://127.0.0.1:{self.port}"

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)
        self.socket.close(linger=0)
        self.context.term()
        if exc_type is None and self.error is not None:
            raise AssertionError("fake GR00T server failed") from self.error

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        try:
            while not self.stop.is_set():
                if not dict(poller.poll(timeout=50)).get(self.socket):
                    continue
                request = unpack(self.socket.recv())
                self.requests.append(request)
                self.socket.send(pack(self._reply(request)))
        except (KeyError, TypeError, ValueError, zmq.ZMQError) as error:  # pragma: no cover - surfaced by __exit__
            self.error = error

    def _reply(self, request: object) -> object:
        if not isinstance(request, dict) or not isinstance(request.get("endpoint"), str):
            return {"error": "request must contain a string endpoint"}
        endpoint = request["endpoint"]
        if endpoint == "ping":
            return {"status": "ok"}
        if endpoint == "describe":
            return _attestation_envelope()
        if endpoint == "reset":
            return {"status": "ok"}
        if endpoint == "get_action":
            try:
                return self._hold_action(request)
            except (KeyError, TypeError, ValueError) as error:
                return {"error": f"invalid get_action request: {error}"}
        return {"error": f"unknown endpoint: {endpoint}"}

    @staticmethod
    def _hold_action(request: dict[str, Any]) -> list[object]:
        data = request["data"]
        observation = data["observation"]
        if data.get("options") is not None:
            raise ValueError("options must be None")
        if set(observation["video"]) != {"table", "side", "wrist"}:
            raise ValueError("all three camera streams are required")
        for image in observation["video"].values():
            array = np.asarray(image)
            if array.shape != (1, 1, 360, 640, 3) or array.dtype != np.uint8:
                raise ValueError(f"unexpected video tensor {array.shape} {array.dtype}")
        state = observation["state"]
        eef = np.asarray(state["eef_9d"])
        gripper = np.asarray(state["gripper"])
        joints = np.asarray(state["joints"])
        if eef.shape != (1, 1, 9) or gripper.shape != (1, 1, 1) or joints.shape != (1, 1, 6):
            raise ValueError("unexpected state tensor shape")
        language = observation["language"]["annotation.human.task_description"]
        if not isinstance(language, list) or not language or not language[0][0].strip():
            raise ValueError("semantic instruction is required")
        unsafe_tcp = np.array([[700.0, 0.0, -200.0, -180.0, 0.0, 0.0]])
        unsafe_eef = tcp_to_eef9d(unsafe_tcp)[None]
        action = {"eef_9d": unsafe_eef, "gripper": gripper.copy()}
        return [action, {"fake": "deterministic-out-of-workspace-row"}]


class _MockCellServer(AbstractContextManager["_MockCellServer"]):
    def __init__(self) -> None:
        from robot.cell.mock_cell import Mock, handler

        self.mock = Mock(1, 1.0, "mock", "assemble product 1", quiet=True)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler(self.mock))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)

    @property
    def client(self) -> Any:
        from robot.cell.cell_client import CellClient

        host, port = self.server.server_address
        return CellClient(f"http://{host}:{port}", "mock")

    def ready_trial(self) -> dict[str, Any]:
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            trials = self.client.trials()
            if trials[0]["status"] == "ready":
                with self.mock.lock:
                    self.mock.tcp = list(INITIAL_TCP)
                return trials[0]
            time.sleep(0.01)
        raise AssertionError("official mock trial did not become ready")


def _attestation_envelope() -> dict[str, object]:
    return {
        "protocol": ATTESTATION_PROTOCOL,
        "attestation": {
            "checkpoint_sha256": "a" * 64,
            "model_id": "GR00T-N1.7",
            "model_revision": "fake-e2e",
            "dataset_manifest_sha256": "b" * 64,
            "catalog_sha256": "c" * 64,
            "grasp_source_sha256": TRUSTED_GRASP_SOURCE.sha256,
            "guard_calibration_sha256": "d" * 64,
            "skills": sorted(REQUIRED_CELL_SKILLS),
            "cameras": ["table", "side", "wrist"],
            "depth": [],
            "image_size": [640, 360],
            "dt_s": 0.2,
            "execution_horizon": 1,
            "action": "tcp",
            "embodiment": "fanuc_er4ia_team_v_fingers",
            "release_protocol": "t09_verified_release_v1",
            "eval_approved": True,
            "server_verified": True,
        },
    }


def _held_reason(policy: Policy) -> str | None:
    record = policy.controller.state_machine.current_record
    if record.state is PackMLState.HELD and isinstance(record.reason, str) and record.reason.strip():
        return record.reason
    return None


@pytest.mark.skipif(not HAS_CELL_CODEC, reason="requires the robot/cell image-codec dependency")
def test_cell_policy_uses_fake_groot_against_official_mock_and_reaches_a_terminal_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken CELL factory, bypassed server, unsafe row, or silent non-progress must fail."""
    from robot.cell.cell_client import run_trial

    monkeypatch.setenv(
        "FANUC_T09_CELL_FACTORY",
        "robot.cell.t09_mock_runtime:build_groot_mock_controller",
    )
    monkeypatch.setenv("FANUC_T09_SOFTWARE_PROOF", "1")
    monkeypatch.syspath_prepend(str(CELL_DIR))
    with _FakeGrootServer() as groot, _MockCellServer() as cell:
        policy = Policy(runtime="cell", checkpoint=groot.url)
        run_trial(cell.client, policy, cell.ready_trial())

        endpoints = [request["endpoint"] for request in groot.requests]
        completed = tuple(policy.controller.completed_products)
        reason = _held_reason(policy)
        outcome = "DONE" if 1 in completed else "hold_andon"
        print(
            f"CELL_PROOF outcome={outcome} completed={completed} "
            f"reason={reason!r} get_action={endpoints.count('get_action')} "
            f"rejections={cell.mock.rejections}"
        )

        assert "get_action" in endpoints, endpoints
        assert "reset" in endpoints, endpoints
        assert cell.mock.rejections == []
        assert 1 in completed or reason, (
            f"product 1 neither completed nor entered an explained Held state; "
            f"completed={completed}, state={policy.controller.state_machine.current_record}"
        )
        if reason is not None:
            assert "GR00T guard trip" in reason
            assert "outside" in reason
