"""Software-only proof of the CELL policy boundary against the official mock."""

from __future__ import annotations

import re
import threading
import time
from contextlib import AbstractContextManager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

import numpy as np
import pytest
import zmq

from assembly.catalog import TRUSTED_GRASP_SOURCE, load_catalog
from assembly.cell_runtime import CellTransport
from assembly.state_machine import PackMLState
from policy_learn.deploy.attestation import REQUIRED_CELL_SKILLS
from policy_learn.deploy.groot_policy import DT_S
from policy_learn.deploy.guards import GuardLimits
from policy_learn.groot.groot_client import ATTESTATION_PROTOCOL, pack, unpack
from policy_learn.groot.tcp_convert import eef9d_to_tcp, tcp_to_eef9d
from robot.cell.policy import Policy
from robot.cell.t09_mock_runtime import MockPerception

CELL_DIR = Path(__file__).resolve().parents[2] / "robot" / "cell"
INITIAL_TCP = [400.0, 0.0, -200.0, -180.0, 0.0, 0.0]
FAKE_GUARD_LIMITS = GuardLimits()


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


class _SafeFakeGrootServer(_FakeGrootServer):
    """Semantic fake that proposes mock-calibrated targets through real ZMQ."""

    @staticmethod
    def _hold_action(request: dict[str, Any]) -> list[object]:
        observation, language = _validated_wire_request(request)
        match = re.search(r"\bM([1-6])_0([1-4])\b", language)
        if match is None:
            raise ValueError(f"unsupported semantic instruction: {language!r}")
        product, step_number = (int(value) for value in match.groups())
        recipe_step = load_catalog().product(product).steps[step_number - 1]
        turn = recipe_step.preferred_grasp.turn_deg
        grasp = np.array(
            [
                210.0 + 30.0 * product,
                -160.0 + 35.0 * step_number,
                -300.0,
                -180.0,
                0.0,
                (10.0 * step_number + turn) % 360.0,
            ]
        )
        insert = np.array(
            [
                360.0 + 20.0 * product,
                80.0,
                -300.0 + step_number,
                -180.0,
                0.0,
                (90.0 + turn) % 360.0,
            ]
        )
        state = observation["state"]
        current_eef = np.asarray(state["eef_9d"], dtype=float)[0, 0]
        current = eef9d_to_tcp(current_eef[None], np.array([-180.0, 0.0, 0.0]))[0]
        gripper = float(np.asarray(state["gripper"], dtype=float)[0, 0, 0])
        desired, target_gripper = _next_semantic_target(
            current,
            gripper,
            grasp,
            insert,
        )
        target = (
            current
            if target_gripper != gripper
            else _bounded_semantic_step(current, desired, insert=insert)
        )
        action = {
            "eef_9d": tcp_to_eef9d(target[None])[None],
            "gripper": np.asarray([[[target_gripper]]], dtype=np.float32),
        }
        return [action, {"fake": "deterministic-in-limit-semantic-row"}]


def _validated_wire_request(request: dict[str, Any]) -> tuple[dict[str, Any], str]:
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
    if np.asarray(state["eef_9d"]).shape != (1, 1, 9):
        raise ValueError("unexpected eef_9d tensor shape")
    if np.asarray(state["gripper"]).shape != (1, 1, 1):
        raise ValueError("unexpected gripper tensor shape")
    if np.asarray(state["joints"]).shape != (1, 1, 6):
        raise ValueError("unexpected joints tensor shape")
    language = observation["language"]["annotation.human.task_description"]
    if not isinstance(language, list) or not language or not language[0][0].strip():
        raise ValueError("semantic instruction is required")
    return observation, language[0][0]


def _next_semantic_target(
    current: np.ndarray,
    gripper: float,
    grasp: np.ndarray,
    insert: np.ndarray,
) -> tuple[np.ndarray, float]:
    at_grasp = _same_tcp(current, grasp)
    at_insert = _same_tcp(current, insert)
    if gripper == 0.0 and _on_release_column(current, insert):
        retreat = insert.copy()
        retreat[2] += 10.0
        return retreat, 0.0
    if gripper == 0.0:
        return (grasp, 1.0) if at_grasp else (grasp, 0.0)
    return (insert, 0.0) if at_insert else (insert, 1.0)


def _same_tcp(left: np.ndarray, right: np.ndarray) -> bool:
    position_error = float(np.linalg.norm(left[:3] - right[:3]))
    angle_error = float(np.abs((left[3:6] - right[3:6] + 180.0) % 360.0 - 180.0).max())
    return position_error <= 0.3 and angle_error <= 1.0


def _on_release_column(current: np.ndarray, insert: np.ndarray) -> bool:
    xy_error = float(np.linalg.norm(current[:2] - insert[:2]))
    angle_error = float(np.abs((current[3:6] - insert[3:6] + 180.0) % 360.0 - 180.0).max())
    z_offset = float(current[2] - insert[2])
    return xy_error <= 0.3 and angle_error <= 1.0 and 0.0 <= z_offset <= 10.0


def _bounded_semantic_step(
    current: np.ndarray,
    target: np.ndarray,
    *,
    insert: np.ndarray,
) -> np.ndarray:
    free_candidate = _bounded_step(
        current,
        target,
        linear_step=0.98 * FAKE_GUARD_LIMITS.max_speed_mm_s * DT_S,
        angular_step=0.98 * FAKE_GUARD_LIMITS.max_rot_deg_s * DT_S,
    )
    if _in_guard_slow_zone(current, insert) or _in_guard_slow_zone(free_candidate, insert):
        candidate = _bounded_step(
            current,
            target,
            linear_step=0.95 * FAKE_GUARD_LIMITS.slow_speed_mm_s * DT_S,
            angular_step=0.95 * FAKE_GUARD_LIMITS.slow_rot_deg_s * DT_S,
        )
    else:
        candidate = free_candidate
    return target.copy() if _same_tcp(candidate, target) else candidate


def _bounded_step(
    current: np.ndarray,
    target: np.ndarray,
    *,
    linear_step: float,
    angular_step: float,
) -> np.ndarray:
    delta = target - current
    delta[3:6] = (delta[3:6] + 180.0) % 360.0 - 180.0
    ratio = max(
        float(np.linalg.norm(delta[:3])) / linear_step,
        float(np.abs(delta[3:6]).max()) / angular_step,
        1.0,
    )
    return current + delta / ratio


def _in_guard_slow_zone(pose: np.ndarray, insert: np.ndarray) -> bool:
    at_contact_xy = float(np.linalg.norm(pose[:2] - insert[:2])) <= 1e-9
    near_contact_z = pose[2] <= insert[2] + FAKE_GUARD_LIMITS.slow_zone_mm
    near_floor = pose[2] <= FAKE_GUARD_LIMITS.z_mm[0] + FAKE_GUARD_LIMITS.slow_zone_mm
    return near_floor or (at_contact_xy and near_contact_z)


class _MockCellServer(AbstractContextManager["_MockCellServer"]):
    def __init__(self, *, trial_s: float = 1.0) -> None:
        from robot.cell.mock_cell import Mock, handler

        self.mock = Mock(1, trial_s, "mock", "assemble product 1", quiet=True)
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
    with _FakeGrootServer() as groot, _MockCellServer(trial_s=15.0) as cell:
        policy = Policy(
            runtime="cell",
            checkpoint=groot.url,
            cell_transport=CellTransport.OFFICIAL_MOCK,
        )
        try:
            trial = cell.ready_trial()
            monitor = threading.Thread(
                target=_end_trial_on_controller_terminal,
                args=(cell, policy),
                daemon=True,
            )
            monitor.start()
            run_trial(cell.client, policy, trial)
            monitor.join(timeout=2.0)

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
        finally:
            policy.controller.groot_executor.close()


def test_cell_policy_network_path_completes_pick_insert_phase_with_guard_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive proof must cross Policy, ZMQ, guards, CellClient, and mock_cell."""
    from robot.cell.cell_client import run_trial

    monkeypatch.setenv(
        "FANUC_T09_CELL_FACTORY",
        "robot.cell.t09_mock_runtime:build_groot_mock_controller",
    )
    monkeypatch.setenv("FANUC_T09_SOFTWARE_PROOF", "1")
    monkeypatch.syspath_prepend(str(CELL_DIR))
    original_configure = MockPerception._configure_from_prompt

    def configure_nominal(self: MockPerception, observation: dict) -> None:
        original_configure(self, observation)
        self.fallen_once.clear()
        self.failed_insertions.clear()

    monkeypatch.setattr(MockPerception, "_configure_from_prompt", configure_nominal)
    with _SafeFakeGrootServer() as groot, _MockCellServer(trial_s=90.0) as cell:
        policy = Policy(
            runtime="cell",
            checkpoint=groot.url,
            cell_transport=CellTransport.OFFICIAL_MOCK,
        )
        try:
            trial = cell.ready_trial()
            monitor = threading.Thread(
                target=_end_trial_on_controller_terminal,
                args=(cell, policy),
                daemon=True,
            )
            monitor.start()
            run_trial(cell.client, policy, trial)
            monitor.join(timeout=2.0)

            endpoints = [request["endpoint"] for request in groot.requests]
            completed = _completed_steps(policy)
            reason = _held_reason(policy)
            guard_log = policy.controller.groot_executor._policy.guard_log
            print(
                f"CELL_PROOF outcome={'phase_complete' if completed else 'hold_andon'} "
                f"completed_steps={completed} reason={reason!r} "
                f"get_action={endpoints.count('get_action')} rejections={cell.mock.rejections}"
            )

            assert completed
            assert reason is None
            assert endpoints.count("get_action") > 0
            assert "reset" in endpoints
            assert cell.mock.rejections == []
            assert all(not entry["held"] for entry in guard_log)
        finally:
            policy.controller.groot_executor.close()


def _end_trial_on_controller_terminal(cell: _MockCellServer, policy: Policy) -> None:
    deadline = time.monotonic() + 75.0
    while time.monotonic() < deadline:
        if _completed_steps(policy) or _held_reason(policy) is not None:
            with cell.mock.lock:
                cell.mock.queue.clear()
                cell.mock.trials[0]["status"] = "done"
                cell.mock.state = "homing"
            return
        time.sleep(0.01)


def _completed_steps(policy: Policy) -> tuple[str, ...]:
    return tuple(
        event.object_id
        for event in policy.controller.events
        if event.kind == "installed" and event.object_id is not None
    )
