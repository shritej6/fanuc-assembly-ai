"""A deterministic, PolicyServer-compatible stand-in for GR00T (rehearsal only).

It answers the same ZMQ endpoints as the real server (ping/describe/reset/get_action) and proposes rows that
would complete one pick-insert of the mock-calibrated M<p>_0<s> step: approach the grasp, close, carry to the
insert pose, open, retreat. Steps are sized to the live guard limits so the guard has nothing to refuse on the
nominal path; the slow zone uses the guard's *current* contact target, whatever radius Codex's envelope uses.

``inject_low_row`` makes it emit, once, a row 0.5 mm off-axis and ``low_row_below_floor_mm`` below the contact
floor while it carries the part over the insert target.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from typing import Any, Self

import numpy as np
import zmq

from assembly.catalog import TRUSTED_GRASP_SOURCE, load_catalog
from policy_learn.deploy.attestation import REQUIRED_CELL_SKILLS
from policy_learn.deploy.cell_groot_policy import DT_S
from policy_learn.deploy.cell_guards import GuardLimits
from policy_learn.groot.groot_client import ATTESTATION_PROTOCOL, pack, unpack
from policy_learn.groot.tcp_convert import eef9d_to_tcp, tcp_to_eef9d

LIMITS = GuardLimits()
DOWN = np.array([-180.0, 0.0, 0.0])
GuardTargetFn = Callable[[], Any]


def attestation_envelope() -> dict[str, object]:
    """The same software-proof attestation Codex's CELL_PROOF fake server returns."""
    return {
        "protocol": ATTESTATION_PROTOCOL,
        "attestation": {
            "checkpoint_sha256": "a" * 64, "model_id": "GR00T-N1.7", "model_revision": "rehearsal-scripted",
            "dataset_manifest_sha256": "b" * 64, "catalog_sha256": "c" * 64,
            "grasp_source_sha256": TRUSTED_GRASP_SOURCE.sha256, "guard_calibration_sha256": "d" * 64,
            "skills": sorted(REQUIRED_CELL_SKILLS), "cameras": ["table", "side", "wrist"], "depth": [],
            "image_size": [640, 360], "dt_s": 0.2, "execution_horizon": 1, "action": "tcp",
            "embodiment": "fanuc_er4ia_team_v_fingers", "release_protocol": "t09_verified_release_v1",
            "eval_approved": True, "server_verified": True,
        },
    }


def _same(left: np.ndarray, right: np.ndarray, tol_mm: float = 0.3) -> bool:
    angle = float(np.abs((left[3:6] - right[3:6] + 180.0) % 360.0 - 180.0).max())
    return float(np.linalg.norm(left[:3] - right[:3])) <= tol_mm and angle <= 1.0


def _poses(language: str) -> tuple[np.ndarray, np.ndarray]:
    match = re.search(r"\bM([1-6])_0([1-4])\b", language)
    if match is None:
        raise ValueError(f"unsupported semantic instruction: {language!r}")
    product, step = (int(v) for v in match.groups())
    turn = load_catalog().product(product).steps[step - 1].preferred_grasp.turn_deg
    grasp = np.array([210.0 + 30.0 * product, -160.0 + 35.0 * step, -300.0, -180.0, 0.0,
                      (10.0 * step + turn) % 360.0])
    insert = np.array([360.0 + 20.0 * product, 80.0, -300.0 + step, -180.0, 0.0, (90.0 + turn) % 360.0])
    return grasp, insert


def _in_slow_zone(pose: np.ndarray, target: Any) -> bool:
    if pose[2] <= LIMITS.z_mm[0] + LIMITS.slow_zone_mm + 1.0:
        return True
    if target is None:
        return False
    near = float(np.hypot(pose[0] - target.x_mm, pose[1] - target.y_mm)) <= target.radius_mm + 1.0
    return near and pose[2] <= target.z_contact_mm + LIMITS.slow_zone_mm + 1.0


def _step(current: np.ndarray, goal: np.ndarray, linear: float, angular: float) -> np.ndarray:
    delta = goal - current
    delta[3:6] = (delta[3:6] + 180.0) % 360.0 - 180.0
    ratio = max(float(np.linalg.norm(delta[:3])) / linear, float(np.abs(delta[3:6]).max()) / angular, 1.0)
    return current + delta / ratio


def bounded_step(current: np.ndarray, goal: np.ndarray, target: Any) -> np.ndarray:
    """One row toward ``goal`` that the guard passes unclamped (fast outside, slow inside its slow zone)."""
    fast = _step(current, goal, 0.95 * LIMITS.max_speed_mm_s * DT_S, 0.95 * LIMITS.max_rot_deg_s * DT_S)
    if _in_slow_zone(current, target) or _in_slow_zone(fast, target):
        fast = _step(current, goal, 0.9 * LIMITS.slow_speed_mm_s * DT_S, 0.9 * LIMITS.slow_rot_deg_s * DT_S)
    return goal.copy() if _same(fast, goal, 0.05) else fast


def next_row(current: np.ndarray, gripper: float, grasp: np.ndarray, insert: np.ndarray,
             target: Any) -> tuple[np.ndarray, float]:
    """Semantic pick-insert: reach grasp, close, carry to insert, open, retreat 10 mm."""
    # 0.3 mm windows: a measured TCP carries lag and +-0.05 mm noise, so "at the insert" can read slightly below it.
    over_insert = (float(np.linalg.norm(current[:2] - insert[:2])) <= 0.3
                   and -0.3 <= current[2] - insert[2] <= 10.5)
    if gripper == 0.0 and over_insert:
        retreat = insert.copy()
        retreat[2] += 10.0
        return bounded_step(current, retreat, target), 0.0
    goal, close_when_there = (grasp, 1.0) if gripper == 0.0 else (insert, 0.0)
    if _same(current, goal):
        return current.copy(), close_when_there  # change the gripper only while still
    return bounded_step(current, goal, target), gripper


class ScriptedGroot:
    """ZMQ REP server speaking the GR00T PolicyServer protocol with scripted semantic rows."""

    def __init__(self, *, inject_low_row: bool = False, low_row_below_floor_mm: float = 5.0) -> None:
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.port = self.socket.bind_to_random_port("tcp://127.0.0.1")
        self.inject_low_row, self.low_below = inject_low_row, low_row_below_floor_mm
        self.injected_row: list[float] | None = None
        self.injected_floor: float | None = None
        self.injected_at: float | None = None
        self.guard_target: GuardTargetFn = lambda: None
        self.get_actions = 0
        self.error: Exception | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    @property
    def url(self) -> str:
        return f"groot://127.0.0.1:{self.port}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.socket.close(linger=0)
        self.context.term()

    def _serve(self) -> None:
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(timeout=50)).get(self.socket):
                continue
            request = unpack(self.socket.recv())
            try:
                reply = self._reply(request)
            except (KeyError, TypeError, ValueError) as error:
                self.error = error
                reply = {"error": f"scripted GR00T could not answer: {error}"}
            self.socket.send(pack(reply))

    def _reply(self, request: dict[str, Any]) -> object:
        endpoint = request.get("endpoint")
        if endpoint in ("ping", "reset"):
            return {"status": "ok"}
        if endpoint == "describe":
            return attestation_envelope()
        if endpoint != "get_action":
            return {"error": f"unknown endpoint: {endpoint}"}
        self.get_actions += 1
        observation = request["data"]["observation"]
        language = observation["language"]["annotation.human.task_description"][0][0]
        state = observation["state"]
        current = eef9d_to_tcp(np.asarray(state["eef_9d"], dtype=float)[0, 0][None], DOWN)[0]
        gripper = float(np.asarray(state["gripper"], dtype=float)[0, 0, 0])
        grasp, insert = _poses(language)
        target = self.guard_target()
        row, grip = self._maybe_low_row(current, gripper, insert, target)
        if row is None:
            row, grip = next_row(current, gripper, grasp, insert, target)
        action = {"eef_9d": tcp_to_eef9d(row[None])[None], "gripper": np.asarray([[[grip]]], dtype=np.float32)}
        return [action, {"rehearsal": "scripted-pick-insert"}]

    def _maybe_low_row(self, current: np.ndarray, gripper: float, insert: np.ndarray,
                       target: Any) -> tuple[np.ndarray | None, float]:
        if not self.inject_low_row or self.injected_row is not None or gripper != 1.0 or target is None:
            return None, gripper
        if float(np.hypot(current[0] - target.x_mm, current[1] - target.y_mm)) > 5.0:
            return None, gripper
        floor = target.z_contact_mm - LIMITS.floor_margin_mm
        row = current.copy()
        row[:3] = (target.x_mm + 0.5, target.y_mm, floor - self.low_below)
        row[3:6] = insert[3:6]
        self.injected_row, self.injected_floor, self.injected_at = row.tolist(), floor, time.monotonic()
        return row, gripper
