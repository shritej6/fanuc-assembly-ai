"""A real-arm-like test double for the official mock cell (rehearsal only, never motion authority).

The official ``robot/cell/mock_cell.py`` jumps the TCP to each row the instant it is dequeued and flips the
gripper the same tick. A real FANUC ER-4iA does neither. ``LaggyMock`` keeps the exact HTTP protocol, limits and
queue timing of the official mock (it subclasses it and reuses its handler), but:

* the measured TCP follows the commanded row with first-order lag (time constant ``tau_s``; about 3*tau to
  settle, 0.3-0.6 s for the defaults) plus uniform +-``noise_mm`` measurement noise; ``tau_s=0`` is an ideal
  arm (at the row on dequeue, like the official mock) used to probe behaviour past a lag-induced freeze;
* the gripper reaches a new command only after ``gripper_delay_s`` and reports only 0 or 1;
* optional faults: drop the held part (gripper keeps reading 1), drop ``tcp_pose`` from one observation.

Every commanded row is recorded so a scenario can prove what actually reached the arm.
"""

from __future__ import annotations

import math
import random
import threading
import time
from dataclasses import dataclass, field
from http.server import ThreadingHTTPServer
from typing import Any, Self

import cv2

from robot.cell.mock_cell import HOME, Mock, handler

PHYSICS_HZ = 200.0


@dataclass(frozen=True)
class CellFaults:
    """Faults injected by the laggy cell; ``None`` disables each one."""

    drop_after_grasp_s: float | None = None  # part falls this long after the gripper reads closed
    missing_tcp_after_s: float | None = None  # first act()-eligible observation after this drops tcp_pose


@dataclass
class CellLog:
    """What the cell saw: commanded rows, timings, and fault times (cell clock, time.monotonic)."""

    commanded: list[tuple[float, list[float]]] = field(default_factory=list)
    trial_started_at: float | None = None
    grasp_closed_at: float | None = None
    dropped_at: float | None = None
    tcp_missing_at: float | None = None
    trial_rows: list[int] = field(default_factory=list)


def _wrap180(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class LaggyMock(Mock):
    """Official mock protocol with a lagging, noisy arm and a slow binary gripper."""

    def __init__(
        self,
        n_trials: int,
        trial_s: float,
        instruction: str,
        *,
        tau_s: float = 0.15,
        noise_mm: float = 0.05,
        gripper_delay_s: float = 0.5,
        faults: CellFaults | None = None,
        seed: int = 7,
    ) -> None:
        self.tau_s, self.noise_mm, self.gripper_delay_s = tau_s, noise_mm, gripper_delay_s
        self.faults = faults or CellFaults()
        self.log = CellLog()
        self.rng = random.Random(seed)
        self.actual: list[float] | None = None
        self.cmd: list[float] | None = None
        self.grip_actual, self.grip_cmd, self.grip_due = 0.0, 0.0, None
        self.part_present = False
        self._tcp_fault_used = False
        super().__init__(n_trials, trial_s, "mock", instruction, quiet=True)
        threading.Thread(target=self._physics, daemon=True).start()

    def place_arm(self, tcp: list[float]) -> None:
        """Put the arm at a measured pose after homing, as the operator's ready state does."""
        with self.lock:
            self.actual, self.cmd = list(tcp), list(tcp)
            self.grip_actual = self.grip_cmd = 0.0
            self.grip_due, self.part_present = None, False
            self.tcp = self._noisy(self.actual)
            self.q = self.q[:6] + [0.0]

    def end_trial(self) -> None:
        """End the running trial now (the operator's stop); the operator thread then moves on."""
        with self.lock:
            self.queue.clear()
            if self.active is not None and self.active["status"] == "running":
                self.active["status"] = "done"
            self.state = "homing"

    def _noisy(self, pose: list[float]) -> list[float]:
        return [v + self.rng.uniform(-self.noise_mm, self.noise_mm) if k < 3 else v for k, v in enumerate(pose)]

    def _operator(self) -> None:  # the official loop, but a trial may also end early via end_trial()
        for t in self.trials:
            with self.lock:
                self.q, self.step, self.active, self.state, self.tcp = list(HOME), 0, t, "homing", None
                self.actual = self.cmd = None
            time.sleep(0.5)
            with self.lock:
                self.state, t["status"], t["instruction"] = "idle", "ready", self.instruction
            self.current = t["index"] + 1
            while t["status"] == "ready":
                time.sleep(0.02)
            self.log.trial_started_at = time.monotonic()
            t0 = time.time()
            while time.time() - t0 < self.trial_s and t["status"] == "running":
                time.sleep(0.02)
            with self.lock:
                self.queue.clear()
                t["status"], self.state = "done", "homing"
            self.log.trial_rows.append(self.stats["actions"])
            self.rollouts.append({"actions": self.stats["actions"], "rejected": self.stats["rejected"],
                                  "latency": list(self.latency)})
            self.stats.clear()
            self.latency.clear()
            time.sleep(0.5)
        with self.lock:
            self.active, self.state = None, "idle"

    def _play(self) -> None:  # same cadence as the official mock; rows become setpoints, not teleports
        t_next = time.perf_counter()
        while True:
            with self.lock:
                item = self.queue.popleft() if self.queue and self.state == "running" else None
                if item:
                    row = list(item[0])
                    self.cmd = row[:6]
                    if self.tau_s <= 0.0 and self.actual is not None:  # ideal arm: at the row on dequeue
                        self.actual = list(self.cmd)
                        self.tcp = self._noisy(self.actual)
                    if row[6] != self.grip_cmd:
                        self.grip_cmd, self.grip_due = row[6], time.monotonic() + self.gripper_delay_s
                    self.log.commanded.append((time.monotonic(), row))
                    self.step += 1
                    self.stats["actions"] += 1
            if item is None:
                time.sleep(0.005)
                t_next = time.perf_counter()
                continue
            t_next += item[1]
            time.sleep(max(0.0, t_next - time.perf_counter()))

    def _physics(self) -> None:
        period = 1.0 / PHYSICS_HZ
        alpha = 1.0 - math.exp(-period / self.tau_s) if self.tau_s > 0.0 else 1.0
        while True:
            time.sleep(period)
            with self.lock:
                self._advance(alpha)

    def _advance(self, alpha: float) -> None:
        if self.actual is None or self.cmd is None:
            return
        for k in range(6):
            delta = self.cmd[k] - self.actual[k] if k < 3 else _wrap180(self.cmd[k] - self.actual[k])
            self.actual[k] += alpha * delta
        now = time.monotonic()
        if self.grip_due is not None and now >= self.grip_due:
            self.grip_actual, self.grip_due = self.grip_cmd, None
            self.part_present = self.grip_actual == 1.0 and self.log.dropped_at is None
            if self.part_present:
                self.log.grasp_closed_at = now
        drop = self.faults.drop_after_grasp_s
        closed_at = self.log.grasp_closed_at
        if drop is not None and self.part_present and closed_at is not None and now - closed_at >= drop:
            self.part_present = False  # the gripper still reads 1: a 0/1 gripper cannot see this
            self.log.dropped_at = now
        self.tcp = self._noisy(self.actual)
        self.q = self.q[:6] + [self.grip_actual]

    def take_missing_tcp_fault(self) -> bool:
        """True once, for the first act()-eligible observation after the configured delay."""
        after = self.faults.missing_tcp_after_s
        started = self.log.trial_started_at
        if after is None or self._tcp_fault_used or started is None or self.state != "running" or self.queue:
            return False
        if time.monotonic() - started < after:
            return False
        self._tcp_fault_used = True
        self.log.tcp_missing_at = time.monotonic()
        return True

    def card(self, cam: str):
        img = super().card(cam)
        if cam in ("table", "wrist") and self.log.dropped_at is not None:
            scale = img.shape[1] / 640
            cv2.putText(img, "PART DROPPED", (int(20 * scale), int(250 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        1.4 * scale, (255, 0, 0), max(1, int(3 * scale)))
        return img



def laggy_handler(mock: LaggyMock):
    """The official handler, with ``tcp_pose`` removed from one observation when that fault is armed."""
    base = handler(mock)

    class LaggyHandler(base):  # type: ignore[misc, valid-type]
        def _send(self, code: int, body: Any) -> None:
            if isinstance(body, dict) and "tcp_pose" in body and mock.take_missing_tcp_fault():
                body = {**body, "tcp_pose": None}
            super()._send(code, body)

    return LaggyHandler


class LaggyCellServer:
    """Serve a ``LaggyMock`` on a free local port with the official HTTP handler."""

    def __init__(self, mock: LaggyMock) -> None:
        self.mock = mock
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), laggy_handler(mock))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
