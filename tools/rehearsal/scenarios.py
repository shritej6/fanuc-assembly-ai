"""Rehearsal scenarios through Codex's real CELL code path against the laggy cell.

Path exercised: ``robot/cell/trial_runner.run_cell_trials`` -> official ``cell_client.run_all/run_trial`` ->
``Policy(runtime="cell")`` -> ``create_cell_policy`` -> ``T09GrootExecutor`` -> ``ChunkGuard`` -> HTTP ->
``LaggyMock``. Only the official-mock transport and the reviewed mock proof factory are used, because Codex's
runtime (correctly) refuses the REAL transport until a reviewed real factory exists.

Two harness substitutions, both documented in the report:
* the mock factory's ``ExecutionTolerances(0.0, 0.0)`` assume TCP == command; the laggy arm cannot meet a zero
  tolerance, so the Saturday tolerances (0.3 mm / 1 deg, ``SATURDAY_EXECUTION_TOLERANCES``) are used instead;
* mock perception's scripted fallen/failed-insertion fixtures are cleared (as Codex's positive proof does), and
  after a drop it reports the held part missing - what a real wrist/table camera would show.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(os.environ.get("REHEARSAL_REPO", Path(__file__).resolve().parents[2])).resolve()
CELL_DIR = REPO / "robot" / "cell"
for entry in (str(CELL_DIR), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools.rehearsal.laggy_cell import CellFaults, LaggyCellServer, LaggyMock
from tools.rehearsal.scripted_groot import ScriptedGroot

INITIAL_TCP = [400.0, 0.0, -200.0, -180.0, 0.0, 0.0]
INSTRUCTION = "assemble product 1"
FACTORY_ENV, PROOF_ENV = "FANUC_T09_CELL_FACTORY", "FANUC_T09_SOFTWARE_PROOF"
MOCK_FACTORY = "robot.cell.t09_mock_runtime:build_groot_mock_controller"


@dataclass
class Outcome:
    """What happened, measured from the controller and from the rows the laggy cell actually received."""

    name: str
    completed: bool = False
    held_reason: str | None = None
    held_after_rows: int | None = None
    held_after_s: float | None = None
    drop_detected_after_s: float | None = None
    false_install_after_drop: bool = False
    low_row: str = "n/a"
    process_exit: str | None = None
    trial_rows: list[int] = field(default_factory=list)
    rejections: int = 0
    installs: int = 0
    get_actions: int = 0
    elapsed_s: float = 0.0
    error: str | None = None


@contextmanager
def _patched_runtime(cell: LaggyMock):
    """Use Saturday tolerances and camera-truthful perception in the mock proof factory."""
    import robot.cell.t09_mock_runtime as mock_runtime
    from assembly.cell_runtime import SATURDAY_EXECUTION_TOLERANCES

    perception_cls = mock_runtime.MockPerception
    saved = {name: getattr(perception_cls, name)
             for name in ("_configure_from_prompt", "pre_pick", "post_place", "collision_scene")}
    saved_kwargs = mock_runtime._controller_kwargs
    saved_env = {key: os.environ.get(key) for key in (FACTORY_ENV, PROOF_ENV)}

    def configure(self: Any, observation: dict) -> None:
        saved["_configure_from_prompt"](self, observation)
        self.fallen_once.clear()
        self.failed_insertions.clear()

    def post_place(self: Any, observation: dict, part: Any) -> Any:
        evidence = saved["post_place"](self, observation, part)
        if cell.log.dropped_at is None:
            return evidence
        return replace(evidence, seated=False, pose_agrees=False)

    def collision_scene(self: Any, observation: dict, moving_instance_id: str, base_instance_id: str) -> Any:
        # A wrist/table camera sees the held part, the 0/1 gripper does not: after a drop it is no longer attached.
        scene = saved["collision_scene"](self, observation, moving_instance_id, base_instance_id)
        if scene.moving_attached and cell.log.dropped_at is not None:
            return replace(scene, moving_attached=False)
        return scene

    def controller_kwargs(perception: Any) -> dict[str, object]:
        kwargs = saved_kwargs(perception)
        kwargs["execution_tolerances"] = SATURDAY_EXECUTION_TOLERANCES
        return kwargs

    perception_cls._configure_from_prompt = configure
    perception_cls.post_place = post_place
    perception_cls.collision_scene = collision_scene
    mock_runtime._controller_kwargs = controller_kwargs
    os.environ[FACTORY_ENV], os.environ[PROOF_ENV] = MOCK_FACTORY, "1"
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(perception_cls, name, value)
        mock_runtime._controller_kwargs = saved_kwargs
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _held_reason(policy: Any) -> str | None:
    from assembly.state_machine import PackMLState

    record = policy.controller.state_machine.current_record
    if record.state is PackMLState.HELD:
        return str(record.reason or "Held (no reason recorded)")
    return None


def _installs(policy: Any) -> int:
    return sum(event.kind == "installed" for event in policy.controller.events)


def _installed(policy: Any) -> bool:
    return _installs(policy) > 0


class _Monitor(threading.Thread):
    """Watch the controller; end the trial on completion, Held, or the wall-clock budget."""

    def __init__(self, cell: LaggyMock, policy: Any, outcome: Outcome, budget_s: float, installs: int = 1) -> None:
        super().__init__(daemon=True)
        self.cell, self.policy, self.outcome, self.budget_s = cell, policy, outcome, budget_s
        self.installs = installs
        self.stop = threading.Event()

    def run(self) -> None:
        start = time.monotonic()
        while not self.stop.is_set() and time.monotonic() - start < self.budget_s:
            self._observe_drop()
            if self._terminal():
                self.cell.end_trial()
                return
            time.sleep(0.02)
        self.cell.end_trial()

    def _observe_drop(self) -> None:
        dropped = self.cell.log.dropped_at
        if dropped is None or self.outcome.drop_detected_after_s is not None:
            return
        flagged = any(event.kind in ("dropped", "andon", "abort") for event in self.policy.controller.events)
        if flagged or _held_reason(self.policy) is not None:
            self.outcome.drop_detected_after_s = round(time.monotonic() - dropped, 2)
        elif _installed(self.policy):
            self.outcome.false_install_after_drop = True

    def _terminal(self) -> bool:
        reason = _held_reason(self.policy)
        if reason is not None and self.outcome.held_reason is None:
            started = self.cell.log.trial_started_at or time.monotonic()
            self.outcome.held_reason = reason
            self.outcome.held_after_rows = self.cell.stats["actions"]
            self.outcome.held_after_s = round(time.monotonic() - started, 2)
        self.outcome.installs = _installs(self.policy)
        if self.outcome.installs >= self.installs:
            self.outcome.completed = True
        if self.cell.log.dropped_at is not None and self.outcome.drop_detected_after_s is None:
            return self.outcome.completed  # keep watching a dropped part until detection or the budget
        # The CURRENT Held state ends this trial; a Held recorded in an earlier trial must not end the next one.
        return self.outcome.completed or reason is not None


def _low_row_verdict(groot: ScriptedGroot, cell: LaggyMock, guard_log: list[dict]) -> str:
    """Refused / clamped by the guard, or COMMANDED if any row below the floor reached the arm."""
    if groot.injected_row is None or groot.injected_at is None:
        return "not_reached"
    floor = groot.injected_floor if groot.injected_floor is not None else float("-inf")
    x, y = groot.injected_row[0], groot.injected_row[1]
    below = [row for at, row in cell.log.commanded if at >= groot.injected_at and row[2] < floor - 1e-6
             and float(np.hypot(row[0] - x, row[1] - y)) <= 30.0]
    if below:
        return f"COMMANDED z={min(row[2] for row in below):.2f} below floor {floor:.2f}"
    reasons = [(entry["held"], " ".join(entry["reasons"])) for entry in guard_log]
    if any(held and "below the contact floor" in text for held, text in reasons):
        return "refused"
    if any("raised to the contact floor" in text for _, text in reasons):
        return "clamped"
    return "UNGUARDED (no guard entry and no command below floor)"


def run_scenario(name: str, *, faults: CellFaults | None = None, inject_low_row: bool = False,
                 groot_url: str | None = None, n_trials: int = 1, budget_s: float = 75.0,
                 tau_s: float = 0.15, noise_mm: float = 0.05, installs: int = 1) -> Outcome:
    """Run one rehearsal scenario end to end and return its measured outcome."""
    from cell_client import CellClient, run_all
    from policy import Policy
    from trial_runner import run_cell_trials

    from assembly.cell_runtime import CellTransport

    outcome, started = Outcome(name), time.monotonic()
    mock = LaggyMock(n_trials, budget_s + 5.0, INSTRUCTION, faults=faults, tau_s=tau_s,
                     noise_mm=noise_mm)
    with ExitStack() as stack:
        groot = None if groot_url else stack.enter_context(ScriptedGroot(inject_low_row=inject_low_row))
        stack.enter_context(_patched_runtime(mock))
        server = stack.enter_context(LaggyCellServer(mock))
        policy = Policy(checkpoint=groot_url or groot.url, runtime="cell",
                        cell_transport=CellTransport.OFFICIAL_MOCK)
        stack.callback(policy.close)
        if groot is not None:
            guard = policy.controller.groot_executor._policy.guard
            groot.guard_target = lambda: guard.target
        _arm_when_ready(mock)
        monitors = _MonitorChain(mock, policy, outcome, budget_s, n_trials, installs)
        monitors.start()
        try:
            run_cell_trials(CellClient(server.url, "mock"), policy, run_all_fn=run_all)
        except SystemExit as exit_:
            outcome.process_exit = f"SystemExit({exit_.code})"
        except Exception as error:  # noqa: BLE001 - any escape is the measured result: run.py exits or crashes
            outcome.process_exit = f"{type(error).__name__}: {error}"[:160]
        monitors.join()
        outcome.trial_rows = list(mock.log.trial_rows)
        outcome.rejections = len(mock.rejections)
        outcome.get_actions = groot.get_actions if groot else -1
        guard_log = policy.controller.groot_executor._policy.guard_log
        outcome.low_row = _low_row_verdict(groot, mock, guard_log) if inject_low_row and groot else "n/a"
        outcome.error = repr(groot.error) if groot and groot.error else None
    outcome.elapsed_s = round(time.monotonic() - started, 1)
    return outcome


def _arm_when_ready(mock: LaggyMock) -> None:
    """Place the arm at its measured start pose each time a trial becomes ready."""

    def watch() -> None:
        seen: set[str] = set()
        while len(seen) < len(mock.trials):
            for trial in mock.trials:
                if trial["status"] == "ready" and trial["trial_id"] not in seen:
                    mock.place_arm(INITIAL_TCP)
                    seen.add(trial["trial_id"])
            time.sleep(0.005)

    threading.Thread(target=watch, daemon=True).start()


class _MonitorChain(threading.Thread):
    """One monitor per trial, started when that trial starts running."""

    def __init__(self, cell: LaggyMock, policy: Any, outcome: Outcome, budget_s: float, n_trials: int,
                 installs: int = 1) -> None:
        super().__init__(daemon=True)
        self.args, self.n_trials = (cell, policy, outcome, budget_s, installs), n_trials

    def run(self) -> None:
        cell = self.args[0]
        for trial in cell.trials[: self.n_trials]:
            while trial["status"] != "running":
                if trial["status"] == "done":
                    break
                time.sleep(0.01)
            monitor = _Monitor(*self.args)
            monitor.start()
            monitor.join()
