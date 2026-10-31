"""Rehearse path B (T39 DirectGrootRuntime, robot/cell/policy_direct.py) against the laggy, noisy, faulty arm.

Run from the root of a T39 worktree with this directory copied to tools/rehearsal (tools/rehearsal/run_b.sh does it):
    uv run python -m tools.rehearsal.direct_b [--groot groot://127.0.0.1:5555]

Path exercised: official cell_client.run_all -> policy_direct.Policy (DirectGrootRuntime) -> GrootPolicy -> ChunkGuard
-> HTTP -> LaggyMock, on the trial-0 rollout of the branch's own placement.yaml. GR00T is T39's own test stand-in
(tests/policy_learn/direct_fakes.ScriptedGroot: plays the runtime's current step, as a well-trained model would), or
the real server with --groot. Nothing here reaches a robot.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO = Path.cwd().resolve()
CELL = REPO / "robot" / "cell"
for entry in (str(REPO / "tests" / "policy_learn"), str(CELL), str(REPO)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from tools.rehearsal.laggy_cell import CellFaults, LaggyCellServer, LaggyMock

INITIAL_TCP = [400.0, 0.0, -200.0, -180.0, 0.0, 0.0]
REAL_ARM = {"tau_s": 0.15, "noise_mm": 0.05}
IDEAL_ARM = {"tau_s": 0.0, "noise_mm": 0.0}


@dataclass
class BOutcome:
    name: str
    steps_done: int = 0
    ended: str | None = None
    ended_after_s: float | None = None
    trial_rows: list[int] = field(default_factory=list)
    rejections: int = 0
    process_exit: str | None = None
    drop_at_s: float | None = None
    steps_at_drop: int | None = None
    step_done_after_drop: bool = False
    elapsed_s: float = 0.0


def _prompt() -> str:
    data = yaml.safe_load((CELL / "submission" / "placement.yaml").read_text(encoding="utf-8"))
    return str(data["rollouts"][0]["prompt"])


class LatchedGroot:
    """ScriptedGroot with the gripper held at its last commanded value, as a trained model holds it.

    ScriptedGroot copies the OBSERVED gripper into every motion row. TinyCell and the official mock flip the gripper
    the tick it is commanded, so that works there; with a real 0.5 s gripper the rows after a close still read
    "open", re-command open and cancel the close (the arm sits on the pad until the pick budget runs out). GR00T
    emits its own gripper action, so this keeps the commanded intent instead of echoing the reading.

    ScriptedGroot also plans every chunk from the observed pose and advances its waypoint list as if each chunk was
    played. On a lagging arm the observed pose trails the last row, so its next chunk can open with a step the guard
    refuses (correctly), and its script then skips ahead (it closed 9 mm above the pad). A trained model re-predicts
    from where the arm is; here the wrapper re-sends the last row until the arm is within SETTLE_MM of it. Farther
    than TRACK_MM (more than one fast row) the arm was moved by something else, e.g. the FK-estimate hold after a
    missing tcp_pose (the mock never updates the joints on tcp actions, so its FK estimate is the home pose), and the
    stand-in re-plans from where the arm is."""

    SETTLE_MM, TRACK_MM = 1.0, 30.0

    def __init__(self) -> None:
        from direct_fakes import ScriptedGroot

        self.inner, self.intent, self.last = ScriptedGroot(), None, None

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def reset(self) -> None:  # the runtime resets the model at each step start, gripper open
        self.inner.reset()
        self.intent = self.last = None

    def predict(self, obs: dict):
        import numpy as np

        tcp = np.asarray(obs["tcp_pose"], float)[:3]
        lag = None if self.last is None else float(np.linalg.norm(tcp - self.last[:3]))
        if lag is not None and self.SETTLE_MM < lag <= self.TRACK_MM:
            return self.last[None].copy()  # the arm has not reached the last row yet
        rows = self.inner.predict(obs).copy()
        observed = 1.0 if float(obs["joints"][6]) >= 0.5 else 0.0
        if len(rows) == 1 and rows[0, 6] != observed and rows[0, 6] != self.intent:
            self.intent = float(rows[0, 6])  # the scripted gripper-change row: new intent
        elif self.intent is not None:
            rows[:, 6] = self.intent
        self.last = rows[-1].copy()
        return rows


def _policy(groot_url: str | None):
    import policy_direct

    if groot_url:
        return policy_direct.Policy(groot_url)
    fake = LatchedGroot()
    policy = policy_direct.Policy(model=fake)
    fake.inner.runtime = policy
    return policy


def _arm_when_ready(mock: LaggyMock) -> None:
    def watch() -> None:
        seen: set[str] = set()
        while len(seen) < len(mock.trials):
            for trial in mock.trials:
                if trial["status"] == "ready" and trial["trial_id"] not in seen:
                    mock.place_arm(INITIAL_TCP)
                    seen.add(trial["trial_id"])
            time.sleep(0.005)

    threading.Thread(target=watch, daemon=True).start()


def _monitor(mock: LaggyMock, policy, out: BOutcome, budget_s: float, n_trials: int, steps_needed: int) -> None:
    """End each trial when the runtime ends it (done or held) or at the budget; record the first trial's result."""
    for k, trial in enumerate(mock.trials[:n_trials]):
        while trial["status"] not in ("running", "done"):
            time.sleep(0.01)
        start = time.monotonic()
        while trial["status"] == "running" and time.monotonic() - start < budget_s:
            done = sum(e["type"] == "step_done" for e in policy.events)
            if mock.log.dropped_at is not None and out.drop_at_s is None:
                out.drop_at_s = round(mock.log.dropped_at - (mock.log.trial_started_at or start), 2)
                out.steps_at_drop = done
            if out.drop_at_s is not None and out.steps_at_drop is not None and done > out.steps_at_drop:
                out.step_done_after_drop = True
            ended = policy.progress.ended
            if k == 0:
                out.steps_done = done
            if ended is not None or (k == 0 and done >= steps_needed):
                if k == 0 and out.ended is None:
                    out.ended = ended or "done"
                    out.ended_after_s = round(time.monotonic() - (mock.log.trial_started_at or start), 2)
                break
            time.sleep(0.02)
        mock.end_trial()


def run_b(name: str, *, budget_s: float, tau_s: float, noise_mm: float, faults: CellFaults | None = None,
          n_trials: int = 1, steps_needed: int = 4, groot_url: str | None = None) -> BOutcome:
    from cell_client import CellClient, run_all

    out, started = BOutcome(name), time.monotonic()
    mock = LaggyMock(n_trials, budget_s + 5.0, _prompt(), faults=faults, tau_s=tau_s, noise_mm=noise_mm)
    with ExitStack() as stack:
        server = stack.enter_context(LaggyCellServer(mock))
        policy = _policy(groot_url)
        stack.callback(policy.close)
        _arm_when_ready(mock)
        mon = threading.Thread(target=_monitor, args=(mock, policy, out, budget_s, n_trials, steps_needed),
                               daemon=True)
        mon.start()
        try:
            run_all(CellClient(server.url, "mock"), policy)
        except SystemExit as exit_:
            out.process_exit = f"SystemExit({exit_.code})"
        except Exception as error:  # noqa: BLE001 - any escape is the measured result
            out.process_exit = f"{type(error).__name__}: {error}"[:160]
        mon.join(timeout=budget_s * n_trials + 30.0)
        out.trial_rows = list(mock.log.trial_rows)
        out.rejections = len(mock.rejections)
    out.elapsed_s = round(time.monotonic() - started, 1)
    return out


def _verdicts(o: BOutcome, kind: str) -> tuple[str, str]:
    if o.process_exit:
        return "FAIL", f"process would exit: {o.process_exit}"
    if o.rejections:
        return "FAIL", f"cell rejected {o.rejections} action(s)"
    if kind == "complete":
        ok = o.steps_done >= 4
        return ("PASS" if ok else "FAIL"), f"{o.steps_done}/4 steps, ended {o.ended!r} after {o.ended_after_s} s"
    if kind == "drop":
        if o.step_done_after_drop:
            return "KNOWN", f"drop at {o.drop_at_s} s NOT detected: step reported done after it ({o.ended!r})"
        return "PASS", f"drop at {o.drop_at_s} s: ended {o.ended!r} after {o.ended_after_s} s"
    if kind == "next_trial":
        ok = len(o.trial_rows) >= 2 and o.trial_rows[1] > 0
        return ("PASS" if ok else "FAIL"), f"rows per trial {o.trial_rows}"
    return "PASS", f"no exit, no rejection; {o.steps_done} steps; ended {(o.ended or '')[:110]!r}"


SCENARIOS = (
    ("B_real_arm_full_product", {**REAL_ARM, "budget_s": 400.0}, "complete"),
    ("B_ideal_arm_full_product", {**IDEAL_ARM, "budget_s": 300.0}, "complete"),
    ("B_real_arm_drop", {**REAL_ARM, "budget_s": 120.0, "faults": CellFaults(drop_after_grasp_s=1.0)}, "drop"),
    ("B_missing_tcp_next_trial", {**IDEAL_ARM, "budget_s": 30.0, "n_trials": 2,
                                  "faults": CellFaults(missing_tcp_after_s=5.0)}, "next_trial"),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groot", help="also run the real GR00T server, e.g. groot://127.0.0.1:5555")
    parser.add_argument("--only", help="run only this scenario")
    args = parser.parse_args()
    scenarios = list(SCENARIOS)
    if args.groot:
        scenarios.append(("B_real_groot_real_arm", {**REAL_ARM, "budget_s": 60.0, "groot_url": args.groot}, "safe"))
    rows, worst = [], 0
    for name, kwargs, kind in scenarios:
        if args.only and name != args.only:
            continue
        o = run_b(name, **kwargs)
        result, detail = _verdicts(o, kind)
        worst = max(worst, result == "FAIL")
        rows.append((name, result, sum(o.trial_rows), detail))
        print(json.dumps({"scenario": name, "result": result, **asdict(o)}, default=str), flush=True)
    print(f"\n{'scenario':28} {'result':7} {'rows':>5}  detail")
    for name, result, n, detail in rows:
        print(f"{name:28} {result:7} {n:>5}  {detail[:160]}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
