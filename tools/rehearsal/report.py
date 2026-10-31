"""Scenario catalogue, expected-safe verdicts, and the short table printed by run.sh."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.rehearsal.laggy_cell import CellFaults
from tools.rehearsal.scenarios import Outcome, run_scenario

DROP_DETECT_LIMIT_S = float(os.environ.get("REHEARSAL_DROP_DETECT_S", "5.0"))  # PRELIMINARY: before the insert
REAL_ARM = {"tau_s": 0.15, "noise_mm": 0.05}  # ~0.45 s to settle, +-0.05 mm measurement noise
NOISY_IDEAL_ARM = {"tau_s": 0.0, "noise_mm": 0.05}  # no lag, real-sensor noise only
IDEAL_ARM = {"tau_s": 0.0, "noise_mm": 0.0}  # official-mock arm (+0.5 s binary gripper): probes past freezes
Verdict = tuple[bool, str]


@dataclass(frozen=True)
class Scenario:
    """One rehearsal: how to run it, what expected-safe means, and the known blocker it documents (if any)."""

    name: str
    run_kwargs: dict[str, Any]
    check: Callable[[Outcome], Verdict]
    blocker: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


def _no_exit(o: Outcome) -> Verdict | None:
    if o.process_exit:
        return False, f"process would exit: {o.process_exit}"
    if o.rejections:
        return False, f"cell rejected {o.rejections} action(s)"
    return None


def expect_complete(o: Outcome) -> Verdict:
    if (bad := _no_exit(o)) is not None:
        return bad
    if o.completed:
        return True, f"pick-insert completed in {o.elapsed_s} s"
    if o.held_reason:
        return False, f"Held after {o.held_after_rows} rows / {o.held_after_s} s: {o.held_reason[:110]}"
    return False, f"no completion and no Held in {o.elapsed_s} s ({sum(o.trial_rows)} rows): livelock"


def expect_drop_detected(o: Outcome) -> Verdict:
    if (bad := _no_exit(o)) is not None:
        return bad
    if o.false_install_after_drop:
        return False, "reported the dropped part as installed"
    if o.drop_detected_after_s is None:
        return False, "drop never detected"
    if o.drop_detected_after_s > DROP_DETECT_LIMIT_S:
        return False, (f"drop noticed only after {o.drop_detected_after_s} s (limit {DROP_DETECT_LIMIT_S} s): "
                       f"{(o.held_reason or '')[:90]}")
    return True, f"drop detected after {o.drop_detected_after_s} s"


def expect_low_row_guarded(o: Outcome) -> Verdict:
    if (bad := _no_exit(o)) is not None:
        return bad
    if o.low_row in ("refused", "clamped"):
        return True, f"off-axis low row {o.low_row}; nothing below the floor reached the arm"
    return False, f"off-axis low row: {o.low_row}"


def expect_trial_contained(o: Outcome) -> Verdict:
    if (bad := _no_exit(o)) is not None:
        return bad
    if o.held_reason and "tcp" in o.held_reason.lower():
        return True, f"trial Held after {o.held_after_rows} rows, process kept running"
    return False, f"missing TCP not contained as a trial Held: {o.held_reason!r}"


def expect_next_trial_runs(o: Outcome) -> Verdict:
    if (bad := _no_exit(o)) is not None:
        return bad
    if len(o.trial_rows) >= 2 and o.trial_rows[1] > 0:
        return True, f"trial 2 sent {o.trial_rows[1]} rows after trial 1's Held"
    return False, f"trial 2 sent no rows after trial 1's Held (rows per trial {o.trial_rows}): Held is permanent"


def expect_real_groot_safe(o: Outcome) -> Verdict:
    if (bad := _no_exit(o)) is not None:
        return bad
    state = "completed" if o.completed else f"Held: {(o.held_reason or 'none')[:90]}"
    return True, f"no exit, no rejection; {state}"


MISSING_TCP_RUN = {**IDEAL_ARM, "faults": CellFaults(missing_tcp_after_s=5.0), "n_trials": 2, "budget_s": 30.0}
SCENARIOS: tuple[Scenario, ...] = (
    Scenario("real_arm_nominal", {**REAL_ARM, "budget_s": 40.0}, expect_complete,
             "settle-poll (8b47413 groot_queue.confirm_previous_endpoint) re-commands the OBSERVED pose while "
             "waiting, stopping a lagging arm short of its endpoint -> settle timeout Held"),
    Scenario("noisy_sensor_nominal", {**NOISY_IDEAL_ARM, "budget_s": 40.0}, expect_complete,
             "fixed_inspection.next_inspection_leg uses exact 1e-9 xy/angle/z checks on the measured TCP; "
             "+-0.05 mm noise livelocks inspection forever with no Held"),
    Scenario("ideal_arm_nominal", {**IDEAL_ARM, "budget_s": 45.0}, expect_complete),
    Scenario("ideal_arm_drop", {**IDEAL_ARM, "faults": CellFaults(drop_after_grasp_s=1.0), "budget_s": 45.0},
             expect_drop_detected, "no drop/tip detection in CELL (gripper 0/1, scene=None, no per-intent budget)"),
    Scenario("ideal_arm_low_offaxis_row", {**IDEAL_ARM, "inject_low_row": True, "budget_s": 45.0},
             expect_low_row_guarded),
    Scenario("ideal_arm_missing_tcp", MISSING_TCP_RUN, expect_trial_contained),
    Scenario("ideal_arm_missing_tcp_next_trial", MISSING_TCP_RUN, expect_next_trial_runs,
             "every Held is final: acknowledge_resume is not wired in CELL, trial 2 never moves"),
    Scenario("real_groot_real_arm", {**REAL_ARM, "budget_s": 40.0}, expect_real_groot_safe, tags=("groot",)),
    Scenario("real_groot_ideal_arm", {**IDEAL_ARM, "budget_s": 45.0}, expect_real_groot_safe, tags=("groot",)),
)


_RUNS: dict[str, Outcome] = {}


def outcome_for(name: str) -> Outcome:
    """Run a scenario once per process; scenarios with identical run settings share one run."""
    scenario = next(s for s in SCENARIOS if s.name == name)
    kwargs = dict(scenario.run_kwargs)
    if "groot" in scenario.tags:
        kwargs["groot_url"] = os.environ["REHEARSAL_GROOT_URL"]
    key = json.dumps({k: repr(v) for k, v in kwargs.items()}, sort_keys=True)
    if key not in _RUNS:
        _RUNS[key] = run_scenario(name, **kwargs)
    return _RUNS[key]


def record(scenario: Scenario, outcome: Outcome, verdict: Verdict) -> None:
    """Append one row to $REHEARSAL_REPORT (JSON lines) for run.sh's table."""
    path = os.environ.get("REHEARSAL_REPORT")
    if not path:
        return
    row = {"scenario": scenario.name, "safe": verdict[0], "detail": verdict[1], "blocker": scenario.blocker,
           "outcome": asdict(outcome)}
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=repr) + "\n")


def print_table(path: Path) -> None:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"{'scenario':34} {'result':16} {'rows':>5} {'held@s':>7}  detail")
    for row in rows:
        o = row["outcome"]
        if row["safe"]:
            result = "XPASS(fixed?)" if row["blocker"] else "PASS"
        else:
            result = "XFAIL(known)" if row["blocker"] else "FAIL"
        held = "-" if o["held_after_s"] is None else str(o["held_after_s"])
        print(f"{row['scenario']:34} {result:16} {sum(o['trial_rows']):>5} {held:>7}  {row['detail'][:150]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    print_table(parser.parse_args().report)


if __name__ == "__main__":
    main()
