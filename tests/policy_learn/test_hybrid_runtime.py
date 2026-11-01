"""Tests for hybrid runtime.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import numpy as np
from direct_fakes import ScriptedGroot, TinyCell

from policy_learn.deploy.direct_plan import load_placement
from policy_learn.deploy.hybrid_runtime import HybridRuntime

PROMPT_1 = load_placement()[0].prompt
AIM_OFF_Y_MM = 12.0   # D's sim traces: GR00T lands 6-17 mm off the pad in +y


class OffsetGroot(ScriptedGroot):
    """A GR00T that aims AIM_OFF_Y_MM off every target, stops 20 mm above it and never opens or closes."""

    def _waypoints(self, step, phase):
        if phase == "retreat":
            return super()._waypoints(step, phase)
        wps = [w for w in super()._waypoints(step, phase) if isinstance(w, tuple)]
        z_stop = (step.grasp_tcp if phase == "pick" else step.seat_tcp)[2] + 20.0
        return [(x, y + AIM_OFF_Y_MM, max(z, z_stop), r) for x, y, z, r in wps]


class LaggyCell(TinyCell):
    """While the planner drives, each chunk gets the arm only 60 % of the way to its last row (the real arm lags the 5 Hz rows)."""

    pol = None

    def play(self, rows) -> None:
        if getattr(self.pol, "_plan_path", None) is None:
            return super().play(rows)
        last = np.asarray(rows, float).reshape(-1, 7)[-1]
        self.tcp = list(np.asarray(self.tcp) + 0.6 * (last[:6] - np.asarray(self.tcp)))
        self.grip = float(last[6])


def _run(cell, horizon=8, max_acts=4000):
    fake = OffsetGroot()
    pol = HybridRuntime(model=fake, exec_horizon=horizon)
    pol.reset()
    fake.runtime = pol
    cell.pol = pol
    for _ in range(max_acts):
        if pol.progress.ended:
            break
        cell.play(pol.act(cell.obs()))
    return pol


def test_planner_takeover_assembles_product_5_when_groot_aims_off():
    pol = _run(TinyCell(PROMPT_1))
    assert pol.progress.ended == "done", pol.events[-3:]
    assert [e["part"] for e in pol.events if e["type"] == "step_done"] == ["M5_01", "M5_02", "M5_03", "M5_04"]
    kinds = [e["mode"] for e in pol.events if e["type"] == "takeover"]
    assert kinds == ["grasp", "insert"] * 4
    assert [g for g in pol.guard_log if g["held"]] == []


def test_takeover_still_completes_on_a_lagging_arm():
    pol = _run(LaggyCell(PROMPT_1))
    assert pol.progress.ended == "done", pol.events[-3:]
    assert [g for g in pol.guard_log if g["held"]] == []


def test_groot_alone_with_the_same_aim_never_grasps():
    from policy_learn.deploy.direct_runtime import Budgets, DirectGrootRuntime

    class Clock:
        t = 0.0

        def __call__(self):
            Clock.t += 0.2
            return Clock.t

    fake = OffsetGroot()
    pol = DirectGrootRuntime(model=fake, exec_horizon=8, clock=Clock(), budgets=Budgets(pick_s=30.0))
    pol.reset()
    fake.runtime = pol
    cell = TinyCell(PROMPT_1)
    while not pol.progress.ended:
        cell.play(pol.act(cell.obs()))
    assert "pick budget" in pol.progress.ended
