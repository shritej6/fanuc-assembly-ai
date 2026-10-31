"""PlannerRuntime: the planner drives every step from placement.yaml through the same guard; GR00T is never asked."""
from __future__ import annotations

from itertools import pairwise

import numpy as np
from direct_fakes import ScriptedGroot, TinyCell
from test_hybrid_runtime import LaggyCell

from policy_learn.deploy.direct_plan import load_placement
from policy_learn.deploy.hybrid_runtime import EDGE_MM
from policy_learn.deploy.planner_runtime import PlannerRuntime

PROMPT_1 = load_placement()[0].prompt


class NeverAskedGroot(ScriptedGroot):
    def predict(self, obs: dict):
        raise AssertionError("the planner-driven runtime asked GR00T for motion")


def _run(cell, horizon=8, max_acts=4000):
    fake = NeverAskedGroot()
    pol = PlannerRuntime(model=fake, exec_horizon=horizon)
    pol.reset()
    fake.runtime = pol
    cell.pol = pol
    for _ in range(max_acts):
        if pol.progress.ended:
            break
        cell.play(pol.act(cell.obs()))
    return pol


def test_planner_assembles_product_5_from_home_without_groot():
    pol = _run(TinyCell(PROMPT_1))
    assert pol.progress.ended == "done", pol.events[-3:]
    assert [e["part"] for e in pol.events if e["type"] == "step_done"] == ["M5_01", "M5_02", "M5_03", "M5_04"]
    assert [e["mode"] for e in pol.events if e["type"] == "takeover"] == ["planner grasp", "planner insert"] * 4
    assert [g for g in pol.guard_log if g["held"]] == []


def test_planner_completes_on_a_lagging_arm():
    pol = _run(LaggyCell(PROMPT_1))
    assert pol.progress.ended == "done", pol.events[-3:]
    assert [g for g in pol.guard_log if g["held"]] == []


def test_a_lagging_arm_far_above_the_pad_descends_fast_to_the_slow_zone_edge():
    """Real cell 16:07: the arm lagged 40 mm behind, and the rows crawled at slow-zone speed all the way down."""
    pol = PlannerRuntime(model=NeverAskedGroot(), exec_horizon=8)
    pol._plan_contact = -324.6
    pose = np.array([373.8, 183.8, -284.6, -180.0, 0.0, 150.0])
    out, reached = pol._toward(pose, (373.8, 183.8, -324.6, 150.0))
    assert not reached
    assert out[2] == -284.6 - PlannerRuntime.STEP_MM_DEG[0] or out[2] == -324.6 + EDGE_MM
    assert out[2] >= -324.6 + EDGE_MM



class RecordingCell(TinyCell):
    pol = None
    played: list

    def play(self, rows) -> None:
        pr = self.pol.progress
        self.played.append((pr.index, pr.phase, np.asarray(self.tcp, float).copy(), np.asarray(rows, float).reshape(-1, 7)))
        super().play(rows)


def test_inserts_move_at_slow_speed_from_above_first_contact():
    """Run r20261031-57450f: M5_02 engages 13 mm above its seat and hit the base at free-space speed."""
    cell = RecordingCell(PROMPT_1)
    cell.played = []
    pol = _run(cell)
    assert pol.progress.ended == "done", pol.events[-3:]
    step = pol.progress.steps[1]
    assert step.contact_z - step.seat_tcp[2] > EDGE_MM   # M5_02: the case the old slow zone missed
    slow_mm, checked = PlannerRuntime.STEP_MM_DEG[1], 0
    for index, phase, tcp, rows in cell.played:
        if index != 1 or phase != "insert":
            continue
        z = np.concatenate([[tcp[2]], rows[:, 2]])
        for a, b in pairwise(z):
            if min(a, b) < step.contact_z + EDGE_MM - 0.1:
                assert abs(b - a) <= slow_mm + 1e-6, (a, b)
                checked += 1
    assert checked > 5
