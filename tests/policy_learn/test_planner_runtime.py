"""PlannerRuntime: the planner drives every step from placement.yaml through the same guard; GR00T is never asked."""
from __future__ import annotations

from direct_fakes import ScriptedGroot, TinyCell
from test_hybrid_runtime import LaggyCell

from policy_learn.deploy.direct_plan import load_placement
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
