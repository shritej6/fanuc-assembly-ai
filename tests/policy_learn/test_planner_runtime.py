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


def test_a_lagging_arm_far_above_the_pad_descends_fast_to_the_slow_zone_edge():
    """Real cell 16:07: the arm lagged 40 mm behind, and the rows crawled at slow-zone speed all the way down."""
    import numpy as np

    from policy_learn.deploy.hybrid_runtime import EDGE_MM

    pol = PlannerRuntime(model=NeverAskedGroot(), exec_horizon=8)
    pol._plan_contact = -324.6
    pose = np.array([373.8, 183.8, -284.6, -180.0, 0.0, 150.0])
    out, reached = pol._toward(pose, (373.8, 183.8, -324.6, 150.0))
    assert not reached
    assert out[2] == -284.6 - PlannerRuntime.STEP_MM_DEG[0] or out[2] == -324.6 + EDGE_MM
    assert out[2] >= -324.6 + EDGE_MM
