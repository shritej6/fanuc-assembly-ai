"""DirectGrootRuntime: the 14:00 fallback. GR00T drives every motion through the guard; the runtime only sequences
placement.yaml's steps, advances on the observed TCP + gripper, and otherwise holds."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from policy_learn.deploy.direct_plan import load_placement, plan_rollout
from policy_learn.deploy.direct_runtime import RETREAT_MM, Budgets, DirectGrootRuntime
from policy_learn.deploy.groot_policy import GrootPolicy
from sim.assembly_lab.mimic.catalog import encode_plan_ids
from sim.assembly_lab.mimic.instructions import instruction

sys.path.insert(0, str(Path(__file__).resolve().parent))
from direct_fakes import HOME_TCP, ScriptedGroot, TinyCell

REPO = Path(__file__).resolve().parents[2]
ROLLOUTS = load_placement()
PROMPT_1 = ROLLOUTS[0].prompt


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class Fixed:
    def __init__(self, chunk):
        self.chunk, self.calls, self.resets = np.asarray(chunk, float), 0, 0

    def reset(self):
        self.resets += 1

    def predict(self, model_obs):
        self.calls += 1
        return self.chunk.copy()


class Echo(Fixed):
    """Proposes standing still: the tests below move the arm themselves through the observations."""

    def __init__(self):
        super().__init__([HOME_TCP + [0.0]])

    def predict(self, model_obs):
        self.calls += 1
        grip = 1.0 if float(model_obs["joints"][6]) >= 0.5 else 0.0
        return np.array([[*model_obs["tcp_pose"][:6], grip]])


def runtime(model=None, **kw) -> tuple[DirectGrootRuntime, Clock]:
    clock = Clock()
    pol = DirectGrootRuntime(model=model or Echo(), clock=clock, **kw)
    pol.reset()
    return pol, clock


def obs(tcp, grip=0.0, prompt=PROMPT_1, index=0, **extra) -> dict:
    o = TinyCell(prompt, index, tcp).obs()
    o["joints"][6] = grip
    return o | extra


def ended(pol) -> str | None:
    return pol.progress.ended


def test_official_policy_interface():
    assert issubclass(DirectGrootRuntime, GrootPolicy)
    assert DirectGrootRuntime.ACTION == "tcp" and DirectGrootRuntime.PREFETCH is False
    assert DirectGrootRuntime.CAMERAS == ["table", "side", "wrist"] and DirectGrootRuntime.IMAGE_SIZE == (640, 360)


def test_every_rollout_plans_in_prompt_then_sheet_order():
    for r in ROLLOUTS:
        steps = plan_rollout(r)
        assert [s.object_id for s in steps] == [f"M{p}_0{k}" for p in r.products for k in range(1, 5)]
        assert r.products[0] == 5  # agent A's 14:00 order: product 5 first in every rollout


def test_rollout_1_first_step_is_product_5_part_1():
    pol, _ = runtime()
    pol.act(obs(HOME_TCP))
    step = pol.progress.step
    assert pol.progress.rollout.index == 0 and (step.product, step.object_id, step.base_id) == (5, "M5_01", "F05")


def test_instruction_is_the_training_format():
    pol, _ = runtime()
    pol.act(obs(HOME_TCP))
    step = pol.progress.step
    want = instruction(encode_plan_ids("pick_insert", "M5_01", "F05", 0, "nominal"), list(step.plan_target))
    assert pol.instruction == want
    assert want == "pick M5_01 at (332, -47, 30) gripping Y at roll 210, insert on F05 at (420, 0, 45)"


def test_rollout_is_matched_by_prompt_or_forced(monkeypatch):
    pol, _ = runtime()
    pol.act(obs(HOME_TCP, prompt=ROLLOUTS[1].prompt, index=0))
    assert pol.progress.rollout.index == 1
    monkeypatch.setenv("DIRECT_ROLLOUT", "3")
    pol.reset()
    pol.act(obs(HOME_TCP, prompt=PROMPT_1))
    assert pol.progress.rollout.index == 2


def test_guard_contact_target_follows_the_step():
    pol, _ = runtime()
    pol.act(obs(HOME_TCP))
    g, s = pol.progress.step.grasp_tcp, pol.progress.step.seat_tcp
    assert (pol.guard.target.x_mm, pol.guard.target.z_contact_mm) == (g[0], g[2])
    pol.act(obs(list(g), grip=1.0))
    assert pol.progress.phase == "insert" and (pol.guard.target.x_mm, pol.guard.target.z_contact_mm) == (s[0], s[2])


def test_step_advances_on_grasp_release_at_seat_and_retreat():
    model = Echo()
    pol, _ = runtime(model)
    pol.act(obs(HOME_TCP))
    g, s = pol.progress.step.grasp_tcp, pol.progress.step.seat_tcp
    pol.act(obs(list(g), grip=1.0))                          # closed on the pad: grasped
    pol.act(obs(list(s), grip=0.0))                          # opened on the seat: released
    assert pol.progress.phase == "retreat" and pol.progress.index == 0
    pol.act(obs([s[0], s[1], s[2] + RETREAT_MM - 1, *s[3:]]))
    assert pol.progress.index == 0                           # not yet RETREAT_MM above the seat
    pol.act(obs([s[0], s[1], s[2] + RETREAT_MM, *s[3:]]))
    assert pol.progress.index == 1 and pol.progress.phase == "pick"
    assert pol.instruction.startswith("pick M5_02 ") and model.resets == 3  # trial reset + step 1 + step 2
    assert ended(pol) is None


def test_guard_refusal_holds_and_three_in_a_row_end_the_trial():
    model = Fixed([[700.0, 0.0, -90.0, -180.0, 0.0, 0.0, 0.0]])   # outside the workspace box
    pol, _ = runtime(model)
    tcp = [310.0, 5.0, -100.0, -180.0, 0.0, 0.0]
    for k in range(3):
        rows = pol.act(obs(tcp))
        np.testing.assert_allclose(rows, [tcp + [0.0]])
    assert "stuck" in ended(pol)
    rows = pol.act(obs(tcp))
    np.testing.assert_allclose(rows, [tcp + [0.0]])
    assert model.calls == 3                                  # held: the model is no longer asked


def test_pick_time_budget_holds():
    model = Fixed([HOME_TCP + [0.0]])
    pol, clock = runtime(model)
    pol.act(obs(HOME_TCP))
    clock.t = Budgets().pick_s + 0.1
    rows = pol.act(obs(HOME_TCP))
    assert "pick budget" in ended(pol) and model.calls == 1
    np.testing.assert_allclose(rows, [HOME_TCP + [0.0]])


def test_insert_time_budget_holds_with_the_part_still_held():
    pol, clock = runtime()
    pol.act(obs(HOME_TCP))
    g = list(pol.progress.step.grasp_tcp)
    pol.act(obs(g, grip=1.0))
    clock.t = Budgets().insert_s + 0.1
    rows = pol.act(obs(g, grip=1.0))
    assert "insert budget" in ended(pol)
    np.testing.assert_allclose(rows, [g + [1.0]])            # gripper stays closed: a held part stays held


def test_release_away_from_the_seat_is_a_drop():
    pol, _ = runtime()
    pol.act(obs(HOME_TCP))
    g = list(pol.progress.step.grasp_tcp)
    pol.act(obs(g, grip=1.0))
    pol.act(obs([g[0] + 30, g[1], g[2] + 40, *g[3:]], grip=0.0))
    assert "dropped or misplaced" in ended(pol)


def test_closing_away_from_the_pad_ends_the_trial():
    pol, _ = runtime()
    pol.act(obs(HOME_TCP))
    g = list(pol.progress.step.grasp_tcp)
    pol.act(obs([g[0] + 20, g[1], g[2], *g[3:]], grip=1.0))
    assert "away from the grasp pad" in ended(pol)


def test_closed_but_empty_gripper_is_a_drop():
    pol, _ = runtime()
    pol.act(obs(HOME_TCP))
    g = list(pol.progress.step.grasp_tcp)
    pol.act(obs(g, grip=1.0))
    pol.act(obs([g[0], g[1], g[2] + 20, *g[3:]], grip=1.0, gripper_width_mm=3.0))
    assert "closed but empty" in ended(pol)


def test_tcp_far_from_the_plan_ends_the_trial():
    pol, _ = runtime()
    pol.act(obs(HOME_TCP))
    pol.act(obs([300.0, 250.0, -90.0, -180.0, 0.0, 0.0]))
    assert "left the path" in ended(pol)


def test_missing_tcp_pose_holds_without_planning():
    model = Fixed([HOME_TCP + [0.0]])
    pol, _ = runtime(model)
    o = obs(HOME_TCP)
    o["tcp_pose"] = None
    rows = pol.act(o)
    assert rows.shape == (1, 7) and model.calls == 0 and pol.progress.rollout is None


def test_full_product_5_with_a_scripted_groot_in_process():
    fake = ScriptedGroot()
    pol, _ = runtime(fake)
    fake.runtime = pol
    cell = TinyCell(PROMPT_1)
    while pol.progress.index < 4 and not ended(pol):
        cell.play(pol.act(cell.obs()))
    done = [e["part"] for e in pol.events if e["type"] == "step_done"]
    assert done == ["M5_01", "M5_02", "M5_03", "M5_04"] and ended(pol) is None
    assert pol.progress.step.object_id == "M1_01"
    assert [e for e in pol.guard_log if e["held"]] == []     # the nominal path never needed a hold
    assert len(set(fake.instructions)) == 5                  # one training-format instruction per step


def test_full_product_5_through_run_py_mock(tmp_path):
    """The official run.py --mock (mock_cell.py over HTTP) with policy_direct.Policy and the scripted GR00T, in
    robot/cell's own environment (opencv, requests), as the slot will run it."""
    report = tmp_path / "report.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("DIRECT_", "VIRTUAL_ENV", "GROOT_"))}
    proc = subprocess.run(["uv", "run", "--project", str(REPO / "robot" / "cell"), "python",
                           str(Path(__file__).with_name("direct_mock_driver.py")), "60", str(report)],
                          cwd=REPO / "robot" / "cell", capture_output=True, text=True, timeout=300, env=env, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "Ready for the real robot" in proc.stdout
    data = json.loads(report.read_text())[0]
    done = [e["part"] for e in data["events"] if e["type"] == "step_done"]
    assert done[:4] == ["M5_01", "M5_02", "M5_03", "M5_04"], proc.stdout[-3000:]
    holds = [g for g in data["guard_log"] if g["held"] and "tcp_pose missing" not in g["reasons"][0]]
    assert holds == []


def test_a_chunk_that_closes_on_the_pad_then_lifts_is_a_grasp():
    """GR00T chunks run ~3 s: the close row is on the pad, the rows after it already lift the part."""
    g = None

    class CloseThenLift(Echo):
        def predict(self, model_obs):
            if g is None:
                return super().predict(model_obs)
            self.calls += 1
            lift = [[g[0], g[1], g[2] + dz, *g[3:6], 1.0] for dz in (2.0, 4.0, 6.0, 26.0, 46.0)]  # slow, then fast
            return np.array([[*g[:6], 0.0], [*g[:6], 1.0], *lift])

    pol, _ = runtime(CloseThenLift())
    pol.act(obs(HOME_TCP))
    g = list(pol.progress.step.grasp_tcp)
    rows = pol.act(obs(g))                                    # at the pad, open: the model closes, then lifts
    assert pol.guard_log == [] and rows[-1, 2] == g[2] + 46.0
    pol.act(obs(list(rows[-1, :6]), grip=1.0))                # observed 46 mm above the pad, closed
    assert ended(pol) is None and pol.progress.phase == "insert"


def test_a_chunk_that_opens_on_the_seat_then_rises_finishes_the_step():
    s = None

    class OpenThenRise(Echo):
        def predict(self, model_obs):
            self.calls += 1
            if float(model_obs["joints"][6]) < 0.5:
                return super().predict(model_obs) if s is None else np.array([[*model_obs["tcp_pose"][:6], 0.0]])
            return np.array([[*s[:6], 1.0], [*s[:6], 0.0]]
                            + [[s[0], s[1], s[2] + 2.0 * k, *s[3:6], 0.0] for k in range(1, 7)])

    pol, _ = runtime(OpenThenRise())
    pol.act(obs(HOME_TCP))
    g, s = list(pol.progress.step.grasp_tcp), list(pol.progress.step.seat_tcp)
    pol.act(obs(g, grip=1.0))
    rows = pol.act(obs(s, grip=1.0))                          # on the seat, closed: open, then rise 12 mm
    pol.act(obs(list(rows[-1, :6]), grip=0.0))
    assert ended(pol) is None and pol.progress.index == 1


def test_cell_entry_point_fits_both_run_py_versions():
    """robot/cell/policy_direct.Policy: upstream run.py calls Policy(checkpoint); PR #24's calls
    Policy(checkpoint, runtime=...) and then close()."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("policy_direct", REPO / "robot" / "cell" / "policy_direct.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert issubclass(mod.Policy, DirectGrootRuntime) and mod.Policy.ACTION == "tcp"
    pol = mod.Policy(None, runtime="cell", model=Echo())
    pol.close()
    assert pol.runtime == "cell" and mod.INSTANCES[-1] is pol
