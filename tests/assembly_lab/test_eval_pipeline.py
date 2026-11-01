"""Evaluator end to end on a kinematic fake env: layouts -> director -> guarded policy -> KPIs -> gates.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

from policy_learn.deploy.groot_policy import GrootPolicy
from sim.assembly_lab.eval import cad, kpis, layouts
from sim.assembly_lab.eval.director import Director
from sim.assembly_lab.eval.run_eval import REPO, aggregate, missing_requirement, run_episode
from sim.assembly_lab.eval.scripted import ScriptedOracleModel

sys.path.insert(0, str(REPO / "sim/assembly_lab/env"))
import layout as env_layout

HOME = np.array([400.0, 0.0, 100.0, -180.0, 0.0, 0.0])


class FakeEnv:
    def __init__(self, push_mm: float = 0.0):
        self.solver = env_layout.LayoutSolver()
        self.pool = env_layout.default_pool(self.solver.geometry, 1)
        self.push_mm = push_mm  # > 0: shove a non-target object this far during the first carry

    def reset(self, layout):
        placed = self.solver.solve(layout, self.pool, -400.0)
        self.poses = {p.uid: np.array(env_layout.pose6_from_tf(p.tf_uf0)) for p in placed if p.role != "parked"}
        self.tcp, self.grip, self.held, self.pushed = HOME.copy(), 0.0, None, False
        return self.obs()

    def obs(self):
        img = np.zeros((36, 64, 3), np.uint8)
        return {"images": {"table": img, "side": img, "wrist": img}, "joints": np.array([0.0] * 6 + [self.grip]),
                "tcp_pose": self.tcp.tolist(), "instruction": "",
                "sim": {"object_poses": {u: p.tolist() for u, p in self.poses.items()}, "contact": {},
                        "contact_peak": {u: {"normal_N": 1.0, "lateral_N": 0.5, "torque_Nm": 0.01}
                                         for u in self.poses},
                        "gripper_obj": self.held[0] if self.held else None}}

    def step_tcp(self, rows, dt):
        for row in np.asarray(rows, float):
            self.tcp = row[:6].copy()
            self.tcp[[3, 5]] = (self.tcp[[3, 5]] + 180.0) % 360.0 - 180.0  # the cell reports w, r in [-180, 180)
            closing, opening = row[6] > 0.5 >= self.grip, row[6] <= 0.5 < self.grip
            self.grip = float(row[6] > 0.5)
            if closing and self.held is None:
                for uid, pose in self.poses.items():
                    if uid.startswith("M") and np.linalg.norm(cad.grip_tcp(uid, pose, 0, self.tcp[5])[:3] - self.tcp[:3]) <= 1.5:
                        rel = np.linalg.inv(cad.pose_to_matrix(self.tcp)) @ cad.pose_to_matrix(pose)
                        self.held = (uid, rel)
                        break
            if self.held:
                uid, rel = self.held
                self.poses[uid] = cad.matrix_to_pose(cad.pose_to_matrix(self.tcp) @ rel)
                if self.push_mm and not self.pushed:
                    other = next(u for u in self.poses if u.startswith("F"))
                    self.poses[other][0] += self.push_mm
                    self.pushed = True
                if opening:
                    self.held = None
        return self.obs()


def episode(scenario: str, env=None, layout_k: int = 0):
    raw, scs = layouts.load()
    sc = next(s for s in scs if s.id == scenario)
    ev = layouts.build(raw, sc, layout_k)
    director = Director.for_layout(ev, float(raw["task_time_s"]))
    policy = GrootPolicy(model=ScriptedOracleModel(lambda: director))
    return run_episode(env or FakeEnv(), policy, ev, sc.time_limit_s, kpis.load_limits(), director, None)


@pytest.mark.parametrize("scenario", ["nominal_M4", "nominal_M1", "damaged_part", "crowded_blocker",
                                      "fix_wrong_part", "lookalike_by_size", "missing_then_spare"])
def test_scripted_oracle_passes_in_the_kinematic_double(scenario):
    res = episode(scenario)
    failed = {k: v for k, v in res["gates"].items() if not v["ok"]}
    assert res["success"], (failed, res["failed"], res["kpi"]["disturbed"], res["guard"])
    assert res["guard"]["held"] == 0


def test_no_spare_raises_andon_and_still_rejects_the_bad_part():
    res = episode("damaged_no_spare")
    assert res["gates"]["andon"]["ok"] and res["gates"]["andon"]["raised"]
    assert all(v["ok"] for k, v in res["gates"].items() if k.startswith("defect_area"))


def test_disturbed_non_target_fails_the_episode():
    res = episode("nominal_M4", env=FakeEnv(push_mm=1.5))
    assert not res["success"] and res["kpi"]["disturbed"] and res["kpi"]["collisions"] >= 1


def test_over_limit_force_counts_as_damage():
    class Hard(FakeEnv):
        def obs(self):
            o = super().obs()
            o["sim"]["contact_peak"] = {"M4_02": {"normal_N": 16.0, "lateral_N": 0.0, "torque_Nm": 0.0}}
            return o

    res = episode("nominal_M4", env=Hard())
    assert not res["success"] and res["kpi"]["parts_damaged"] == 1 and res["kpi"]["peak_axial_N"] == 16.0


def test_robot_contact_unmeasured_is_reported_not_passed():
    res = episode("nominal_M4")
    assert res["kpi"]["robot_contacts"] == "UNMEASURED" and res["success_note"]


def test_robot_touch_on_a_non_target_is_a_collision():
    class Touchy(FakeEnv):
        def obs(self):
            o = super().obs()
            o["sim"]["robot_touch"] = {"F05": 0.2, "F04": 0.2, "F06": 0.2}
            return o

    res = episode("nominal_M4", env=Touchy())
    assert not res["success"] and res["kpi"]["robot_contacts"]


def test_requirements_skip_instead_of_pass():
    assert "inject_foreign" in missing_requirement(["env:inject_foreign"], FakeEnv(), False)
    assert "T09" in missing_requirement(["director:t09"], FakeEnv(), False)
    assert missing_requirement(["director:t09"], FakeEnv(), True) is None


def test_aggregate_counts_skips_separately():
    rows = [{"scenario": "a", "case": "nominal", "success": True}, {"scenario": "a", "case": "nominal",
                                                                     "success": False},
            {"scenario": "b", "case": "foreign", "skipped": "env", "success": False}]
    agg = aggregate(rows, {"a": ["M1"], "b": ["M3"]})
    assert agg["per_product"]["M1"] == {"pass": 1, "ran": 2, "skipped": 0, "rate": 0.5, "jams": 0, "damaged": 0,
                                        "collisions": 0}
    assert agg["per_case"]["foreign"]["rate"] is None and agg["overall"]["skipped"] == 1


def test_base_sliding_during_insertion_fails_the_run():
    """The base being inserted into is a non-target: a 1.5 mm slide fails the 'nothing else moved' gate."""
    res = episode("nominal_M4", env=FakeEnv(push_mm=1.5))
    assert any(d["uid"].startswith("F") for d in res["kpi"]["disturbed"]) and not res["success"]


def test_per_part_force_limit_from_objects_yaml():
    """M5_04's axial limit is 2.5 N (objects.yaml max_force), far under the global 15 N."""
    lim = kpis.load_limits()
    assert kpis.limit_for(lim, "M5_04#2", "axial_N") == 2.5 and kpis.limit_for(lim, "M4_02", "axial_N") == 15.0

    class Lock(FakeEnv):
        def obs(self):
            o = super().obs()
            o["sim"]["contact_peak"] = {"M5_04": {"normal_N": 3.0, "lateral_N": 0.0, "torque_Nm": 0.0}}
            return o

    res = episode("nominal_M5", env=Lock())
    assert res["kpi"]["damaged"]["M5_04"]["limit"] == 2.5 and not res["success"]


def test_jam_is_a_part_stuck_high_on_its_mate():
    base = np.array([400.0, 0.0, -327.0, 0.0, 0.0, 0.0])
    seat = cad.seated_pose("M4_01", "F04", base)
    high = seat.copy()
    high[2] += 1.5
    assert kpis.jam_check("M4_01", "F04", {"F04": base, "M4_01": high})["dz"] == pytest.approx(1.5, abs=1e-3)
    assert kpis.jam_check("M4_01", "F04", {"F04": base, "M4_01": seat}) is None


def test_scripted_plan_across_r_180_passes_the_guard():
    """Regression (Isaac smoke): r -179.9 -> 180.x is a tiny turn; through the adapter it must not ask for 360 deg."""
    from types import SimpleNamespace

    from policy_learn.deploy.groot_policy import nearest_turns
    from policy_learn.deploy.guards import ChunkGuard
    from sim.assembly_lab.eval.scripted import ScriptedOracleModel

    start = np.array([300.0, 0.0, -90.0, -180.0, 0.0, -179.9])
    task = SimpleNamespace(uid="M1_01", skill="pick_insert", attempts=0, started_s=0.0,
                           grasp_tcp=np.array([300.0, 20.0, -200.0, -180.0, 0.0, 179.6]),
                           goal_tcp=np.array([400.0, 20.0, -200.0, -180.0, 0.0, 179.6]))
    model = ScriptedOracleModel(lambda: SimpleNamespace(current=task))
    tcp = start
    for _ in range(3):
        chunk = nearest_turns(model.predict({"tcp_pose": tcp, "joints": [0.0] * 7}), tcp)
        res = ChunkGuard().guard(chunk, tcp, 0.0, 0.2)
        assert not res.held, res.reasons
        tcp = res.rows[-1, :6].copy()
        tcp[5] = (tcp[5] + 180.0) % 360.0 - 180.0  # the cell reports r in [-180, 180)
