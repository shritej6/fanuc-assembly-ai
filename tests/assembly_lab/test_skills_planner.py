"""Agent 2's planner -> skill plans -> rows, for all 24 parts in sheet order (host-side, no physics)."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from sim.assembly_lab.objects import planner
from sim.assembly_lab.skills import backends, frames
from sim.assembly_lab.skills.motion import WORKSPACE, rows_of
from sim.assembly_lab.skills.plan import SEAT_FLOOR_MM

pick_mod = importlib.import_module("sim.assembly_lab.skills.pick")
insert_mod = importlib.import_module("sim.assembly_lab.skills.insert")
HOME = np.array([400.0, 0.0, 100.0, -180.0, 0.0, 0.0])


@pytest.mark.parametrize("product", [1, 2, 3, 4, 5, 6])
def test_every_part_plans_and_obeys_the_rules(product):
    rng = np.random.default_rng(product)
    base = backends.base_of(product)
    ids = [base, *backends.sheet_parts(product)]
    installed: dict[str, str] = {}
    for part in backends.sheet_parts(product):
        layout = backends.sample_layout(rng, ids, installed)
        scene = {"objects": layout["objects"], "spares": []}
        g = planner.plan_pick(scene, part)
        p = planner.plan_place(scene, part, base, flip=g.flip)
        assert g.feasible and p.feasible, (part, g.reason, p.reason)
        pk, ins = backends.to_pick(g), backends.to_insert(p, part, base)
        entry = frames.catalog()[part]["insertion"]
        assert ins.capture_mm == pytest.approx(entry["xy_capture_mm"])
        assert ins.yaw_tol_deg == entry["yaw_tol_deg"]
        rows = np.vstack([rows_of(pick_mod.pick(pk, HOME)), rows_of(insert_mod.insert(ins, HOME, 1.0))])
        for k, (lo, hi) in enumerate(WORKSPACE):
            assert lo <= rows[:, k].min() and rows[:, k].max() <= hi
        assert rows[-1, 2] >= ins.seated_tcp[2] - SEAT_FLOOR_MM
        installed[part] = base


@pytest.mark.parametrize("oid", ["M1_04", "M4_02", "F04"])
def test_body_to_local_matches_the_planner_frame(oid):
    from sim.assembly_lab.env.layout import LayoutSolver, pose6_from_tf

    body = pose6_from_tf(LayoutSolver().table_tf(oid, 400.0, 50.0, 30.0, "installed"))
    local = frames.body_to_local(oid, body)
    assert frames.standing(local)
    assert frames.heading_deg(local) == pytest.approx(30.0, abs=1e-6)
    assert local[2] == pytest.approx(-335.0 - frames.catalog()[oid]["bbox_mm"]["min"][2], abs=1e-6)


def test_scene_from_obs_marks_installed_and_fallen():
    obs = {"sim": {"object_poses": {"F01": [400, 0, -327, 0, 0, 10], "M1_01": [400, 0, -320, 0, 0, 10],
                                    "M1_02": [300, 100, -335, 90, 0, 0]}}}
    scene = frames.scene_from_obs(obs, {"M1_01": "F01"})
    by = {o["key"]: o for o in scene["objects"]}
    assert by["M1_01"]["state"] == "on:F01"
    assert by["M1_02"]["state"] == "fallen"
    assert by["F01"]["state"] == "installed"
