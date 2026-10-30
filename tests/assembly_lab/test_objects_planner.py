"""T12 agent 2: the grasp / place / pull-out planner on crowded and ordered layouts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sim.assembly_lab.objects.planner import (
    blockers,
    j6_estimate,
    plan_pick,
    plan_place,
    plan_pull_out,
    roll_candidates,
)

ROOT = Path(__file__).resolve().parents[2]


def product4(base_yaw=10.0):
    return {"objects": [{"id": "F04", "x": 380, "y": 120, "yaw": base_yaw, "state": "installed"},
                        {"id": "M4_01", "x": 450, "y": -100, "yaw": 30},
                        {"id": "M4_02", "x": 300, "y": -50, "yaw": -45},
                        {"id": "M4_03", "x": 550, "y": 50, "yaw": 100},
                        {"id": "M4_04", "x": 520, "y": -200, "yaw": 0}]}


def _set(scene, oid, **kw):
    next(o for o in scene["objects"] if o["id"] == oid).update(kw)


def test_full_build_then_dismantle_in_sheet_order():
    scene = product4()
    for oid in ("M4_01", "M4_02", "M4_03", "M4_04"):
        pick = plan_pick(scene, oid)
        assert pick.feasible, (oid, pick.reason)
        place = plan_place(scene, oid, "F04", flip=pick.flip, current_r=pick.r_deg)
        assert place.feasible, (oid, place.reason)
        assert place.guard_floor_z == pytest.approx(place.tcp[2] - 1.0)
        assert place.contact_z > place.tcp[2] and place.hover_z == pytest.approx(place.contact_z + 5.0)
        assert place.path[-4][6] == 1.0 and place.path[-3][6] == 0.0        # release only at the seat
        _set(scene, oid, state="on:F04")
    for i, oid in enumerate(("M4_04", "M4_03", "M4_02", "M4_01")):
        out = plan_pull_out(scene, oid)
        assert out.feasible, (oid, out.reason)
        _set(scene, oid, state="installed", x=600, y=300 - 80 * i, yaw=0)


def test_place_pose_is_base_pose_plus_installed_origin():
    scene = product4(base_yaw=90.0)
    plan = plan_place(scene, "M4_01", "F04")
    assert plan.target_pose[:2] == pytest.approx([380.0, 120.0]) and plan.target_pose[3] == 90.0
    scene["objects"][1]["state"] = "on:F04"
    lock_order = plan_place(scene, "M4_02", "F04")
    assert lock_order.feasible
    # housing grip: closing along local Y (roll offset 0); r = base yaw + 0 or + 180
    assert (plan.r_deg - 90.0) % 180.0 == pytest.approx(0.0)


def test_out_of_order_wrong_type_and_wrong_product_are_refused():
    scene = product4()
    assert "sheet order" in plan_place(scene, "M4_03", "F04").reason
    scene["objects"].append({"id": "F01", "x": 250, "y": 250, "yaw": 0})
    assert "wrong base type" in plan_place(scene, "M4_01", "F01").reason
    scene["objects"].append({"id": "M5_01", "state": "on:F04"})
    assert "not product 4" in plan_place(scene, "M4_02", "F04").reason
    assert "already on" in plan_place(scene, "M4_01", "F04").reason


def test_bases_never_move_fallen_and_unknown_are_not_picked():
    scene = product4()
    _set(scene, "M4_03", state="fallen")
    scene["objects"].append({"key": "junk", "id": "UNKNOWN", "x": 300, "y": 200, "yaw": 0, "bbox_mm": [20, 20, 10]})
    assert plan_pick(scene, "F04").reason == "bases never move"
    assert plan_pick(scene, "M4_03").reason.startswith("fallen")
    assert plan_pick(scene, "junk").reason.startswith("unknown")


def test_lock_beside_insert_blocks_both_rolls_and_is_the_blocker():
    scene = {"objects": [{"id": "M4_02", "x": 400, "y": 0, "yaw": 0},
                         {"id": "M5_04", "x": 412, "y": 0, "yaw": 90}]}
    plan = plan_pick(scene, "M4_02")
    assert not plan.feasible and plan.blocked_by == {0: ["M5_04"], 1: ["M5_04"]}
    assert blockers(scene, "M4_02") == ["M5_04"]


def test_neighbour_along_the_pad_is_not_a_blocker():
    # insert gripped across X; a lock 25 mm away along Y is outside the finger footprint
    scene = {"objects": [{"id": "M4_02", "x": 400, "y": 0, "yaw": 0},
                         {"id": "M5_04", "x": 400, "y": 25, "yaw": 0}]}
    assert plan_pick(scene, "M4_02").feasible and blockers(scene, "M4_02") == []


def test_low_neighbour_under_the_fingertips_is_not_a_blocker():
    # a flat guide (13 mm tall) next to a guard gripped 27.9 mm up: fingertips pass above it
    scene = {"objects": [{"id": "M1_04", "x": 400, "y": 0, "yaw": 0},
                         {"id": "M4_03", "x": 420, "y": 0, "yaw": 0}]}
    assert plan_pick(scene, "M1_04").feasible


def test_crowded_ring_lists_every_blocker():
    scene = {"objects": [{"id": "M4_02", "x": 400, "y": 0, "yaw": 0},
                         {"id": "M5_04", "x": 412, "y": 0, "yaw": 90},
                         {"id": "M6_04", "x": 388, "y": 0, "yaw": 90}]}
    assert blockers(scene, "M4_02") == ["M5_04", "M6_04"]


def test_stacked_parts_block_in_reverse_sheet_order():
    scene = product4()
    for oid in ("M4_01", "M4_02", "M4_03", "M4_04"):
        _set(scene, oid, state="on:F04")
    assert blockers(scene, "M4_02") == ["M4_04", "M4_03"]
    assert plan_pull_out(scene, "M4_02").reason.startswith("parts above")


def test_place_refused_when_a_foreign_object_sits_in_the_way():
    scene = product4()
    scene["objects"].append({"id": "M1_04", "x": 380, "y": 125, "yaw": 0})   # a guard standing on the empty base
    plan = plan_place(scene, "M4_01", "F04")
    assert not plan.feasible and plan.blocked_by[0] == ["M1_04"]


def test_roll_is_nearest_the_current_r_and_both_grasps_are_offered():
    cands = roll_candidates(yaw=30.0, offset=90.0, current_r=-170.0)
    # k=0: 120 -> -240 (70 deg away); k=1: 300 -> -60 (110 deg away)
    assert cands == [(0, pytest.approx(-240.0)), (1, pytest.approx(-60.0))]
    plan = plan_pick({"objects": [{"id": "M4_02", "x": 400, "y": 0, "yaw": 30}]}, "M4_02", current_r=120.0)
    assert plan.r_deg == pytest.approx(120.0) and plan.flip == 0


def test_j6_limit_forces_the_other_grasp():
    scene = {"objects": [{"id": "M4_02", "x": 400, "y": 0, "yaw": 0}]}
    plan = plan_pick(scene, "M4_02", current_r=180.0, current_j6=345.0)      # wound up near +360
    # k=1 (r = 270) would need J6 = 435; k=0 is r = 90 -> J6 = 255
    assert plan.feasible and plan.flip == 0 and plan.j6_estimate_deg == pytest.approx(255.0)
    assert plan.blocked_by == {1: ["J6_LIMIT"]} or plan.blocked_by == {}


def test_workspace_box_is_enforced():
    plan = plan_pick({"objects": [{"id": "M4_02", "x": 100, "y": 0, "yaw": 0}]}, "M4_02")
    assert not plan.feasible and "workspace" in plan.reason


def test_j6_formula_matches_the_twin_forward_kinematics():
    sys.path.insert(0, str(ROOT / "sim" / "starter"))
    fk = pytest.importorskip("fk")
    for q1, q6 in ((0.0, 0.0), (30.0, -50.0), (-60.0, 200.0)):
        q = np.radians([q1, 20.0, -30.0, 0.0, -40.0, q6])           # q2 - q3 - q5 = 90: tool straight down
        rot, _ = fk.fk(q)
        assert rot @ np.array([1.0, 0, 0]) == pytest.approx([0, 0, -1], abs=1e-9)
        spread = rot @ np.array([0.0, 1.0, 0.0])                    # FINGER_SPREAD_AXIS = rot(w,p,r) @ +y
        r = math.degrees(math.atan2(spread[0], -spread[1]))         # (sin r, -cos r)
        assert (r - q1 - q6 - 180.0) % 360.0 == pytest.approx(0.0, abs=1e-6) or \
            (r - q1 - q6 - 180.0) % 360.0 == pytest.approx(360.0, abs=1e-6)
    # continuation: turning r by +90 at the same xy turns J6 by +90
    assert j6_estimate(270.0, 400.0, 0.0, (180.0, 400.0, 0.0, 0.0, 0.0)) == pytest.approx(90.0)
