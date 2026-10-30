"""Precision gates (INTERFACES.md) and unseen-layout generation for the evaluator."""
from __future__ import annotations

import sys

import numpy as np
import pytest

from sim.assembly_lab.eval import cad, layouts
from sim.assembly_lab.eval.run_eval import REPO

BASE = np.array([400.0, 50.0, -327.0, 0.0, 0.0, 30.0])


def seated(pid: str, base: str = "F04") -> np.ndarray:
    return cad.seated_pose(pid, base, BASE)


def moved(pose, dx=0.0, dz=0.0, dyaw=0.0) -> np.ndarray:
    t = cad.pose_to_matrix([0, 0, 0, 0, 0, dyaw]) @ cad.pose_to_matrix(pose)
    out = cad.matrix_to_pose(t)
    out[:3] = np.asarray(pose[:3]) + [dx, 0.0, dz]
    return out


@pytest.mark.parametrize("dx,dz,dyaw,ok", [(0.29, 0, 0, True), (0.31, 0, 0, False), (0, 0.29, 0, True),
                                           (0, -0.31, 0, False), (0, 0, 0.9, True), (0, 0, 1.1, False)])
def test_precision_gates_xy_height_yaw(dx, dz, dyaw, ok):
    s = seated("M4_02")  # symmetry "every 360 deg"
    a = moved(s, dx=dx, dz=dz)
    if dyaw:  # turn about the part's own vertical axis, so only yaw changes
        a = cad._turned_about_part_axis("M4_02", s, dyaw)
    assert cad.seat_check("M4_02", a, s)[0] is ok


def test_tilt_over_1deg_fails():
    s = seated("M4_02")
    a = s.copy()
    a[3] += 1.2
    assert not cad.seat_check("M4_02", a, s)[0]


def test_symmetric_part_turned_by_its_symmetry_is_seated():
    pid = next(p for p, t in cad.tolerances().items() if t["symmetry"] == "every 180 deg")
    base = cad.base_type(pid[:2])
    s = cad.seated_pose(pid, base, BASE)
    assert cad.seat_check(pid, cad._turned_about_part_axis(pid, s, 180.0), s)[0]
    one_way = next(p for p, t in cad.tolerances().items() if t["symmetry"] == "every 360 deg")
    s1 = cad.seated_pose(one_way, cad.base_type(one_way[:2]), BASE)
    assert not cad.seat_check(one_way, cad._turned_about_part_axis(one_way, s1, 180.0), s1)[0]


def test_tilt_is_yaw_independent_for_parts_whose_stl_z_is_not_up():
    p = cad.table_pose("M1_04", 300.0, 0.0, 0.0)  # M1_04's STL z axis lies horizontal when it stands
    q = cad.table_pose("M1_04", 300.0, 0.0, 90.0)
    assert cad.tilt_between(p, q) == pytest.approx(0.0, abs=1e-6)


def test_eval_seeds_are_disjoint_from_mimic_and_reproducible():
    raw, scs = layouts.load()
    seeds = [layouts.build(raw, sc, k).seed for sc in scs for k in range(sc.layouts)]
    assert len(set(seeds)) == len(seeds) and int(raw["mimic_seed"]) not in seeds
    sc = scs[0]
    assert layouts.build(raw, sc, 1).layout == layouts.build(raw, sc, 1).layout


def test_every_case_of_robot_brain_section_3_is_covered():
    _, scs = layouts.load()
    assert set(layouts.CASES) == {sc.case for sc in scs}
    assert {p for sc in scs if sc.case == "nominal" for p in sc.products} == set(cad.PRODUCTS)


def test_layouts_are_accepted_by_the_env_solver():
    sys.path.insert(0, str(REPO / "sim/assembly_lab/env"))
    import layout as env_layout

    solver = env_layout.LayoutSolver()
    pool = env_layout.default_pool(solver.geometry, 1)
    for ev in layouts.all_layouts():
        solver.solve(ev.layout, pool, -400.0)


def test_crowded_layouts_are_blocked_per_the_finger_model():
    from sim.assembly_lab.objects.planner import blockers

    for ev in layouts.all_layouts(only=["crowded"]):
        lab = ev.layout["eval"]["labels"]
        scene = {"objects": [dict(o, key=o["uid"]) for o in ev.layout["objects"]]}
        assert blockers(scene, lab["crowded_target"]) == [lab["blocker"]]


def test_grip_tcp_is_the_flange_axis_not_the_pad_point():
    """pad_point_local_mm is the pad point; the commanded TCP is 1.8 mm off it along the roll (objects/planner.py)."""
    from sim.assembly_lab.objects.planner import CONTACT_V_MM, plan_pick, v_dir

    pose = cad.table_pose("M4_02", 350.0, 40.0, 37.0)
    tcp = cad.grip_tcp("M4_02", pose, 0, 180.0)
    pad = cad.matrix_to_pose(cad.pose_to_matrix(pose) @ cad.pose_to_matrix(
        [*cad.grasp_frames()["M4_02"]["grips"][0]["pad_point_local_mm"], 180.0, 0.0, 0.0]))
    np.testing.assert_allclose(pad[:2] - tcp[:2], CONTACT_V_MM * v_dir(tcp[5]), atol=1e-6)
    plan = plan_pick({"objects": [{"id": "M4_02", "key": "M4_02", "x": 350.0, "y": 40.0, "yaw": 37.0}]}, "M4_02",
                     current_r=180.0)
    np.testing.assert_allclose(tcp[:3], plan.tcp[:3], atol=0.01)
    assert tcp[5] == pytest.approx(plan.r_deg, abs=0.01)


@pytest.mark.parametrize("pid", ["M4_02", "M1_04", "M6_01", "M2_03"])
@pytest.mark.parametrize("yaw,current_r", [(0.0, 180.0), (37.0, 0.0), (-120.0, 90.0)])
def test_eval_tcp_equals_grasp_frames_tcp_for_the_turn_used(pid, yaw, current_r):
    """grasp_frames.json tcp_local_mm[turn] (T29, per wrist turn) is the commanded TCP; eval must land on it."""
    g = cad.grasp_frames()[pid]
    grip = g["grips"][0]
    pose = cad.table_pose(pid, 350.0, 40.0, yaw)
    tcp = cad.grip_tcp(pid, pose, 0, current_r)
    local = cad.pose_to_matrix(pose) @ np.linalg.inv(np.array(g["T_local_from_stl"]))
    local_yaw = np.degrees(np.arctan2(local[1, 0], local[0, 0]))
    turn = round(cad.wrap180(tcp[5] - local_yaw))
    turn = 180 if turn == -180 else turn
    assert str(turn) in grip["tcp_local_mm"], (turn, list(grip["tcp_local_mm"]))
    want = local @ np.array([*grip["tcp_local_mm"][str(turn)], 1.0])
    np.testing.assert_allclose(tcp[:3], want[:3], atol=1e-6)


@pytest.mark.parametrize("pid", [p for p in cad.geometry() if not p.startswith("F")])
def test_seated_pose_matches_env_layout_seat(pid):
    """Regression (agent 1, 3bb7db3): seat = base_pose @ installed_in_base_frame, no extra base rest offset."""
    sys.path.insert(0, str(REPO / "sim/assembly_lab/env"))
    import layout as env_layout

    base, base_pose = "F01", [400.0, 10.0, -300.0, 0.0, 0.0, 37.0]
    want = env_layout.LayoutSolver().seat_tf_for(pid, base, {base: cad.pose_to_matrix(base_pose)})
    np.testing.assert_allclose(cad.pose_to_matrix(cad.seated_pose(pid, base, base_pose)), want, atol=1e-6)


@pytest.mark.parametrize("pid", [p for p, o in cad.geometry().items() if str(o.get("install_step")) == "1"])
def test_first_part_sits_on_the_base_seating_face(pid):
    """Base STL z = 0 is the seating face: the first part's lowest point is there, not 8 mm inside the base."""
    lo, hi = np.array(cad.geometry()[pid]["bounds_mm"], float)
    corners = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    base_pose = [400.0, 10.0, -300.0, 0.0, 0.0, 37.0]
    world = cad.pose_to_matrix(cad.seated_pose(pid, "F01", base_pose)) @ corners.T
    assert world[2].min() == pytest.approx(base_pose[2], abs=0.05)
