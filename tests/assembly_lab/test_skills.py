"""Skill motion rules (docs/ROBOT_BRAIN.md section 2): every row is checked, not just the endpoints."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from sim.assembly_lab.skills import dry_run, geometry
from sim.assembly_lab.skills.insert import insert, retry_entry, retry_lift
from sim.assembly_lab.skills.motion import SLOW, WORKSPACE, Motion, MotionError, rows_of
from sim.assembly_lab.skills.pick import pick
from sim.assembly_lab.skills.place_free import place_free
from sim.assembly_lab.skills.plan import (
    INSERT_TILT_DEG,
    INSERT_TWIST_DEG,
    PULL_WIGGLE_DEG,
    SEAT_FLOOR_MM,
    SLOW_ZONE_MM,
    PickPlan,
    PlanError,
    pick_from_planner,
)
from sim.assembly_lab.skills.pull_out import pull_out

HOME = np.array(dry_run.HOME_TCP)


@pytest.fixture
def plans():
    objs, seats = dry_run.demo_world()
    return dry_run.demo_plans(objs, seats)


def _steps(rows):
    d = np.linalg.norm(np.diff(rows[:, :3], axis=0), axis=1)
    rot = np.array([geometry.rotation_distance_deg(a, b) for a, b in itertools.pairwise(rows)])
    return d, rot


def _in_workspace(rows):
    for k, (lo, hi) in enumerate(WORKSPACE):
        assert (rows[:, k] >= lo).all() and (rows[:, k] <= hi).all()


def test_wpr_roundtrip_and_tilt():
    for pose in ([0, 0, 0, -180, 0, 37], [1, 2, 3, 10, -20, 170], [0, 0, 0, -177, 2, -90]):
        back = geometry.matrix_to_pose(geometry.pose_to_matrix(pose))
        assert np.allclose(geometry.pose_to_matrix(back), geometry.pose_to_matrix(pose), atol=1e-9)
    down = np.array([0, 0, 0, -180, 0, 25.0])
    assert geometry.tilt_deg(geometry.tilted(down, 40.0, 3.0)) == pytest.approx(3.0, abs=1e-6)
    twisted = geometry.tilted(down, 0.0, 0.0, 2.0)
    assert geometry.rotation_distance_deg(down, twisted) == pytest.approx(2.0, abs=1e-6)


def test_tilt_lowers_the_lead_side():
    down = np.array([0, 0, 0, -180, 0, 0.0])
    t = geometry.pose_to_matrix(geometry.tilted(down, 90.0, 3.0))  # lead side +y
    # a point 10 mm toward +y under the TCP, rigidly attached, must end lower than its mirror at -y
    plus = t @ np.linalg.inv(geometry.pose_to_matrix(down)) @ np.array([0, 10, -10, 1])
    minus = t @ np.linalg.inv(geometry.pose_to_matrix(down)) @ np.array([0, -10, -10, 1])
    assert plus[2] < minus[2]


def test_pick_corridor_slow_zone_and_grip(plans):
    p = plans[0]
    segs = pick(p, HOME)
    rows = rows_of(segs)
    _in_workspace(rows)
    assert rows[:, 2].min() >= p.grasp_tcp[2] - SEAT_FLOOR_MM - 1e-9
    phases = [s.phase for s in segs]
    assert phases.index("open") < phases.index("descend") < phases.index("close") < phases.index("lift_check")
    descend = next(s for s in segs if s.phase == "descend").rows
    assert np.allclose(descend[:, :2], p.grasp_tcp[:2], atol=1e-6), "descent must be straight down the corridor"
    assert (descend[:, 6] == 0).all()
    d, _ = _steps(np.vstack([segs[phases.index("descend") - 1].rows[-1:], descend]))
    slow = descend[1:, 2] <= p.grasp_tcp[2] + SLOW_ZONE_MM + 1e-9
    assert (d[1:][slow[: len(d) - 1]] <= SLOW.mm + 1e-6).all()
    assert rows[-1, 6] == 1 and rows[-1, 2] == pytest.approx(p.lift_z_mm)


def test_insert_is_human_style_and_never_below_floor(plans):
    ins = plans[1]
    start = np.r_[ins.seated_tcp[:2] + 60, ins.approach_z_mm + 20, ins.seated_tcp[3:]]
    segs = insert(ins, start, 1.0)
    rows = rows_of(segs)
    _in_workspace(rows)
    zs = ins.seated_tcp[2]
    assert rows[:, 2].min() >= zs - SEAT_FLOOR_MM - 1e-9
    assert rows[:, 2].min() <= zs, "the light push must reach the seated height"
    assert (rows[:, 6] == 1).all(), "the part is held for the whole insert"
    order = [s.phase for s in segs]
    for a, b in (("tilt", "touch"), ("touch", "level"), ("level", "twist"), ("twist", "push")):
        assert order.index(a) < order.index(b)
    touch = next(s for s in segs if s.phase == "touch").rows
    assert geometry.tilt_deg(touch[-1]) == pytest.approx(INSERT_TILT_DEG, abs=0.05)
    twist = np.vstack([s.rows for s in segs if s.phase == "twist"])
    yaw = np.array([geometry.wrap_deg(r - ins.seated_tcp[5]) for r in twist[:, 5]])
    assert yaw.max() == pytest.approx(INSERT_TWIST_DEG, abs=0.05)
    assert yaw.min() == pytest.approx(-INSERT_TWIST_DEG, abs=0.05)
    d, rot = _steps(rows)
    slow = (rows[1:, 2] <= zs + SLOW_ZONE_MM) & (rows[:-1, 2] <= zs + SLOW_ZONE_MM)
    assert (d[slow] <= SLOW.mm + 1e-6).all() and (rot[slow] <= SLOW.deg + 1e-6).all(), "slow last 5 mm"


def test_retry_lifts_3mm_then_reenters(plans):
    ins = plans[1]
    pushed = np.r_[ins.seated_tcp[:2], ins.seated_tcp[2] + 1.0, ins.seated_tcp[3:]]
    lift = rows_of(retry_lift(ins, pushed, 1.0))
    assert lift[-1, 2] - pushed[2] == pytest.approx(3.0)
    assert np.allclose(lift[:, :2], pushed[:2])
    again = rows_of(retry_entry(ins, lift[-1, :6], 1.0, (0.2, -0.1, 0.5)))
    assert again[:, 2].min() >= ins.seated_tcp[2] - SEAT_FLOOR_MM - 1e-9
    assert again[-1, 0] == pytest.approx(ins.seated_tcp[0] + 0.2)


def test_pull_out_straight_with_wiggle(plans):
    pull = plans[2]
    segs = pull_out(pull, HOME)
    wig = np.vstack([s.rows for s in segs if s.phase == "wiggle_out"])
    assert np.allclose(wig[:, :2], pull.grasp_tcp[:2], atol=1e-6), "straight up"
    yaw = np.array([geometry.wrap_deg(r - pull.grasp_tcp[5]) for r in wig[:, 5]])
    assert np.abs(yaw).max() == pytest.approx(PULL_WIGGLE_DEG, abs=0.05)
    assert wig[-1, 2] >= pull.grasp_tcp[2] + pull.engage_mm
    d, _ = _steps(wig)
    assert (d <= SLOW.mm + 1e-6).all()


def test_place_free_sets_down_and_defect_drops(plans):
    _, _, _, spot, defect = plans
    start = np.r_[300.0, 0.0, 0.0, -180.0, 0.0, 0.0]
    set_down = rows_of(place_free(spot, start, 1.0))
    release = np.flatnonzero(np.diff(set_down[:, 6]) < 0)[0] + 1
    assert set_down[release, 2] == pytest.approx(spot.spot_tcp[2])
    drop = rows_of(place_free(defect, start, 1.0))
    release = np.flatnonzero(np.diff(drop[:, 6]) < 0)[0] + 1
    assert drop[release, 2] == pytest.approx(defect.spot_tcp[2] + defect.drop_mm)


def test_plan_rules():
    g = np.array([400, 0, -320, -180, 0, 0.0])
    with pytest.raises(PlanError):
        PickPlan("M1_02", g, 4.0, -250, -250)  # narrower than the closed fingers: would crush / miss
    with pytest.raises(PlanError):
        PickPlan("M1_02", g, 17.0, -250, -250)  # wider than open
    with pytest.raises(PlanError):
        pick_from_planner({"object_id": "M1_02", "grasp_tcp": g, "pad_width_mm": 7.0, "approach_z_mm": -250})
    ok = pick_from_planner({"object_id": "M1_02", "grasp_tcp": g, "pad_width_mm": 7.0, "approach_z_mm": -250,
                            "lift_z_mm": -250, "grip_index": 0})
    assert ok.pad_width_mm == 7.0


def test_motion_refuses_floor_and_workspace():
    m = Motion(np.array([400, 0, -300, -180, 0, 0.0]), 0.0, "t", "t", "world", floor_z=-310.0)
    with pytest.raises(MotionError):
        m.move("x", [400, 0, -312, -180, 0, 0])
    with pytest.raises(MotionError):
        m.move("x", [700, 0, -300, -180, 0, 0])
