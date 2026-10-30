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
    for a, b in (("tilt", "touch"), ("touch", "level"), ("level", "twist"), ("twist", "slide"), ("slide", "push")):
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


def test_set_down_touches_the_table_at_press_speed(plans):
    """The last SEAT_APPROACH_MM onto the test bed goes at PRESS speed (0.25 mm rows gave 14-19 N touch impulses)."""
    from sim.assembly_lab.skills.insert import PRESS, SEAT_APPROACH_MM

    _, _, _, spot, _ = plans
    segs = place_free(spot, np.r_[300.0, 0.0, 0.0, -180.0, 0.0, 0.0], 1.0)
    touch = np.vstack([s.rows for s in segs if s.phase == "touch_down"])
    assert touch[-1, 2] == pytest.approx(spot.spot_tcp[2])
    prev = np.vstack([s.rows for s in segs if s.phase == "set_down"])[-1]
    assert prev[2] == pytest.approx(spot.spot_tcp[2] + SEAT_APPROACH_MM)
    d, _ = _steps(np.vstack([prev, touch]))
    assert (d <= PRESS.mm + 1e-6).all()


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


def test_start_outside_box_enters_with_in_box_rows(plans):
    home = np.array([87.0, 0.0, 200.0, -180.0, 0.0, 0.0])  # the cell's home TCP is outside the box
    rows = rows_of(pick(plans[0], home))
    _in_workspace(rows)
    assert rows[0, 0] == pytest.approx(WORKSPACE[0][0] + 5.0)
    tilted_home = np.array([87.0, 0.0, 200.0, 170.0, 30.0, 0.0])  # tool far from straight down at home
    rows = rows_of(pick(plans[0], tilted_home))
    assert max(geometry.tilt_deg(r) for r in rows) <= 6.0 + 1e-6


def test_part_is_level_inside_the_bore(plans):
    ins = plans[1]
    start = np.r_[ins.seated_tcp[:2] + 60, ins.approach_z_mm + 20, ins.seated_tcp[3:]]
    rows = rows_of(insert(ins, start, 1.0))
    bore = rows[:, 2] < ins.seated_tcp[2] + ins.engage_mm - 0.5 * ins.chamfer_mm - 1e-6
    assert bore.any()
    assert max(geometry.tilt_deg(r) for r in rows[bore]) < 1e-6, "a tilted part jams a 0.5 mm clearance fit"
    yaw = [abs(geometry.wrap_deg(r[5] - ins.seated_tcp[5])) for r in rows[bore]]
    assert max(yaw) < 1e-6, "no twist inside the bore"


def test_part_pivoted_in_the_fingers_is_levelled_before_the_bore(plans):
    """The runner's w, p correction (part held 1.5 deg off) is held all the way down inside the bore."""
    ins = plans[1]
    start = np.r_[ins.seated_tcp[:2] + 60, ins.approach_z_mm + 20, ins.seated_tcp[3:]]
    level = geometry.tilted(ins.seated_tcp, 0.0, 1.5)
    rows = rows_of(insert(ins, start, 1.0, (0.0, 0.0, 0.0, level[3], level[4])))
    bore = rows[:, 2] < ins.seated_tcp[2] + ins.engage_mm - 0.5 * ins.chamfer_mm - 1e-6
    assert bore.any()
    off = [geometry.rotation_distance_deg(np.r_[r[:3], level[3], level[4], ins.seated_tcp[5]], r[:6]) for r in rows[bore]]
    assert max(off) < 1e-6


def test_motion_refuses_tool_tilt_over_6_deg(plans):
    ins = plans[1]
    m = Motion(np.r_[ins.seated_tcp[:2], ins.approach_z_mm, ins.seated_tcp[3:]], 1.0, "t", "t", "world")
    m.move("ok", geometry.tilted(m.pose, 0.0, 6.0), SLOW)
    with pytest.raises(MotionError, match="from straight down"):
        m.move("bad", geometry.tilted(m.pose, 0.0, 0.5), SLOW)


def test_max_level_correction_keeps_every_insert_row_within_6_deg(plans):
    from sim.assembly_lab.skills.runner import MAX_LEVEL_CORRECTION_DEG

    assert MAX_LEVEL_CORRECTION_DEG == pytest.approx(3.0)
    ins = plans[1]
    start = np.r_[ins.seated_tcp[:2] + 60, ins.approach_z_mm + 20, ins.seated_tcp[3:]]
    for lead in (0.0, 90.0, 180.0, 270.0):
        level = geometry.tilted(ins.seated_tcp, lead, MAX_LEVEL_CORRECTION_DEG)
        rows = rows_of(insert(ins, start, 1.0, (0.0, 0.0, 0.0, level[3], level[4])))
        assert max(geometry.tilt_deg(r) for r in rows) <= 6.0 + 1e-6


def test_retry_clears_the_bore_before_remeasuring(plans):
    from sim.assembly_lab.skills.insert import _entry_heights, retry_clear

    ins = plans[1]
    pushed = np.r_[ins.seated_tcp[:3], ins.seated_tcp[3:]]
    lifted = rows_of(retry_lift(ins, pushed, 1.0))[-1]
    rows = rows_of(retry_clear(ins, lifted[:6], 1.0))
    assert rows[-1, 2] == pytest.approx(_entry_heights(ins)["top"])


def test_twist_turns_the_part_about_its_own_axis(plans):
    """The +-2 deg twist pivots on the part's insert axis: the held part's origin stays put, also when it is gripped
    on an edge pad 21 mm from its centre (M1_01-style)."""
    from dataclasses import replace

    ins = replace(plans[1], seated_object_pose=plans[1].seated_object_pose + np.array([0.0, 21.06, 0, 0, 0, 0]))
    t_tcp_obj = np.linalg.inv(geometry.pose_to_matrix(ins.seated_tcp)) @ geometry.pose_to_matrix(ins.seated_object_pose)
    start = np.r_[ins.seated_tcp[:2] + 60, ins.approach_z_mm + 20, ins.seated_tcp[3:]]
    segs = insert(ins, start, 1.0)
    twist = np.vstack([s.rows for s in segs if s.phase == "twist"])
    xy = np.array([(geometry.pose_to_matrix(r[:6]) @ t_tcp_obj)[:2, 3] for r in twist])
    assert np.abs(xy - ins.seated_object_pose[:2]).max() < 0.02  # chord vs arc between rows; a TCP pivot gives 0.73
    assert max(abs(geometry.wrap_deg(r[5] - ins.seated_tcp[5])) for r in twist) == pytest.approx(2.0, abs=0.01)


def test_start_pose_points_the_tool_down_inside_the_box():
    from sim.assembly_lab.skills.motion import start_pose

    p = start_pose([87.0, 0.0, 200.0, 140.0, 0.0, 33.0])  # the Isaac home: x outside the box, tool 50 deg off down
    assert p[0] == pytest.approx(WORKSPACE[0][0] + 5.0) and geometry.tilt_deg(p) == pytest.approx(0.0, abs=1e-9)
    assert p[5] == pytest.approx(33.0)


def test_last_mm_and_push_go_at_press_speed(plans):
    from sim.assembly_lab.skills.insert import PRESS, SEAT_APPROACH_MM

    ins = plans[1]
    start = np.r_[ins.seated_tcp[:2] + 60, ins.approach_z_mm + 20, ins.seated_tcp[3:]]
    rows = rows_of(insert(ins, start, 1.0))
    near = np.flatnonzero(rows[:, 2] <= ins.seated_tcp[2] + SEAT_APPROACH_MM + 1e-6)
    steps = np.linalg.norm(np.diff(rows[near[0]:], axis=0)[:, :3], axis=1)
    assert steps.max() <= PRESS.mm + 1e-9


def _watch(yaw_tol=1.0):
    import dataclasses

    from sim.assembly_lab.skills import dry_run
    from sim.assembly_lab.skills.jam import InsertWatch

    objs, seats = dry_run.demo_world()
    ins = dry_run.demo_plans(objs, seats)[1]
    return InsertWatch(dataclasses.replace(ins, yaw_tol_deg=yaw_tol))


def _obs(oid, pose):
    return {"sim": {"object_poses": {oid: list(pose)}}}


def test_twist_watch_flags_a_part_that_stops_turning_with_the_tcp():
    from sim.assembly_lab.skills.jam import InsertJam

    w = _watch(yaw_tol=1.0)
    part, tcp = w.plan.seated_object_pose + [0, 0, 2.0, 0, 0, 0], w.plan.seated_tcp + [0, 0, 2.0, 0, 0, 0]
    for d in (0.0, 0.5, 1.0, 2.0):  # the part turns with the TCP: fine
        w.observe(_obs(w.plan.object_id, part + [0, 0, 0, 0, 0, d]), tcp + [0, 0, 0, 0, 0, d], "twist", 0.3)
    w = _watch(yaw_tol=1.0)
    w.observe(_obs(w.plan.object_id, part), tcp, "twist", 0.3)
    with pytest.raises(InsertJam, match="off the commanded twist"):  # wedged: the TCP turns 2 deg, the part does not
        w.observe(_obs(w.plan.object_id, part), tcp + [0, 0, 0, 0, 0, 2.0], "twist", 0.3)


@pytest.mark.parametrize("xy_off,sink,rise,jams", [
    (0.0, 0.01, 1.5, True),    # in the bore, not sinking, force rising: jam
    (0.0, 0.05, 1.5, False),   # sinking with the command: seating
    (0.0, 0.01, 0.2, False),   # not loaded: no jam
    (5.0, 0.01, 1.5, False),   # on the rim outside the capture: soft stop + retry handle it
])
def test_jam_watch(xy_off, sink, rise, jams):
    from sim.assembly_lab.skills.jam import JAM_ROWS, InsertJam

    w = _watch()
    oid, part0, tcp0 = w.plan.object_id, w.plan.seated_object_pose + [xy_off, 0, 1.0, 0, 0, 0], w.plan.seated_tcp
    calls = [(tcp0 + [0, 0, 1.0 - 0.05 * i, 0, 0, 0], part0 - [0, 0, sink * i / JAM_ROWS, 0, 0, 0],
              0.5 + rise * i / JAM_ROWS) for i in range(JAM_ROWS + 1)]
    run = lambda: [w.observe(_obs(oid, p), t, "push", f) for t, p, f in calls]
    if jams:
        with pytest.raises(InsertJam, match="jam"):
            run()
    else:
        run()


def test_watch_logs_the_offset_at_first_base_contact():
    w = _watch()
    oid, seat, tcp = w.plan.object_id, w.plan.seated_object_pose, w.plan.seated_tcp
    w.observe(_obs(oid, seat + [0.4, -0.3, 2.0, 0, 0, 0]), tcp + [0, 0, 2.0, 0, 0, 0], "touch", 0.3)
    assert w.first_contact is None  # in-hand baseline only
    w.observe(_obs(oid, seat + [0.4, -0.3, 1.5, 0, 0, 0]), tcp + [0, 0, 1.5, 0, 0, 0], "touch", 1.6)
    w.observe(_obs(oid, seat + [0.1, 0.0, 1.0, 0, 0, 0]), tcp + [0, 0, 1.0, 0, 0, 0], "level", 2.0)
    assert w.first_contact["phase"] == "touch" and w.first_contact["xy_mm"] == pytest.approx(0.5)
    assert w.first_contact["z_above_seat_mm"] == pytest.approx(1.5)


def test_last_fine_zone_over_the_seat_goes_in_fine_rows(plans):
    """run10: the seat went past 2 N within one 0.05 mm row (6-11.6 N kept, 15.4 / 17 N discarded); the last
    FINE_ZONE_MM and the push are pressed at FINE (<= 0.02 mm/row), on inserts and on test-bed set-downs."""
    from sim.assembly_lab.skills.insert import FINE, FINE_ZONE_MM

    ins, spot = plans[1], plans[3]
    start = np.r_[ins.seated_tcp[:2] + 60, ins.approach_z_mm + 20, ins.seated_tcp[3:]]
    rows = np.vstack([s.rows for s in insert(ins, start, 1.0) if s.phase in ("slide", "push")])
    near = np.flatnonzero(rows[:, 2] <= ins.seated_tcp[2] + FINE_ZONE_MM + 1e-6)
    steps = np.linalg.norm(np.diff(rows[near[0]:], axis=0)[:, :3], axis=1)
    assert steps.max() <= FINE.mm + 1e-9 and rows[near[0]:, 2].min() < ins.seated_tcp[2]
    segs = place_free(spot, np.r_[300.0, 0.0, 0.0, -180.0, 0.0, 0.0], 1.0)
    touch = np.vstack([s.rows for s in segs if s.phase == "touch_down"])
    fine = touch[touch[:, 2] <= spot.spot_tcp[2] + FINE_ZONE_MM + 1e-6]
    d, _ = _steps(fine)
    assert (d <= FINE.mm + 1e-6).all() and fine[-1, 2] == pytest.approx(spot.spot_tcp[2])
