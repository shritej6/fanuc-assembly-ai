"""Real-arm guard: every violation is refused (hold + reason) or clamped (with a reason), never passed silently."""
from __future__ import annotations

import numpy as np
import pytest

from policy_learn.deploy.guards import ChunkGuard, ContactTarget, GuardLimits, hold_row, tilt_deg

HOME = [300.0, 0.0, -90.0, -180.0, 0.0, 0.0]
DT = 0.1


def path(start, end, n, grip=0.0):
    """n rows evenly from start (exclusive) to end (inclusive), fingers down."""
    s, e = np.asarray(start, float), np.asarray(end, float)
    rows = [np.concatenate([s + (e - s) * (k + 1) / n, [grip]]) for k in range(n)]
    return np.array(rows)


def test_safe_chunk_passes_unchanged():
    rows = path(HOME, [310, 5, -95, -180, 0, 3], 10)
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.passed and not res.held and res.reasons == ()
    np.testing.assert_allclose(res.rows, rows)


@pytest.mark.parametrize("k,value", [(0, 149.0), (0, 651.0), (1, -451.0), (1, 451.0), (2, -334.5), (2, 301.0)])
def test_outside_workspace_box_is_refused(k, value):
    rows = path(HOME, HOME, 3)
    rows[1, k] = value
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)  # row 0 is valid: only it is sent, row 1 is refused
    assert not res.held and not res.passed and res.truncated_at == 1
    assert "outside workspace" in res.reasons[0]
    np.testing.assert_allclose(res.rows, rows[:1])
    rows[0, k] = value  # no valid prefix: hold
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.held and not res.passed and res.truncated_at is None
    np.testing.assert_allclose(res.rows, [HOME + [0.0]])


def test_box_edges_are_allowed():
    start = [150.0, -450.0, -333.0, -180.0, 0.0, 0.0]
    rows = path(start, [150.0, -450.0, -334.0, -180.0, 0.0, 0.0], 5)
    assert ChunkGuard().guard(rows, start, 0.0, 1.0).held is False


def test_hold_keeps_gripper_closed_so_a_held_part_stays_held():
    rows = path(HOME, HOME, 2, grip=1.0)
    rows[0, 0] = np.nan
    res = ChunkGuard().guard(rows, HOME, 1.0, DT)
    assert res.held and res.rows[0, 6] == 1.0 and "NaN" in res.reasons[0]


@pytest.mark.parametrize("bad", [np.zeros((0, 7)), np.zeros((101, 7)), np.zeros((3, 6)), "abc"])
def test_bad_shapes_are_refused(bad):
    res = ChunkGuard().guard(bad, HOME, 0.0, DT)
    assert res.held and res.reasons


@pytest.mark.parametrize("dt", [0.05, 1.5, float("nan")])
def test_bad_dt_is_refused(dt):
    res = ChunkGuard().guard(path(HOME, HOME, 1), HOME, 0.0, dt)
    assert res.held and "dt" in res.reasons[0]


def test_tilted_fingers_are_refused_but_insert_tilt_is_allowed():
    ok = path(HOME, HOME[:3] + [-177.0, 0.0, 0.0], 2)  # 3 deg insertion tilt
    assert not ChunkGuard().guard(ok, HOME, 0.0, DT).held
    bad = path(HOME, HOME, 2)
    bad[-1, 3:6] = [-170.0, 0.0, 0.0]
    res = ChunkGuard().guard(bad, HOME, 0.0, DT)
    assert res.truncated_at == 1 and len(res.rows) == 1 and "tilted" in res.reasons[0]
    bad[0, 3:6] = [-170.0, 0.0, 0.0]
    res = ChunkGuard().guard(bad, HOME, 0.0, DT)
    assert res.held and "tilted" in res.reasons[0]


def test_tilt_is_wrap_safe():
    assert tilt_deg(180.0, 0.0, 77.0) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg(-180.0, 0.0, -300.0) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg(-180.0, 10.0, 0.0) == pytest.approx(10.0, abs=1e-6)


def test_contact_floor_refuses_far_below_seated_height():
    seated = -300.0
    start = [400.0, 0.0, -298.0, -180.0, 0.0, 0.0]
    guard = ChunkGuard()
    guard.set_target(ContactTarget(400.0, 0.0, seated, label="M4_02 on F04"))
    ok = path(start, [400.0, 0.0, seated - 0.9, -180, 0, 0], 20)  # light push, inside 1 mm
    res = guard.guard(ok, start, 1.0, DT)
    assert res.passed and not res.held
    bad = path(start, [400.0, 0.0, seated - 4.2, -180, 0, 0], 20)  # 3.2 mm below the floor: past the clamp band
    res = guard.guard(bad, start, 1.0, DT)  # the descent down to the clamp band is sent, the deepest row is not
    assert not res.held and res.truncated_at == 19 and len(res.rows) == 19
    assert "contact floor" in res.reasons[0] and "M4_02" in res.reasons[0]
    assert res.rows[:, 2].min() >= seated - 1.0
    res = guard.guard(bad[-1:], start, 1.0, DT)
    assert res.held and "contact floor" in res.reasons[0]


def test_slightly_below_floor_is_clamped_up_to_the_floor():
    """GR00T aims 1-3 mm low at a grasp; refusing the whole chunk froze the arm 3.6 mm above the part (eval 10/31)."""
    seated = -326.7
    start = [267.2, -105.8, -323.1, -180.0, 0.0, 0.0]
    guard = ChunkGuard()
    guard.set_target(ContactTarget(267.2, -105.8, seated, label="grasp M1_01"))
    rows = path(start, [267.2, -105.8, seated - 2.7, -180, 0, 0], 20)  # 1.7 mm below the floor
    res = guard.guard(rows, start, 0.0, DT)
    assert not res.held and not res.passed
    assert res.rows[:, 2].min() == pytest.approx(seated - 1.0)
    assert any("raised to the contact floor" in r and "M1_01" in r for r in res.reasons)


def test_floor_applies_only_near_the_target():
    guard = ChunkGuard()
    guard.set_target(ContactTarget(400.0, 0.0, -300.0, radius_mm=25.0))
    start = [500.0, 100.0, -320.0, -180.0, 0.0, 0.0]
    rows = path(start, [500.0, 100.0, -321.0, -180, 0, 0], 2)
    assert not guard.guard(rows, start, 0.0, DT).held


def test_small_overspeed_is_clamped_with_a_reason():
    lim = GuardLimits()
    step = lim.max_speed_mm_s * DT * 2.0  # 2x the limit: clamp, not refuse
    rows = path(HOME, [HOME[0] + step] + HOME[1:], 1)
    res = ChunkGuard(lim).guard(rows, HOME, 0.0, DT)
    assert not res.held and not res.passed and "clamped" in res.reasons[0]
    assert res.rows[0, 0] - HOME[0] == pytest.approx(lim.max_speed_mm_s * DT)


def test_big_jump_is_refused_as_a_glitch():
    rows = path(HOME, [600.0, 300.0, 100.0, -180, 0, 0], 1)
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.held and "refuse above" in res.reasons[0]


def test_last_5mm_above_contact_is_slow():
    lim = GuardLimits()
    guard = ChunkGuard(lim)
    guard.set_target(ContactTarget(400.0, 0.0, -300.0))
    start = [400.0, 0.0, -297.0, -180.0, 0.0, 0.0]
    fast_in_zone = lim.slow_speed_mm_s * DT * 2.0
    rows = path(start, [400.0, 0.0, -297.0 - fast_in_zone, -180, 0, 0], 1)
    res = guard.guard(rows, start, 1.0, DT)
    assert "slow" in res.reasons[0]
    assert start[2] - res.rows[0, 2] == pytest.approx(lim.slow_speed_mm_s * DT)


def test_large_raw_wrist_turn_is_not_hidden_by_angle_wrap():
    """r 179 -> -179 is 358 deg of J6 travel on the controller, not 2 deg."""
    start = HOME[:5] + [179.0]
    rows = path(start, HOME[:5] + [-179.0], 1)
    res = ChunkGuard().guard(rows, start, 0.0, DT)
    assert res.held


def test_gripper_values_snap_or_refuse():
    rows = path(HOME, HOME, 3)
    rows[:, 6] = 0.95
    rows[0, 6] = 0.95
    res = ChunkGuard().guard(rows, HOME, 1.0, DT)
    assert not res.held and np.all(res.rows[:, 6] == 1.0)
    rows[:, 6] = 0.5
    res = ChunkGuard().guard(rows, HOME, 1.0, DT)
    assert res.held and "neither open" in res.reasons[0]


def test_gripper_chatter_is_refused():
    rows = path(HOME, HOME, 4)
    rows[:, 6] = [1, 0, 1, 1]
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)  # the single close at row 0 (arm still) is a valid prefix
    assert res.truncated_at == 1 and "chatter" in res.reasons[0]
    np.testing.assert_allclose(res.rows, rows[:1])


def test_gripper_change_while_moving_is_refused():
    rows = path(HOME, [HOME[0] + 10.0] + HOME[1:], 1, grip=1.0)
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.held and "while moving" in res.reasons[0]


def test_clamped_row_rechecked_against_floor():
    """Clamping toward a target can land inside the target radius below the floor: refuse."""
    guard = ChunkGuard()
    guard.set_target(ContactTarget(400.0, 0.0, -300.0, radius_mm=10.0))
    start = [389.5, 0.0, -302.0, -180.0, 0.0, 0.0]  # just outside the radius, below the floor
    rows = path(start, [391.5, 0.0, -300.5, -180, 0, 0], 1)  # 2.5x the slow limit: clamps to x 390.3, z -301.4
    res = guard.guard(rows, start, 0.0, 0.1)
    assert res.held and any("after clamp" in r and "contact floor" in r for r in res.reasons)


def test_valid_prefix_before_a_refused_row_is_sent():
    """The 10/31 product-5 evals froze: one row under the workspace floor held the whole chunk, so the policy was
    asked again from the same pose forever. Now rows 0-9 go and the arm stops at row 9."""
    start = [380.0, 74.0, -300.0, -180.0, 0.0, 180.0]
    rows = path(start, [380.0, 74.0, -328.0, -180, 0, 180], 10)  # above the slow zone: unchanged
    rows = np.vstack([rows, [[380.0, 74.0, -334.5, -180.0, 0.0, 180.0, 0.0]] * 6])
    res = ChunkGuard().guard(rows, start, 0.0, 0.2)
    assert not res.held and not res.passed and res.truncated_at == 10
    np.testing.assert_allclose(res.rows, rows[:10])
    assert any("row 10" in r and "outside workspace" in r for r in res.reasons)
    assert any("truncated at row 10" in r for r in res.reasons)


def test_truncated_prefix_still_passes_every_check():
    """The prefix is re-checked on its own: a speed clamp inside it is applied and reported, nothing is loosened."""
    lim = GuardLimits()
    fast = path(HOME, [HOME[0] + lim.max_speed_mm_s * DT * 2.0] + HOME[1:], 1)  # 2x: clamped
    rows = np.vstack([fast, [[HOME[0], 0.0, -335.0, -180.0, 0.0, 0.0, 0.0]]])
    res = ChunkGuard(lim).guard(rows, HOME, 0.0, DT)
    assert res.truncated_at == 1 and any("clamped" in r for r in res.reasons)
    assert res.rows[0, 0] - HOME[0] == pytest.approx(lim.max_speed_mm_s * DT)


def test_a_hold_reports_why_row_0_itself_was_refused():
    rows = path(HOME, [HOME[0] + 10.0] + HOME[1:], 2, grip=1.0)  # closes while moving at row 0
    rows[1, 2] = -335.0
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.held and "outside workspace" in res.reasons[0]
    assert any(r.startswith("no valid prefix:") and "while moving" in r for r in res.reasons)


def test_malformed_chunk_is_held_whole_even_with_a_valid_prefix():
    rows = path(HOME, HOME, 3)
    rows[2, 0] = np.nan
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.held and res.truncated_at is None and len(res.rows) == 1


def test_hold_row_is_clipped_into_the_box():
    row = hold_row([100.0, 0.0, -340.0, -180.0, 0.0, 0.0], 0.7)
    np.testing.assert_allclose(row, [[150.0, 0.0, -334.0, -180.0, 0.0, 0.0, 1.0]])
