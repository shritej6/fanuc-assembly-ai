"""Real-arm guard: every violation is refused (hold + reason) or clamped (with a reason), never passed silently."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from policy_learn.deploy.guards import (
    ChunkGuard,
    ContactTarget,
    GuardLimits,
    InvalidGuardLimitsError,
    UnsafeHoldError,
    hold_row,
    tilt_deg,
)

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
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.held and not res.passed
    assert "outside workspace" in res.reasons[0]
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
    assert res.held and "tilted" in res.reasons[0]


def test_tilt_is_wrap_safe():
    assert tilt_deg(180.0, 0.0, 77.0) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg(-180.0, 0.0, -300.0) == pytest.approx(0.0, abs=1e-6)
    assert tilt_deg(-180.0, 10.0, 0.0) == pytest.approx(10.0, abs=1e-6)


def test_contact_floor_refuses_more_than_1mm_below_seated_height():
    seated = -300.0
    start = [400.0, 0.0, -298.0, -180.0, 0.0, 0.0]
    guard = ChunkGuard()
    guard.set_target(ContactTarget(400.0, 0.0, seated, label="M4_02 on F04"))
    ok = path(start, [400.0, 0.0, seated - 0.9, -180, 0, 0], 20)  # light push, inside 1 mm
    assert not guard.guard(ok, start, 1.0, DT).held
    bad = path(start, [400.0, 0.0, seated - 1.2, -180, 0, 0], 20)
    res = guard.guard(bad, start, 1.0, DT)
    assert res.held and "contact floor" in res.reasons[0] and "M4_02" in res.reasons[0]


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
    res = ChunkGuard().guard(rows, HOME, 0.0, DT)
    assert res.held and "chatter" in res.reasons[0]


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


def test_hold_row_never_turns_an_invalid_pose_into_a_motion_command():
    with pytest.raises(UnsafeHoldError, match="cannot issue a TCP hold"):
        hold_row([100.0, 0.0, -340.0, -180.0, 0.0, 0.0], 0.7)


@pytest.mark.parametrize("gripper", [float("nan"), float("inf"), 0.05, 0.5, 0.95, "unknown", True])
def test_hold_row_never_guesses_an_invalid_gripper_state(gripper):
    with pytest.raises(UnsafeHoldError, match="gripper"):
        hold_row(HOME, gripper)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_speed_mm_s": float("nan")},
        {"max_speed_mm_s": -1.0},
        {"x_mm": (650.0, 150.0)},
        {"x_mm": (150.0, "far")},
        {"slow_speed_mm_s": 151.0},
        {"refuse_factor": 1.0},
        {"gripper_snap": 0.5},
    ],
)
def test_guard_limits_reject_nonfinite_or_inconsistent_values(changes):
    with pytest.raises(InvalidGuardLimitsError):
        GuardLimits(status="MEASURED", source="CSI calibration report r1", **changes)


def test_production_limits_require_measured_status_and_source():
    assert GuardLimits().production_ready is False
    assert GuardLimits(status="MEASURED", source="unbound claim").production_ready is False


def _calibration_report(tmp_path, **changes):
    defaults = GuardLimits()
    values = {
        name: getattr(defaults, name)
        for name in (
            "x_mm",
            "y_mm",
            "z_mm",
            "max_abs_wpr_deg",
            "max_tilt_deg",
            "max_speed_mm_s",
            "slow_speed_mm_s",
            "max_rot_deg_s",
            "slow_rot_deg_s",
            "slow_zone_mm",
            "floor_margin_mm",
            "refuse_factor",
            "gripper_snap",
            "gripper_change_max_step_mm",
        )
    }
    values.update(changes)
    path = tmp_path / "guard-calibration.json"
    path.write_text(json.dumps({"status": "MEASURED", "source": "CSI calibration run r1", "limits": values}))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_production_limits_are_derived_from_a_hash_bound_report(tmp_path):
    path, digest = _calibration_report(tmp_path, max_speed_mm_s=42.0)
    limits = GuardLimits.from_calibration(path, digest)
    assert limits.production_ready is True
    assert limits.max_speed_mm_s == 42.0
    with pytest.raises(InvalidGuardLimitsError, match="SHA-256"):
        GuardLimits.from_calibration(path, "0" * 64)


def test_calibration_report_rejects_unknown_fields(tmp_path):
    path, _ = _calibration_report(tmp_path)
    payload = json.loads(path.read_text())
    payload["limits"]["unreviewed"] = 1.0
    path.write_text(json.dumps(payload))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(InvalidGuardLimitsError, match="unknown"):
        GuardLimits.from_calibration(path, digest)


def test_calibration_report_rejects_symlink(tmp_path):
    path, digest = _calibration_report(tmp_path)
    link = tmp_path / "calibration-link.json"
    link.symlink_to(path)
    with pytest.raises(InvalidGuardLimitsError, match="non-symlink"):
        GuardLimits.from_calibration(link, digest)
