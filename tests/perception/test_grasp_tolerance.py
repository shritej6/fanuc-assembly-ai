"""Tests for grasp tolerance.
Authored by Shritej Gunda.
"""

import json

import numpy as np
import pytest

from perception.pose.grasp_tolerance import grasp_errors, load_grasp_points, report, spread, within_gate


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def _rot_x(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    m = np.eye(4)
    m[1:3, 1:3] = [[c, -s], [s, c]]
    return m


@pytest.fixture
def points(tmp_path):
    frames = {"objects": {
        "F01": {"T_local_from_stl": np.eye(4).tolist(), "grips": []},
        "M1_04": {"T_local_from_stl": [[1, 0, 0, 0], [0, 0, -1, 4], [0, 1, 0, 17], [0, 0, 0, 1]],
                  "grips": [{"tcp_local_mm": {"90": [0.0, 1.8, 27.9], "-90": [0.0, -1.8, 27.9]}}]}}}
    path = tmp_path / "grasp_frames.json"
    path.write_text(json.dumps(frames))
    return load_grasp_points(path)


def test_bases_without_grips_are_not_scored_and_tcp_maps_into_the_stl_frame(points):
    assert set(points) == {"M1_04"}
    local = np.array([[1, 0, 0, 0], [0, 0, -1, 4], [0, 1, 0, 17], [0, 0, 0, 1]], dtype=float)
    tcp_local = local @ np.append(points["M1_04"]["tcp_stl_m"][0] * 1000, 1)
    assert tcp_local[:3] == pytest.approx([0.0, 1.8, 27.9])
    assert local[:3, :3] @ points["M1_04"]["up_stl"] == pytest.approx([0, 0, 1])


def test_truth_scores_zero_and_a_shift_moves_the_tcp(points):
    pose = np.eye(4)
    pose[2, 3] = 0.5
    assert grasp_errors(pose, pose, points["M1_04"]) == pytest.approx({"tcp_mm": 0.0, "tilt_deg": 0.0})
    shifted = pose.copy()
    shifted[0, 3] += 0.002
    assert grasp_errors(shifted, pose, points["M1_04"])["tcp_mm"] == pytest.approx(2.0)
    assert not within_gate(2.0, 0.0) and within_gate(1.8, 3.0)


def test_tilt_is_the_up_axis_angle_and_ignores_yaw_about_up(points):
    up = points["M1_04"]["up_stl"]
    # the installed up axis is STL +Y here; a yaw about it changes no tilt
    yaw = np.eye(4)
    c, s = np.cos(np.radians(40)), np.sin(np.radians(40))
    yaw[[0, 0, 2, 2], [0, 2, 0, 2]] = [c, s, -s, c]
    assert yaw[:3, :3] @ up == pytest.approx(up)
    assert grasp_errors(yaw, np.eye(4), points["M1_04"])["tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert grasp_errors(_rot_z(5), np.eye(4), points["M1_04"])["tilt_deg"] == pytest.approx(5.0)


def test_spread_of_identical_estimates_is_zero(points):
    assert spread([np.eye(4)] * 3, points["M1_04"]) == pytest.approx({"tcp_spread_mm": 0, "tilt_spread_deg": 0})


def test_report_fails_a_part_with_one_estimate_outside_the_gate(points):
    good, bad = np.eye(4), _rot_x(0) @ np.eye(4)
    bad[0, 3] = 0.003
    base = {"object_id": "M1_04", "uid": "o01", "adds_mm": 1.0, "trans_mm": 1.0, "rot_deg": 1.0,
            "truth": np.eye(4).tolist()}
    rows = [{**base, "scene": "a", "run": 0, "estimate": good.tolist()},
            {**base, "scene": "a", "run": 1, "estimate": good.tolist()},
            {**base, "scene": "b", "run": 0, "estimate": bad.tolist()},
            {**base, "object_id": "F01", "scene": "c", "run": 0, "estimate": good.tolist()}]
    result = report(rows, points)
    assert set(result) == {"M1_04"}
    assert result["M1_04"]["share_within_gate"] == 0.5 and not result["M1_04"]["passes_gate"]
    assert result["M1_04"]["repeat_runs"] == 2 and result["M1_04"]["tcp_spread_mm"]["max"] == 0.0


def test_upright_prior_keeps_only_starts_with_the_part_up():
    from perception.pose.grasp_tolerance import upright_hypotheses

    up_stl, up_cam = np.array([0.0, 0.0, 1.0]), np.array([0.0, -1.0, 0.0])
    tilt_to_cam_up = _rot_x(90)[:3, :3]                 # STL +Z -> camera -Y (up)
    upside_down = _rot_x(-90)[:3, :3]
    grid = np.stack([np.eye(4), np.eye(4), np.eye(4)])
    grid[1, :3, :3], grid[2, :3, :3] = tilt_to_cam_up, upside_down
    assert upright_hypotheses(grid, up_stl, up_cam, 45.0).tolist() == [False, True, False]
