"""grasp/insert frames are regenerated from the printed CAD and fit the real gripper.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

grasp_frames = pytest.importorskip("cad.grasp_frames")
COMMITTED = json.loads(grasp_frames.OUT.read_text())


@pytest.fixture(scope="module")
def fresh() -> dict:
    return json.loads(json.dumps(grasp_frames.frames()))


def test_committed_file_is_what_the_cad_produces(fresh):
    assert fresh == COMMITTED


def test_all_30_objects_and_every_product_part_has_a_grip():
    objects = COMMITTED["objects"]
    assert len(objects) == 30
    assert all(v["grips"] for k, v in objects.items() if k.startswith("M"))


def test_every_pad_fits_between_the_closed_and_open_gap():
    gap = COMMITTED["gripper_gap_mm"]
    pads = [g["pad_width_mm"] for v in COMMITTED["objects"].values() for g in v["grips"]]
    assert pads and all(gap["closed_max"] < w < gap["open_min"] for w in pads)


class _Grip:
    axis, center, half_width, span, z_top, height = "X", 0.0, 3.5, (-5.0, 5.0), 10.0, 6.0


def test_too_wide_pad_is_refused():
    class Wide(_Grip):
        half_width = 8.5     # 17 mm pad: wider than the fingers' 16.5 mm open gap

    with pytest.raises(grasp_frames.GraspFrameError, match="does not fit"):
        grasp_frames.grip_entry("wide", Wide, (0.0, 0.0, 0.0), np.array([0.0, 0.0, 8.0]))


def test_contact_off_the_pad_is_refused():
    with pytest.raises(grasp_frames.GraspFrameError, match="not on the CAD grip pad"):
        grasp_frames.grip_entry("off", _Grip, (0.0, 0.0, 0.0), np.array([1.0, 0.0, 8.0]))


def test_pad_point_is_the_contact_moved_down_to_the_fingertip():
    entry = grasp_frames.grip_entry("ok", _Grip, (0.0, 2.0, 1.0), np.array([0.0, 3.0, 8.0]))
    assert entry["pad_point_local_mm"] == pytest.approx([0.0, 1.0, 7.0 - (86.10 - 84.25)])


def test_tcp_is_1_8_mm_off_the_pad_and_flips_with_the_wrist_turn():
    entry = grasp_frames.grip_entry("ok", _Grip, (0.0, 2.0, 1.0), np.array([0.0, 3.0, 8.0]))
    z = 7.0 - (86.10 - 84.25)
    assert entry["tcp_local_mm"] == {"90": pytest.approx([0.0, 2.8, z]), "-90": pytest.approx([0.0, -0.8, z])}


def test_every_tcp_reproduces_the_team_v_validated_gripper_origin():
    """At the validated turn, installed origin + tcp_local_mm[k] is Team V's gripper origin (flange axis), x and y."""
    fingers = json.loads(grasp_frames.FINGERS.read_text())
    for oid, obj in COMMITTED["objects"].items():
        if not oid.startswith("M"):
            continue
        frame = fingers[obj["name"]]
        step_x = np.array(frame["rotation_gripper_to_world"])[:, 0]
        k = math.degrees(math.atan2(step_x[1], step_x[0]))
        tcp = np.add(obj["installed_origin_mm"], obj["grips"][0]["tcp_local_mm"][f"{k:g}"])
        assert tcp[:2] == pytest.approx(frame["translation_mm"][:2], abs=0.01), oid


def test_gripper_origin_not_1_8_mm_from_the_contact_is_refused():
    fingers = json.loads(grasp_frames.FINGERS.read_text())
    frame = dict(fingers["M1_01_Housing"], translation_mm=[0.0, -21.06, 94.5])   # TCP on the contact: the old bug
    with pytest.raises(grasp_frames.GraspFrameError, match="1.8 mm from the contact"):
        grasp_frames._contact({"M1_01_Housing": frame}, "M1_01_Housing")


def test_every_tcp_comes_from_a_team_v_finger_contact():
    fingers = json.loads(grasp_frames.FINGERS.read_text())
    parts = [v["name"] for k, v in COMMITTED["objects"].items() if k.startswith("M")]
    assert len(parts) == 24 and all(name in fingers for name in parts)


def test_rear_guard_stl_frame_is_its_installed_frame_rotated_back():
    guard = COMMITTED["objects"]["M1_04"]
    full = np.array(guard["T_local_from_stl"])
    full[:3, 3] += guard["installed_origin_mm"]
    assert np.allclose(full[:3, :3], [[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    assert np.allclose(full[:3, 3], [0.0, 21.5, 21.0])


@pytest.mark.parametrize("yaw", [-170.0, -35.0, 0.0, 62.0, 145.0])
@pytest.mark.parametrize("axis", ["X", "Y"])
def test_roll_offset_spreads_the_jaws_along_the_closing_axis(axis, yaw):
    """sim/starter/ik.py spreads the jaws along rot(w, p, r) @ +y; with w, p = -180, 0 that is (sin r, -cos r)."""
    r = math.radians(yaw + grasp_frames.ROLL_OFFSET_DEG[axis])
    rx = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])          # w = -180
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    spread = (rz @ rx @ [0, 1, 0])[:2]
    y = math.radians(yaw)
    closing = [math.cos(y), math.sin(y)] if axis == "X" else [-math.sin(y), math.cos(y)]
    assert abs(abs(float(np.dot(spread, closing))) - 1.0) < 1e-9
