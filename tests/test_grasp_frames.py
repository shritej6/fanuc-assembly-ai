"""T29: grasp/insert frames are regenerated from the printed CAD and fit the real gripper."""

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


def test_too_wide_pad_is_refused():
    class Grip:
        axis, center, half_width, span, z_top, height = "X", 0.0, 7.0, (-5.0, 5.0), 10.0, 6.0

    with pytest.raises(grasp_frames.GraspFrameError, match="does not fit"):
        grasp_frames.grip_entry("wide", Grip, (0.0, 0.0, 0.0))


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
