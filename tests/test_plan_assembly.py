"""Tests for plan assembly.
Authored by Shritej Gunda.
"""

import json

import numpy as np
import pytest

from sim.demo.plan_assembly import REPO, PlanError, cross_section, grip_tab, plan, read_stl_mm, uf0_to_base

# Product 1 at step 1 from set-N scene chunk_0000/000003: base + housing installed, carrier and rear guard loose.
BASE_TF = [[-0.101, -0.9949, 0.0, 329.4825], [0.9949, -0.101, 0.0, 44.9882], [0.0, 0.0, 1.0, -327.0], [0, 0, 0, 1]]
CARRIER_TF = [[0.7145, -0.6997, 0.0, 244.7533], [0.6997, 0.7145, 0.0, -117.7569], [0.0, 0.0, 1.0, -335.0], [0, 0, 0, 1]]
GUARD_TF = [[0.3725, 0.0, -0.928, 250.3798], [-0.928, 0.0, -0.3725, -192.625], [0.0, 1.0, 0.0, -318.0], [0, 0, 0, 1]]


def _obj(uid, object_id, kind, state, stl, tf, rest=None):
    return {"uid": uid, "object_id": object_id, "kind": kind, "product": 1, "variant": "ok", "state": state,
            "rest": rest, "stl": f"cad/stl/{stl}.stl", "transform_uf0": tf}


@pytest.fixture
def label(tmp_path):
    objects = [_obj("o01", "F01", "base", "on_table", "F01_Rotary_Fixture", BASE_TF),
               _obj("o04", "M1_01", "part", "installed", "M1_01_Housing", BASE_TF),
               _obj("o05", "M1_02", "part", "on_table", "M1_02_Spindle_Carrier", CARRIER_TF, "installed"),
               _obj("o07", "M1_04", "part", "on_table", "M1_04_Rear_Guard", GUARD_TF, "installed")]
    path = tmp_path / "000003.json"
    path.write_text(json.dumps({"spec": {"objects": objects, "table_z_uf0": -335.0}}))
    return path


def test_cross_section_below_the_top_is_the_grip_tab_outline():
    tab = read_stl_mm(REPO / "cad/stl/M1_02_Spindle_Carrier.stl") / 1000.0
    points = cross_section(tab, tab[:, 2].max() - 0.002)
    assert np.ptp(points[:, 0]) * 1000 == pytest.approx(7.0, abs=0.1)     # the 7 x 10 mm grip tab
    assert np.ptp(points[:, 1]) * 1000 == pytest.approx(10.0, abs=0.1)
    with pytest.raises(PlanError):
        cross_section(tab, tab[:, 2].max() + 0.001)


def test_grip_tab_closes_across_the_thin_side_and_rejects_what_the_fingers_cannot_hold():
    verts = read_stl_mm(REPO / "cad/stl/M1_02_Spindle_Carrier.stl")
    tab = grip_tab(verts, uf0_to_base(CARRIER_TF))
    assert tab["width_mm"] == pytest.approx(7.0, abs=0.1)
    assert abs(tab["spread_part"][0]) > np.cos(np.radians(1.0))           # part x: the tab's 7 mm side, 1 deg search
    with pytest.raises(PlanError, match="outside the fingers"):
        grip_tab(verts * np.array([3.0, 3.0, 1.0]), uf0_to_base(CARRIER_TF))   # a 21 mm tab


def test_plan_seats_each_part_on_its_cad_install_pose_with_the_jaws_aligned(label):
    result = plan(label, 1, {})
    assert [s["object_id"] for s in result["steps"]] == ["M1_02", "M1_04"]     # install order, housing already on
    for step in result["steps"]:
        assert step["seat_error_mm"] < 0.1
        assert step["seat_rotation_error_deg"] < 1.4                          # inside the CAD yaw tolerance
    assert len(result["frames"]) > 0
    assert all(f["jaw"] >= 0 for f in result["frames"])
