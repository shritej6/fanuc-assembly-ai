"""Off-Isaac tests for the env's layout solver and TCP / IK layer:  uv run pytest sim/assembly_lab/env -q"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import layout as L
import tcp

FLOOR_Z = -1230.0


@pytest.fixture(scope="module")
def solver():
    return L.LayoutSolver()


def test_wpr_round_trip():
    for wpr in [(-180, 0, 0), (180, 0, 90), (-177, 2, -30), (10, -20, 135)]:
        back = L.wpr_from_rotation(L.fanuc_rotation(*wpr))
        assert np.allclose(L.fanuc_rotation(*back), L.fanuc_rotation(*wpr), atol=1e-9)


def test_quat_round_trip():
    rot = L.fanuc_rotation(-177, 2, -30)
    assert np.allclose(L.rotation_from_quat_xyzw(L.quat_xyzw(rot)), rot, atol=1e-9)


def test_ik_reaches_poses_within_tolerance():
    q = None
    for pose in ([400, 0, -250, -180, 0, 0], [350, 120, -300, -180, 0, 45], [420, -150, -320, -177, 2, -30]):
        q = tcp.solve(pose, q)
        got = tcp.tcp_pose(q)
        assert np.linalg.norm(np.subtract(got[:3], pose[:3])) < 0.3
        assert np.allclose(L.fanuc_rotation(*got[3:]), L.fanuc_rotation(*pose[3:]), atol=5e-3)


def test_workspace_box_is_enforced():
    with pytest.raises(tcp.TcpError):
        tcp.check_row([400, 0, -340, -180, 0, 0, 0])      # below the lowest allowed fingertip target


def test_pla_mass_is_positive_and_plausible(solver):
    for tid, geo in solver.geometry.items():
        mass, _ = L.pla_mass_kg(L.REPO / geo["stl"])
        assert 0.0005 < mass < 0.1, tid


def test_layout_states(solver):
    pool = ["F04", "M4_01", "M4_02", "M4_03", "M4_03#2", "M4_04#2"]
    lay = {"objects": [{"id": "F04", "x": 300, "y": -60, "yaw": 0},
                       {"id": "M4_01", "state": "on:F04"},
                       {"id": "M4_02", "x": 300, "y": 60, "yaw": 30},
                       {"id": "M4_03", "x": 380, "y": 60, "yaw": 0, "state": "fallen"}],
           "spares": [{"id": "M4_03", "slot": "S2"}], "defects": [{"id": "M4_04"}]}
    out = {p.uid: p for p in solver.solve(lay, pool, FLOOR_Z)}
    assert out["M4_01"].state == "on:F04"
    seat = np.array(solver.geometry["M4_01"]["installed_in_base_frame"])
    assert np.allclose(out["M4_01"].tf_uf0, out["F04"].tf_uf0 @ seat)      # the seat is in the base body (STL) frame
    # seated means sitting on the base's seating face (F04 STL z = 0, 8 mm above the table), not on the table
    assert np.isclose(out["M4_01"].tf_uf0[2, 3], solver.table_z - solver.geometry["F04"]["bounds_mm"][0][2])
    assert np.isclose(out["M4_02"].tf_uf0[2, 3], solver.table_z)
    assert not np.allclose(out["M4_03"].tf_uf0[:3, 2], [0, 0, 1])        # fallen: up axis changed
    assert out["M4_03#2"].role == "spare" and tuple(out["M4_03#2"].tf_uf0[:2, 3]) == solver.spare_slot("S2")
    assert out["M4_04#2"].role == "defect"
    assert all(p.role != "parked" for p in out.values())


def test_layout_rejects_unknown_and_exhausted(solver):
    with pytest.raises(L.LayoutError):
        solver.solve({"objects": [{"id": "M9_01", "x": 0, "y": 0}]}, ["F04"], FLOOR_Z)
    with pytest.raises(L.LayoutError):
        solver.solve({"objects": [{"id": "M4_02", "x": 300, "y": 0}, {"id": "M4_02", "x": 350, "y": 0}]},
                     ["M4_02"], FLOOR_Z)
