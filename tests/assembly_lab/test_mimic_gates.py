"""Precision gates (INTERFACES.md) and the MimicGen core transforms used by mimic/generate.py."""

from __future__ import annotations

import numpy as np
import pytest

from sim.assembly_lab.mimic import gates as G
from sim.assembly_lab.mimic import mimicgen as M
from sim.assembly_lab.mimic.force_guard import ForceLimits

LIM = G.GateLimits()
SEATED = np.array([400.0, 50.0, -320.0, 0.0, 0.0, 90.0])


@pytest.mark.parametrize("delta, failed", [
    ([0.2, 0.1, 0.2, 0.0, 0.0, 0.8], []),
    ([0.3, 0.1, 0.0, 0.0, 0.0, 0.0], ["final_xy_mm"]),           # 0.316 mm > 0.3
    ([0.0, 0.0, 0.0, 0.0, 0.0, 1.2], ["final_yaw_deg"]),
    ([0.0, 0.0, -0.35, 0.0, 0.0, 0.0], ["final_dz_mm"]),
    ([0.0, 0.0, 0.0, 1.1, 0.0, 0.0], ["final_tilt_deg"]),
])
def test_final_pose_gates(delta, failed):
    rep = G.GateReport(ok=False)
    G.gate_final_pose(rep, SEATED + np.array(delta), SEATED, LIM)
    assert G.finish(rep).failed == failed


LOCK_SEATED = np.array([400.0, 0.0, -300.0, 90.0, 0.0, 0.0])  # a *_04 lock: local z lies flat when installed


def _turned(pose, axis, deg):
    from sim.assembly_lab.skills.geometry import matrix_to_pose, pose_to_matrix

    m = pose_to_matrix(pose)
    k = np.asarray(axis, float)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = np.radians(deg)
    m[:3, :3] = (np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx) @ m[:3, :3]
    return np.asarray(matrix_to_pose(m))


def test_lock_yaw_does_not_leak_into_tilt():
    rep = G.GateReport(ok=False)
    G.gate_final_pose(rep, _turned(LOCK_SEATED, (0, 0, 1), 30.0), LOCK_SEATED, LIM, tilt_only=True)
    assert rep.values["final_tilt_deg"] == pytest.approx(0.0, abs=1e-6)
    assert G.finish(rep).failed == []


def test_lock_tilted_1p5_deg_fails():
    rep = G.GateReport(ok=False)
    G.gate_final_pose(rep, _turned(LOCK_SEATED, (1, 0, 0), 1.5), LOCK_SEATED, LIM)
    assert rep.values["final_tilt_deg"] == pytest.approx(1.5, abs=1e-3)
    assert "final_tilt_deg" in G.finish(rep).failed


def test_force_other_objects_and_touch_gates():
    rep = G.GateReport(ok=False)
    G.gate_forces(rep, np.array([[1.0, 1.0, 0.01], [14.0, 4.6, 0.05]]), ForceLimits(15, 4.5, 0.1, "PRELIMINARY", "t"))
    still = np.tile([300.0, 0.0, -330.0, 0.0, 0.0, 10.0], (5, 1))
    bumped = still.copy()
    bumped[2, 0] += 1.2                                           # pushed 1.2 mm, then settled back
    G.gate_others(rep, {"M4_02": still + 50, "F04": still, "M1_01": bumped}, "M4_02", LIM)
    G.gate_touch(rep, [None, "M4_02", "M1_01", None], "M4_02")
    G.finish(rep)
    assert rep.failed == ["peak_lateral_N", "others_move_mm", "fingers_touched_non_target"]
    assert rep.values["others_worst"] == "M1_01" and not rep.ok


def test_grip_gate():
    rep = G.GateReport(ok=False)
    planned = np.array([0.0, 0.0, 13.9, -180.0, 0.0, 90.0])
    G.gate_grip(rep, planned + [0.3, 0.2, 0.1, 0, 0, 0.5], planned, LIM)
    assert G.finish(rep).ok
    rep = G.GateReport(ok=False)
    G.gate_grip(rep, planned + [0.6, 0.0, 0.0, 0, 0, 0.0], planned, LIM)
    assert G.finish(rep).failed == ["grip_pos_mm"]


def test_transform_keeps_motion_relative_to_frame():
    rng = np.random.default_rng(0)
    src_frame = M.pose_to_matrix([300.0, -50.0, -320.0, -180.0, 0.0, 30.0])
    new_frame = M.pose_to_matrix([420.0, 80.0, -320.0, -180.0, 0.0, 170.0])
    rows = np.c_[np.tile([300.0, -50.0, -320.0], (6, 1)) + rng.normal(0, 5, (6, 3)), np.tile([-180.0, 0.0, 30.0], (6, 1)),
                 np.r_[np.zeros(3), np.ones(3)]]
    out = M.transform_rows(rows, src_frame, new_frame)
    for a, b in zip(rows, out):
        rel_src = np.linalg.inv(src_frame) @ M.pose_to_matrix(a[:6])
        rel_new = np.linalg.inv(new_frame) @ M.pose_to_matrix(b[:6])
        assert np.allclose(rel_src, rel_new, atol=1e-9)
    assert np.array_equal(out[:, 6], rows[:, 6]) and np.abs(np.diff(out[:, 5])).max() < 1e-6


def test_boundaries_and_selection():
    sig = [np.r_[np.zeros(4), np.ones(6)], np.r_[np.zeros(7), np.ones(3)]]
    assert M._boundaries(sig, 10) == [(0, 4), (4, 7), (7, 10)]
    with pytest.raises(ValueError):
        M._boundaries([sig[1], sig[0]], 10)
    near = M.pose_to_matrix([400.0, 0, -320, -180, 0, 0])
    far = M.pose_to_matrix([200.0, 0, -320, -180, 0, 0])
    segs = [M.SourceSegment("a", "M4_02", 4, 0, np.zeros((1, 7)), far),
            M.SourceSegment("b", "M4_02", 4, 0, np.zeros((1, 7)), near),
            M.SourceSegment("c", "M1_01", 1, 0, np.zeros((1, 7)), near)]
    pick = M.select_source(segs, M.pose_to_matrix([401.0, 0, -320, -180, 0, 0]), "M4_02", 1, np.random.default_rng(0))
    assert pick.demo == "b"                                       # same part, nearest frame


def test_interpolate_and_noise():
    a = np.array([300.0, 0, -200, -180, 0, 170, 0])
    b = np.array([340.0, 40, -240, -180, 0, -170, 1])
    mid = M.interpolate(a, b, 3)
    assert mid.shape == (3, 7) and np.allclose(mid[1, :3], (a[:3] + b[:3]) / 2) and (mid[:, 6] == 0).all()
    assert np.allclose(np.diff(np.r_[170.0, mid[:, 5], 190.0]), 5.0, atol=1e-6)   # shortest way through +180
    rng = np.random.default_rng(0)
    rows = np.tile(a, (2000, 1))
    noisy = M.add_noise(rows, 0.1, 0.1, rng)
    assert abs(np.std(noisy[:, 0] - rows[:, 0]) - 0.1) < 0.01 and np.abs(noisy[:, 5] - 170).max() < 1.0


def test_generator_dry_run_on_agent3_kinematic_seeds(tmp_path):
    """Glue check only (KINEMATIC_DRY_RUN, no physics): agent 3 seeds -> MimicGen -> agent 3 HDF5 + gates."""
    import argparse
    import json

    h5py = pytest.importorskip("h5py")
    record = pytest.importorskip("sim.assembly_lab.skills.record")
    from sim.assembly_lab.mimic import generate
    from sim.assembly_lab.skills.backends import KinematicBackend

    record.record(KinematicBackend(), "build", [1], 4, tmp_path / "demos", 0, 20, None, True)
    seeds = tmp_path / "demos/pick_insert/product_1.hdf5"
    out = tmp_path / "mimic/pick_insert/p1/w0.hdf5"
    a = argparse.Namespace(skill="pick_insert", product=1, part=None, seeds=seeds, out=out, target=6,
                           max_attempts=12, seed=3, backend="kinematic", limits=None, allow_touch_proxy=True)
    stats = generate.run(a)
    assert stats["attempts"] >= stats["accepted"] == 6
    with h5py.File(out, "r") as f:
        demo = f["data/demo_0"]
        assert demo.attrs["KINEMATIC_DRY_RUN"] and demo.attrs["generator"] == "mimicgen"
        assert {"grasped", "seated"} <= set(demo["obs/datagen_info/subtask_term_signals"].keys())
        assert json.loads(demo.attrs["gates"])["seat_xy_mm"]["pass"]


def test_generate_seat_gates_use_the_parts_own_up(monkeypatch):
    """mimic/generate.py passes installed_up(part): with a part whose installed-up is world +x, a 30 deg turn about
    world z is a 30 deg tilt (it would read 0 with an assumed +z)."""
    from types import SimpleNamespace

    from sim.assembly_lab.mimic import generate
    from sim.assembly_lab.skills import frames, gates

    monkeypatch.setattr(frames, "installed_up", lambda oid: (1.0, 0.0, 0.0))
    plan = SimpleNamespace(seated_object_pose=LOCK_SEATED, yaw_tol_deg=None)
    log = gates.GateLog()
    with pytest.raises(gates.GateFailed, match="seat_tilt_deg"):
        generate.seat_gates(log, _turned(LOCK_SEATED, (0, 0, 1), 30.0), plan, "M5_04")
    assert log.values["seat_tilt_deg"]["value"] == pytest.approx(30.0, abs=1e-3)
