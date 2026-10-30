"""Runner, force guard, precision gates and HDF5 output, end to end on the kinematic test double (no physics)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from sim.assembly_lab.skills import dry_run, episodes, gates, record
from sim.assembly_lab.skills.force_guard import ForceGuard, LimitsError, load_limits
from sim.assembly_lab.skills.runner import EpisodeFailed, EpisodeRecorder

LIMITS = "axial_N: 15.0\nlateral_N: 4.5\ntorque_Nm: 0.1\nstatus: PRELIMINARY\nsource: test\n"


@pytest.fixture
def limits(tmp_path):
    p = tmp_path / "limits.yaml"
    p.write_text(LIMITS)
    return load_limits(p)


def _world():
    objs, seats = dry_run.demo_world()
    env = dry_run.KinematicEnv(objs, seats)
    return env, dry_run.demo_plans(objs, seats)


def test_limits_required(tmp_path):
    with pytest.raises(LimitsError):
        load_limits(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("axial_N: 15\n")
    with pytest.raises(LimitsError):
        load_limits(tmp_path / "bad.yaml")


def test_guard_flags_and_needs_contact(limits):
    g = ForceGuard(limits)
    g.observe(0, {"sim": {"contact_peak": {"A": {"normal_N": 3.0, "lateral_N": 1.0, "torque_Nm": 0.01}}}})
    assert g.ok
    g.observe(1, {"sim": {"contact_peak": {"A": {"normal_N": 3.0, "lateral_N": 4.6, "torque_Nm": 0.01}}}})
    assert not g.ok and g.violations[0].axis == "lateral_N"
    with pytest.raises(LimitsError):
        ForceGuard(limits).observe(0, {"sim": {}})


def test_pick_insert_episode_signals_and_gates(limits):
    env, (pick, ins, *_) = _world()
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    episodes.pick_insert(rec, pick, ins)
    assert rec.marks["grasped"] < rec.marks["seated"] < len(rec.frames)
    assert not rec.gate_log.failed
    for name in ("seat_xy_mm", "seat_yaw_deg", "seat_z_mm", "seat_tilt_deg", "other_moved_mm", "grip_pad_mm",
                 "grip_roll_deg", "touch_non_target_N"):
        assert name in rec.gate_log.values, name
    assert rec.gate_log.touch_method == "robot_touch"
    assert rec.instructions[0].startswith("pick M1_02 at")


def test_force_over_limit_discards(limits, tmp_path):
    p = tmp_path / "tight.yaml"
    p.write_text(LIMITS.replace("axial_N: 15.0", "axial_N: 2.0"))
    env, (pick, ins, *_) = _world()
    rec = EpisodeRecorder(env, ForceGuard(load_limits(p)), env.reset())
    with pytest.raises(EpisodeFailed, match="force guard"):
        episodes.pick_insert(rec, pick, ins)


def test_wont_seat_retries_once_and_masks_the_staged_try(limits):
    env, (pick, ins, *_) = _world()
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    episodes.pick_insert(rec, pick, ins, "wont_seat", misalign=(2.5, 0.0, 0.0))
    assert any("not seated on try 1" in e for e in rec.events)
    mask = np.array([f.seg.learn for f in rec.frames])
    assert (~mask).any() and mask[-1], "staged rows are masked, the recovery is learned"
    assert not rec.gate_log.failed


def test_seat_gate_rejects_0p4mm():
    log = gates.GateLog()
    with pytest.raises(gates.GateFailed, match="seat_xy_mm"):
        gates.seat_gates(log, [100.4, 0, 0, 0, 0, 0], [100.0, 0, 0, 0, 0, 0])
    assert log.values["seat_xy_mm"]["value"] == pytest.approx(0.4)
    log = gates.GateLog()
    with pytest.raises(gates.GateFailed, match="seat_yaw_deg"):
        gates.seat_gates(log, [0, 0, 0, 0, 0, 1.2], [0, 0, 0, 0, 0, 0])


def test_other_moved_gate():
    log = gates.GateLog()
    start = {"A": [0, 0, 0, 0, 0, 0], "B": [50, 0, 0, 0, 0, 0]}
    with pytest.raises(gates.GateFailed):
        gates.others_moved(log, start, {"A": [9, 9, 9, 0, 0, 0], "B": [51.2, 0, 0, 0, 0, 0]}, {"A"})
    assert log.values["other_moved_mm"]["object"] == "B"


def test_touch_proxy_needs_explicit_permission():
    t = gates.TouchMonitor(allow_proxy=False)
    t.observe({"sim": {"gripper_obj": "A"}}, {"A"})
    with pytest.raises(gates.GateFailed):
        t.finish(gates.GateLog())
    t = gates.TouchMonitor(allow_proxy=True)
    t.observe({"sim": {"gripper_obj": "A"}}, {"A"})
    log = gates.GateLog()
    t.finish(log)
    assert log.touch_method == "proxy:gripper_obj"
    t = gates.TouchMonitor(allow_proxy=False)
    t.observe({"sim": {"robot_touch": {"A": 2.0, "B": 0.3}}}, {"A"})
    with pytest.raises(gates.GateFailed):
        t.finish(gates.GateLog())


@pytest.mark.parametrize("scenario,seed_dir,signals", [
    ("build", "pick_insert", {"grasped", "seated"}),
    ("fix_above", "pull_out", {"grasped", "pulled", "placed"}),
    ("crowded", "place_free", {"grasped", "placed"}),
    ("defect", "defect_drop", {"grasped", "dropped"}),
])
def test_record_writes_mimic_hdf5(tmp_path, scenario, seed_dir, signals):
    lim = tmp_path / "limits.yaml"
    lim.write_text(LIMITS)
    summary = record.record(record.KinematicBackend(), scenario, [1], 2, tmp_path, 0, 5, lim)
    assert summary["1:M1_02"]["saved"] == 2
    path = tmp_path / seed_dir / "product_1.hdf5"
    with h5py.File(path) as f:
        data = f["data"]
        assert json.loads(data.attrs["env_args"])["eef_name"] == "er4ia"
        assert data.attrs["total_demos"] == 2
        d = data["demo_0"]
        n = d.attrs["num_samples"]
        assert d["actions"].shape == (n, 7) and bool(d.attrs["success"]) and d.attrs["KINEMATIC_DRY_RUN"]
        assert set(d["obs/datagen_info/subtask_term_signals"]) == signals
        for s in signals:
            sig = d[f"obs/datagen_info/subtask_term_signals/{s}"][()]
            assert sig[-1] and not sig[0] and (np.diff(sig.astype(int)) >= 0).all()
        assert d["obs/datagen_info/eef_pose/er4ia"].shape == (n, 4, 4)
        for k in ("eef_pos", "eef_quat", "arm_joint_pos", "gripper_state", "plan_ids", "plan_target",
                  "sim_contact_peak", "train_mask"):
            assert d[f"obs/{k}"].shape[0] == n, k
        assert json.loads(d.attrs["gates"])
    stats = json.loads(Path(str(path).replace(".hdf5", "_stats.json")).read_text())
    runs = stats[scenario]["M1_02"]["runs"]
    assert all("gates" in r for r in runs)


def test_kinematic_never_writes_real_demo_dir():
    with pytest.raises(SystemExit):
        record.main(["--backend", "kinematic", "--out", str(record.DEMO_ROOT / "x")])
