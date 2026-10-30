"""Runner, force guard, precision gates and HDF5 output, end to end on the kinematic test double (no physics)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from sim.assembly_lab.skills import checks, dry_run, episodes, frames, gates, record
from sim.assembly_lab.skills.force_guard import ForceGuard, LimitsError, load_limits
from sim.assembly_lab.skills.gates import GateFailed
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
    # the first 0.02 mm FINE row on the 10 N/mm spring gives 0.2 N: over this limit in one row, before any soft stop
    p.write_text(LIMITS.replace("axial_N: 15.0", "axial_N: 0.1"))
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


# A *_04 lock seated with its local z lying flat (local z along world -y): yaw must not leak into tilt.
LOCK_SEATED = [400.0, 0.0, -300.0, 90.0, 0.0, 0.0]


def _turned(pose, axis, deg):
    from sim.assembly_lab.skills.geometry import matrix_to_pose, pose_to_matrix

    m = pose_to_matrix(pose)
    k = np.asarray(axis, float)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    th = np.radians(deg)
    m[:3, :3] = (np.eye(3) + np.sin(th) * kx + (1 - np.cos(th)) * kx @ kx) @ m[:3, :3]
    return list(matrix_to_pose(m))


@pytest.mark.parametrize("part", ["M4_04", "M5_04", "M6_04"])
def test_lock_tilt_ignores_yaw(part):
    up = frames.installed_up(part)
    assert up == pytest.approx((0.0, 0.0, 1.0))
    yawed = _turned(LOCK_SEATED, (0, 0, 1), 30.0)
    assert gates.axis_tilt_deg(yawed, LOCK_SEATED, up) == pytest.approx(0.0, abs=1e-6)
    assert gates.yaw_err_deg(yawed, LOCK_SEATED) == pytest.approx(30.0, abs=1e-6)
    assert checks.seat_error(yawed, LOCK_SEATED, up)["tilt"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("axis", [(1, 0, 0), (0, 1, 0)])
def test_lock_tilted_1p5_deg_fails(axis):
    up = frames.installed_up("M5_04")
    tilted = _turned(LOCK_SEATED, axis, 1.5)
    log = gates.GateLog()
    with pytest.raises(gates.GateFailed, match="seat_tilt_deg"):
        gates.seat_gates(log, tilted, LOCK_SEATED, up=up)
    assert log.values["seat_tilt_deg"]["value"] == pytest.approx(1.5, abs=1e-3)
    assert not checks.seated_ok(tilted, LOCK_SEATED, up=up)[0]


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
    assert summary["1:M1_02"]["kept_or_would_keep"] == 2
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


def test_pre_touch_contact_is_reset_invalid_not_ignored(limits):
    g = ForceGuard(limits)
    over = {"F01": {"normal_N": 3.0, "lateral_N": 9.0, "torque_Nm": 0.0}}
    g.observe(0, {"sim": {"contact_peak": over, "robot_touch": {"F01": 0.0}}})
    assert g.pre_touch_violations and not g.violations and not g.touched
    assert g.pre_touch_peaks["F01"]["lateral_N"] == 9.0
    g.observe(1, {"sim": {"contact_peak": over, "robot_touch": {"F01": 0.2}}})
    assert g.touched and g.violations, "from first robot contact on the guard applies in full"


def test_no_touch_data_counts_everything_as_robot_contact(limits):
    g = ForceGuard(limits)
    g.observe(0, {"sim": {"contact_peak": {"F01": {"normal_N": 0, "lateral_N": 9.0, "torque_Nm": 0}}}})
    assert g.violations and not g.pre_touch_violations


def test_per_object_limit_is_tighter(limits):
    g = ForceGuard(limits, per_object={"M5_04": {"axial_N": 2.5}})
    g.observe(0, {"sim": {"contact_peak": {"M5_04": {"normal_N": 3.0, "lateral_N": 0, "torque_Nm": 0}}}})
    assert g.violations and g.violations[0].limit == 2.5


def test_stats_json_accepts_numpy_scalars(tmp_path):
    p = tmp_path / "product_1.hdf5"
    record._write_stats(p, "M1_01", "build", {"ok": np.bool_(True), "x": np.float32(0.5), "n": np.int64(3)})
    out = json.loads(p.with_name("product_1_stats.json").read_text())
    assert out["build"]["M1_01"] == {"ok": True, "x": 0.5, "n": 3}


def test_annotate_skips_empty_reset_episodes(tmp_path):
    from sim.assembly_lab.skills import hdf5_io

    p = tmp_path / "rec.hdf5"
    with h5py.File(p, "w") as f:
        f.create_group("data/demo_0/initial_state")          # a reset that never ran a step
        f.create_dataset("data/demo_1/actions", data=np.zeros((3, 7)))
    meta = {"steps": 3, "train_mask": [True] * 3, "would_keep": False, "reason": "x"}
    Path(str(p) + ".meta.jsonl").write_text(json.dumps(meta) + "\n")
    rep = hdf5_io.annotate_recorder_file(p)
    assert rep["episodes"] == 1 and rep["empty_episodes"] == 1 and not rep["step_mismatch"]
    with h5py.File(p) as f:
        assert f["data/demo_1"].attrs["reason"] == "x" and f["data/demo_1/obs/train_mask"].shape == (3,)


def test_release_opens_with_no_axial_preload(limits):
    """The push preloads the seated part (kinematic spring: 0.5 mm below seat = 5 N); the fingers open only after
    the TCP has backed off to the seated height, so nothing drags the part as they let go."""
    env, (pick, ins, *_) = _world()
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    episodes.pick_insert(rec, pick, ins)
    assert env.release_preload_N == pytest.approx(0.0)
    phases = [f.seg.phase for f in rec.frames]
    assert phases.index("relieve") < phases.index("release")
    assert not rec.gate_log.failed


class _NoCaseAtReset(record.KinematicBackend):
    """Isaac-style backend whose reset refuses the layout (the env's settled layout lacks the recovery case)."""

    def reset(self, rng, product, case):
        raise record.NoCase("not crowded")


def test_no_case_at_reset_is_skipped_not_a_crash(tmp_path):
    lim = tmp_path / "limits.yaml"
    lim.write_text(LIMITS)
    summary = record.record(_NoCaseAtReset(), "crowded", [1], 1, tmp_path, 0, 2, lim)
    part = summary["1:M1_02"]
    assert part["attempts"] == 0 and part["planner_skipped"] == 11
    assert "never produced the case" in part["aborted"]


def test_motion_rule_break_is_a_robot_failure_not_a_crash(tmp_path, monkeypatch):
    from sim.assembly_lab.skills.motion import MotionError

    def tilted(*args, **kwargs):
        raise MotionError("place_free: tool 7.31 deg from straight down > 6.0")

    monkeypatch.setattr(record, "_episode", tilted)
    lim = tmp_path / "limits.yaml"
    lim.write_text(LIMITS)
    part = record.record(record.KinematicBackend(), "crowded", [1], 1, tmp_path, 0, 2, lim)["1:M1_02"]
    assert part["attempts"] == 2 and part["robot_failed"] == 2 and part["kept_or_would_keep"] == 0


def test_wrist_flip_is_a_motion_error_not_a_demo(limits):
    """check10: J6 spun 352 deg in 4 rows at the tilt row; a > 20 deg/row joint jump must end the attempt."""
    from sim.assembly_lab.skills.motion import MotionError

    env, (pick, ins, *_) = _world()
    real_step, calls = env.step_tcp, {"n": 0}

    def flipping_step(rows, dt):
        obs = real_step(rows, dt)
        calls["n"] += 1
        if calls["n"] == 30:
            obs["joints"] = [0.0] * 5 + [90.0, obs["joints"][6]]
        return obs

    env.step_tcp = flipping_step
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    with pytest.raises(MotionError, match="J6 moved 90.0 deg"):
        episodes.pick_insert(rec, pick, ins)


def test_tracking_gate_rejects_an_arm_that_misses_held_commands(limits):
    """check10b: the arm sat ~0.9 mm off its held commands (J5 droop); a kept demo needs <= 0.1 mm."""
    env, (pick, ins, *_) = _world()
    real_observe = env.observe

    def drooping():
        obs = real_observe()
        obs["tcp_pose"] = list(np.asarray(obs["tcp_pose"]) + [0.6, -0.5, 0.0, 0, 0, 0])
        return obs

    env.observe = drooping
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    with pytest.raises(EpisodeFailed):  # the skewed in-hand estimate already spoils the seat here
        episodes.pick_insert(rec, pick, ins)
    assert rec.tracking_mm == pytest.approx(0.781, abs=1e-3)
    with pytest.raises(GateFailed, match="tracking_mm=0.781"):
        rec.finish_gates()


def test_tracking_gate_passes_an_arm_on_its_commands(limits):
    env, (pick, ins, *_) = _world()
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    episodes.pick_insert(rec, pick, ins)
    assert rec.gate_log.values["tracking_mm"] == {"value": 0.0, "limit": 0.1, "pass": True}


def test_in_hand_drift_is_zero_for_a_held_part_and_logged(limits):
    env, (pick, ins, *_) = _world()
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    episodes.pick_insert(rec, pick, ins)
    assert rec.in_hand_drift_mm["carry"] == pytest.approx(0.0, abs=1e-9)
    assert rec.in_hand_drift_mm["contact"] == pytest.approx(0.1, abs=1e-3)  # the 0.1 mm seat push (PUSH_MM)
    assert rec.holding is None  # released at the seat


def test_in_hand_drift_measures_a_part_slipping_in_the_fingers(limits):
    """Pinch is ~14.5 N/finger (target ~34): a part sliding in the fingers on the carry must show in the metadata."""
    env, (pick, ins, *_) = _world()
    real_observe, state = env.observe, {"rec": None}

    def slipping():
        obs = real_observe()
        rec = state["rec"]
        if rec is not None and rec.holding == pick.object_id:
            poses = dict(obs["sim"]["object_poses"])
            poses[pick.object_id] = list(np.asarray(poses[pick.object_id], dtype=float) + [0.3, 0.4, 0, 0, 0, 0])
            obs["sim"] = {**obs["sim"], "object_poses": poses}
        return obs

    env.observe = slipping
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    state["rec"] = rec
    try:
        episodes.pick_insert(rec, pick, ins)
    except (EpisodeFailed, GateFailed):
        pass  # the shifted part misses its seat; the drift is what this test checks
    assert rec.in_hand_drift_mm["carry"] == pytest.approx(0.5, abs=1e-3)


def test_isaac_out_paths_differ_for_jobs_started_together(tmp_path):
    """run10: build p1 and wont_seat p1 started in the same second and both opened one HDF5 (h5py lock error)."""
    a = record.isaac_out_path(tmp_path, "build", 1, 7)
    b = record.isaac_out_path(tmp_path, "wont_seat", 1, 7)
    c = record.isaac_out_path(tmp_path, "build", 1, 8)
    assert a.parent == b.parent == tmp_path / "pick_insert"
    assert len({a, b, c}) == 3 and a.name.startswith("product_1_")


def test_press_steps_drop_to_fine_rows_after_first_seat_contact(limits):
    """run10: 0.05 mm rows into a ~260 N/mm seat overshot to 6-11.6 N; after first contact rows are <= 0.02 mm."""
    from sim.assembly_lab.skills.runner import FINE_PRESS_MM, SEAT_PHASES

    env, (pick, ins, *_) = _world()
    real_step = env.step_tcp
    touch_z = ins.seated_tcp[2] + 0.5  # the part meets the seat 0.5 mm above the CAD seated height

    def seat_contact(rows, dt):
        obs = real_step(rows, dt)
        if np.asarray(rows)[-1, 2] <= touch_z and np.asarray(rows)[-1, 6] > 0.5:
            obs["sim"]["contact_peak"][ins.object_id]["normal_N"] = 1.0
        return obs

    env.step_tcp = seat_contact
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    episodes.pick_insert(rec, pick, ins)
    seat = [f for f in rec.frames if f.seg.phase in SEAT_PHASES]
    z = np.array([f.action[2] for f in seat])
    first = int(np.argmax(z <= touch_z))
    steps = np.abs(np.diff(z[first:]))
    assert steps.size and steps.max() <= FINE_PRESS_MM + 1e-9
    assert any("pressing at 0.02 mm/row" in e for e in rec.events)
    assert z[-1] == pytest.approx(ins.seated_tcp[2] - 0.1)  # still ends on the planned push height


def test_jaw_close_is_in_the_substep_trace(limits):
    """run10 M2_01 hit 20.7 N during the grasp before any insert: the close rows must be traced per substep."""
    env, (pick, ins, *_) = _world()
    real_step = env.step_tcp

    def with_substeps(rows, dt):
        obs = real_step(rows, dt)
        obs["sim"]["substep_contact"] = {pick.object_id: [[3.0, 0.5, 0.0]] * 4}
        return obs

    env.step_tcp = with_substeps
    rec = EpisodeRecorder(env, ForceGuard(limits), env.reset())
    episodes.pick_insert(rec, pick, ins)
    assert any(t["phase"] == "close" for t in rec.substep_trace)


def test_isaac_runs_close_one_file_per_part(tmp_path, monkeypatch):
    """Mimic reads closed files while run10 goes on: every part gets its own recorder file, rotated between parts."""
    class Rotating:
        name = "isaac"
        products = (1,)

        def __init__(self):
            self.rotations = []

        def parts(self, product):
            return ["M1_01", "M1_02", "M1_03"]

        def rotate(self, next_path):
            self.rotations.append(next_path)

    seen = []

    def fake_part(backend, scenario, product, part, target, rng, run):
        seen.append((part, run.path, len(run.metas)))
        run.metas.append({"part": part})
        return {"attempts": 1, "kept_or_would_keep": 1}

    monkeypatch.setattr(record, "_record_part", fake_part)
    lim = tmp_path / "limits.yaml"
    lim.write_text(LIMITS)
    base = record.isaac_out_path(tmp_path, "build", 1, 7)
    backend = Rotating()
    record.record(backend, "build", [1], 3, tmp_path, 7, 8, lim, path=base)
    paths = [p for _, p, _ in seen]
    assert [p.name for p in paths] == [f"{base.stem}_{part}.hdf5" for part in ("M1_01", "M1_02", "M1_03")]
    assert backend.rotations == paths[1:]  # the first part records into the file the env was built with
    assert [n for *_, n in seen] == [0, 0, 0]  # metadata restarts with each file (demo_0 is the file's first)
    for p in paths:
        assert p.with_name(p.stem + "_stats.json").exists()
