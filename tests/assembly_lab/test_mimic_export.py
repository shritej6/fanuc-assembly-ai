"""Mimic HDF5 -> GR00T LeRobot export: success + force guard filtering, units, wrap handling, format checks."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from sim.assembly_lab.mimic import export_lerobot as E
from sim.assembly_lab.mimic import lerobot_writer as W
from sim.assembly_lab.mimic.catalog import encode_plan_ids
from sim.assembly_lab.mimic.validate_lerobot import DatasetError, validate

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")
T, SIM_DT, FPS = 60, 0.2, 5


def _demo(g, success: bool, contact_peak: float, yaws: np.ndarray, rng) -> None:
    """Agent 3's layout (skills/hdf5_io.py): actions are act_tcp rows in mm / deg; obs joints / tcp_pose API units."""
    g.attrs["success"] = success
    g.attrs["num_samples"] = T
    xyz = np.stack([np.linspace(300.0, 400.0, T), np.linspace(-50.0, 50.0, T), np.full(T, -300.0)], 1)
    w = np.where(np.arange(T) % 2 == 0, 179.9, -179.9)          # fingers down, noisy: must not jump 360
    r = (yaws + 180.0) % 360.0 - 180.0                            # wrapped like a controller reports it
    grip = (np.arange(T) > T // 2).astype(float)
    g["actions"] = np.concatenate([xyz, w[:, None], np.zeros((T, 1)), r[:, None], grip[:, None]], 1)
    o = g.create_group("obs")
    o["tcp_pose"] = np.concatenate([xyz, w[:, None], np.zeros((T, 1)), r[:, None]], 1)
    o["joints"] = np.concatenate([rng.uniform(-90, 90, (T, 6)), grip[:, None]], 1)
    o["train_mask"] = np.arange(T) >= 2                            # first 2 rows only stage the case
    o["plan_ids"] = np.tile(encode_plan_ids("pick_insert", "M4_02", "F04", 0, "nominal"), (T, 1))
    o["plan_target"] = np.tile([300.4, -50.0, 37.0, 127.0, 400.0, 50.0, 90.0], (T, 1))
    contact = np.zeros((T, 3))
    contact[T - 5] = [contact_peak, 1.0, 0.01]
    o["sim_contact_peak"] = contact
    for cam in ("table_cam", "side_cam", "wrist_cam"):
        o[cam] = rng.integers(0, 255, (T, 24, 32, 3), dtype=np.uint8)


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "gen.hdf5"
    yaw_cross = np.linspace(170.0, 190.0, T)  # r crosses +180: must stay continuous
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        _demo(d.create_group("demo_0"), True, 10.0, yaw_cross, rng)
        _demo(d.create_group("demo_1"), True, 40.0, yaw_cross, rng)   # over the axial limit
        _demo(d.create_group("demo_2"), False, 1.0, yaw_cross, rng)   # task failed
        _demo(d.create_group("demo_3"), True, 2.0, np.full(T, -20.0), rng)
    limits = tmp_path / "limits.yaml"
    limits.write_text("axial_N: 15.0\nlateral_N: 5.0\ntorque_Nm: 0.5\nstatus: PRELIMINARY\nsource: test\n")
    return path, limits, tmp_path / "ds"


def test_frame_indices_label_end_of_interval():
    obs, act = E.frame_indices(60, 0.02, 10)
    assert obs.tolist() == list(range(0, 60, 5))
    assert act.tolist() == [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59]
    with pytest.raises(ValueError):
        E.frame_indices(60, 0.03, 10)


@needs_ffmpeg
def test_export_filters_and_validates(setup):
    path, limits, out = setup
    report = E.export([str(path)], out, SIM_DT, FPS, (64, 36), None, workers=1, limits_path=limits)
    assert (report["seen"], report["not_success"], report["force"], report["exported"]) == (4, 1, 1, 2)
    v = validate(out)
    assert v == {"episodes": 2, "frames": 2 * (T - 2), "tasks": 1}
    meta = [json.loads(x) for x in (out / "meta/t12_episodes.jsonl").read_text().splitlines()]
    assert [m["source"][1] for m in meta] == ["demo_0", "demo_3"]
    assert meta[0]["product"] == 4 and meta[0]["skill"] == "pick_insert"
    task = json.loads((out / "meta/tasks.jsonl").read_text().splitlines()[0])["task"]
    assert task == "pick M4_02 at (300, -50, 37) gripping X at roll 127, insert on F04 at (400, 50, 90)"


@needs_ffmpeg
def test_units_and_wrist_turn_continuity(setup):
    import pandas as pd

    path, limits, out = setup
    E.export([str(path)], out, SIM_DT, FPS, (64, 36), None, workers=1, limits_path=limits)
    df = pd.read_parquet(W.parquet_path(out, 0))
    state, action = np.stack(df["observation.state"]), np.stack(df["action"])
    assert len(df) == T - 2                                                        # train_mask applied
    assert np.allclose(state[0, 7:10], [300.0 + 2 * 100.0 / (T - 1), -50.0 + 2 * 100.0 / (T - 1), -300.0], atol=1e-3)
    assert np.allclose(state[:, 10], -180.0, atol=0.2) and np.allclose(action[:, 3], -180.0, atol=0.2)
    assert np.abs(np.diff(state[:, 12])).max() < 5 and np.abs(np.diff(action[:, 5])).max() < 5   # no 360 jump
    assert abs(action[0, 5] - state[0, 12]) < 5                                  # same wrist turn
    assert set(np.unique(action[:, 6])) <= {0.0, 1.0}
    assert np.allclose(action[:, :3], state[:, 7:10], atol=1e-3)                   # stride 1: row t at obs t


@needs_ffmpeg
def test_validator_catches_stale_stats(setup):
    path, limits, out = setup
    E.export([str(path)], out, SIM_DT, FPS, (64, 36), None, workers=1, limits_path=limits)
    info = json.loads((out / "meta/info.json").read_text())
    info["features"]["action"]["shape"] = [8]
    (out / "meta/info.json").write_text(json.dumps(info))
    with pytest.raises(DatasetError):
        validate(out)


def test_export_refuses_without_limits(setup):
    from sim.assembly_lab.mimic.force_guard import ForceLimitError

    path, _, out = setup
    with pytest.raises(ForceLimitError):
        E.export([str(path)], out, SIM_DT, FPS, (64, 36), None, workers=1, limits_path=out.parent / "missing.yaml")


def test_generation_rollup_measures_from_worker_stats(tmp_path):
    from sim.assembly_lab.mimic.run_generation import rollup

    for w, (acc, att, wall) in enumerate([(90, 100, 1800.0), (10, 50, 1800.0)]):
        d = tmp_path / "pick_insert" / "p4"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"w{w}.hdf5.stats.json").write_text(json.dumps({
            "skill": "pick_insert", "accepted": acc, "attempts": att, "wall_s": wall,
            "failed_gates": {"final_xy_mm": att - acc}}))
    s = rollup(tmp_path)["pick_insert"]
    assert s["acceptance_rate"] == round(100 / 150, 4) and s["demos_per_worker_hour"] == 100.0
    assert s["failed_gates"] == {"final_xy_mm": 50}


@needs_ffmpeg
def test_agent5_prepare_accepts_export(setup, tmp_path):
    """The binding check: agent 5's prepare_dataset.py (DATA_FORMAT.md) must accept the export unchanged."""
    prep = pytest.importorskip("policy_learn.groot.prepare_dataset")
    path, limits, out = setup
    E.export([str(path)], out, SIM_DT, FPS, (64, 36), None, workers=1, limits_path=limits)
    assert not (out / "meta/modality.json").exists()        # DATA_FORMAT.md: the prepare step writes it
    info = json.loads((out / "meta/info.json").read_text())
    assert info["features"]["observation.images.wrist"]["shape"] == [36, 64, 3] and info["fps"] == 5
    res = prep.prepare(out, tmp_path / "groot", 0.5, 0)
    assert res


def test_screen_rejects_dry_runs_and_missing_cameras(setup, tmp_path):
    from sim.assembly_lab.mimic.force_guard import load_limits

    path, limits, _ = setup
    with h5py.File(path, "a") as f:
        f["data/demo_0"].attrs["KINEMATIC_DRY_RUN"] = True
        del f["data/demo_3/obs/wrist_cam"]
    refs, counts = E.screen(str(path), load_limits(limits), "sim_contact_peak")
    assert refs == [] and counts["dry_run"] == 1 and counts["no_images"] == 1
