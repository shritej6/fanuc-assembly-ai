"""Demo timing end to end: recorded rows, exported LeRobot rows and the runtime's playback share one clock.

The recorder writes one row per Isaac env step (skills/motion.py DT_S); the export keeps every stride-th row so each
exported row is one runtime step (policy_learn/deploy/groot_policy.py DT_S, the real cell's 5 Hz). A waypoint the
recorded arm reached at wall time t must be the exported action the runtime reaches at the same t.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pd = pytest.importorskip("pandas")
pytest.importorskip("pyarrow")

from policy_learn.deploy import groot_policy
from sim.assembly_lab.mimic import export_lerobot as E
from sim.assembly_lab.mimic import lerobot_writer as W
from sim.assembly_lab.mimic.catalog import encode_plan_ids
from sim.assembly_lab.skills import motion

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")
RUNTIME_DT = groot_policy.DT_S
X_MM_PER_S = 50.0


def x_at(t: np.ndarray, end_s: float) -> np.ndarray:
    """The recorded arm's x at wall time t: constant speed, then it holds the last target (DATA_FORMAT.md)."""
    return 300.0 + X_MM_PER_S * np.minimum(t, end_s)


def write_recording(path, n_rows: int, row_dt: float, env_args: dict | None) -> None:
    """One kept demo in the Isaac recorder layout: obs row i at i * row_dt, action i = the pose at (i + 1) * row_dt."""
    t_obs = np.arange(n_rows) * row_dt
    end_s = n_rows * row_dt
    tcp = np.zeros((n_rows, 6))
    tcp[:, 0], tcp[:, 1], tcp[:, 2], tcp[:, 3] = x_at(t_obs, end_s), 0.0, -300.0, -180.0
    act = np.zeros((n_rows, 7))
    act[:, :6] = tcp
    act[:, 0] = x_at(t_obs + row_dt, end_s)
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        if env_args is not None:
            d.attrs["env_args"] = json.dumps(env_args)
        g = d.create_group("demo_0")
        g.attrs["success"] = True
        g["actions"] = act
        o = g.create_group("obs")
        o["tcp_pose"] = tcp
        o["joints"] = np.zeros((n_rows, 7))
        o["plan_ids"] = np.tile(encode_plan_ids("pick_insert", "M4_02", "F04", 0, "nominal"), (n_rows, 1))
        o["plan_target"] = np.tile([300.4, -50.0, 37.0, 127.0, 400.0, 50.0, 90.0], (n_rows, 1))
        o["sim_contact_peak"] = np.zeros((n_rows, 3))
        for cam in ("table_cam", "side_cam", "wrist_cam"):
            o[cam] = np.zeros((n_rows, 24, 32, 3), dtype=np.uint8)


@pytest.fixture
def limits(tmp_path):
    p = tmp_path / "limits.yaml"
    p.write_text("axial_N: 15.0\nlateral_N: 5.0\ntorque_Nm: 0.5\nstatus: PRELIMINARY\nsource: test\n")
    return p


def test_export_defaults_agree_with_recorder_and_runtime():
    assert E.SIM_DT_S == motion.DT_S                                   # one recorded row = one Motion row
    assert E.FPS * RUNTIME_DT == pytest.approx(1.0)                    # one exported row = one runtime step
    stride = round(RUNTIME_DT / E.SIM_DT_S)
    assert stride * E.SIM_DT_S == pytest.approx(RUNTIME_DT)            # whole recorded rows per exported row


def test_recorded_dt_reads_the_skills_recorder_step(tmp_path):
    path = tmp_path / "rec.hdf5"
    write_recording(path, 4, 0.2, {"env_name": "T12", "dt_s": 0.2})
    assert E.recorded_dt(str(path)) == pytest.approx(0.2)
    write_recording(path, 4, 0.1, {"sim_args": {"dt": 0.002, "decimation": 50}, "dt_s": 0.2})
    assert E.recorded_dt(str(path)) == pytest.approx(0.1)              # the simulator's own step wins


def test_export_refuses_a_recording_at_another_row_period(tmp_path, limits):
    path = tmp_path / "old.hdf5"
    write_recording(path, 20, 0.2, {"env_name": "T12", "dt_s": 0.2})   # pre-3172a6c rows: 0.2 s each
    with pytest.raises(ValueError, match="records one row per 0.2"):
        E.export([str(path)], tmp_path / "ds", E.SIM_DT_S, E.FPS, (32, 24), None, workers=1, limits_path=limits)


@needs_ffmpeg
@pytest.mark.parametrize(("row_dt", "n_rows"), [(motion.DT_S, 61), (0.2, 30)])
def test_waypoints_reached_at_the_same_wall_time_after_export(tmp_path, limits, row_dt, n_rows):
    path, out = tmp_path / "rec.hdf5", tmp_path / "ds"
    write_recording(path, n_rows, row_dt, {"env_name": "T12", "dt_s": row_dt})
    E.export([str(path)], out, row_dt, E.FPS, (32, 24), None, workers=1, limits_path=limits)
    df = pd.read_parquet(W.parquet_path(out, 0))
    ts = df["timestamp"].to_numpy(np.float64)
    state, action = np.stack(df["observation.state"]), np.stack(df["action"])
    end_s = n_rows * row_dt
    assert np.allclose(np.diff(ts), RUNTIME_DT, atol=1e-6)             # exported spacing == runtime step
    assert ts[-1] + RUNTIME_DT == pytest.approx(np.ceil(end_s / RUNTIME_DT - 1e-9) * RUNTIME_DT)  # nothing dropped
    assert np.allclose(state[:, 7], x_at(ts, end_s), atol=1e-3)        # frame k is the arm at k * runtime DT
    assert np.allclose(action[:, 0], x_at(ts + RUNTIME_DT, end_s), atol=1e-3)  # reached when the runtime gets there
