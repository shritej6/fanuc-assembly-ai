from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("trimesh")
pytest.importorskip("scipy")
pytest.importorskip("pxr")

from perception.pose import validate_foundationpose as v  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
NO_JITTER = {"translate_mm": [0.0, 0.0, 0.0], "rotate_deg": [0.0, 0.0, 0.0]}


def test_intrinsics_match_the_usd_camera_and_scale_per_axis_to_the_stored_depth():
    cfg = v.load_camera_config(REPO)
    full = v.intrinsics(cfg, tuple(cfg["resolution"]))
    assert full[0, 0] == pytest.approx(1027.6, abs=0.1)
    depth = v.intrinsics(cfg, v.DEPTH_SIZE)
    assert depth[0, 2] == 320 and depth[1, 2] == 200
    assert depth[0, 0] == pytest.approx(full[0, 0] * 640 / 1920)
    assert depth[1, 1] == pytest.approx(full[1, 1] * 400 / 1080)


def test_unjittered_camera_sits_at_the_eye_and_looks_at_the_target_with_opencv_axes():
    cfg = v.load_camera_config(REPO)
    cam = v.camera_to_world_cv(cfg, NO_JITTER)
    assert cam[:3, 3] == pytest.approx(np.array(cfg["eye"]), abs=1e-9)
    forward = np.array(cfg["look_at"]) - np.array(cfg["eye"])
    assert cam[:3, 2] == pytest.approx(forward / np.linalg.norm(forward), abs=1e-9)
    assert cam[:3, 1] @ np.array(cfg["up"]) < 0                  # image +Y points down


def test_errors_are_zero_for_the_truth_and_report_a_shift_in_millimetres():
    points = np.random.default_rng(0).normal(size=(500, 3)) * 0.02
    truth = np.eye(4)
    assert v.errors(truth, truth, points) == pytest.approx({"adds_mm": 0, "trans_mm": 0, "rot_deg": 0}, abs=1e-6)
    shifted = truth.copy()
    shifted[0, 3] = 0.010
    assert v.errors(shifted, truth, points)["trans_mm"] == pytest.approx(10.0)
