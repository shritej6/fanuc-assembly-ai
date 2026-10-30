"""T30 CalibratedCellGeometry on the official calibration, and the real-model adapters' load guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from perception.cell.models import ModelLoadError
from perception.port.backends import DefectHeadClassifier, PoseFit, WristReading
from perception.port.errors import ModelOutputError, ObservationInputError, PerceptionPortError
from perception.port.geometry import TABLE_SURFACE_Z_MM, CalibratedCellGeometry
from robot.cell.calibration import geometry
from tests.port_fakes import observation

GEOMETRY = CalibratedCellGeometry()


def test_a_missing_depth_map_names_the_camera():
    obs = observation()
    del obs["depth"]["wrist"]
    with pytest.raises(ObservationInputError, match="wrist depth"):
        GEOMETRY.wrist_points(obs)


def test_table_boxes_are_projected_and_scaled_to_the_observation_image():
    obs = observation()
    point = np.array([[400.0, 0.0, TABLE_SURFACE_Z_MM]])
    uv = geometry.project(geometry.load()["table"], (point - (0, 0, geometry.UF0_Z_MM)) / 1000.0)[0]
    width, height = geometry.load()["table"]["size"]
    x0, y0, x1, y1 = GEOMETRY.table_box(obs, point)
    assert (x0, y0) == pytest.approx((uv[0] * 64 / width, uv[1] * 36 / height))
    assert (x1, y1) == pytest.approx((x0, y0))


def test_wrist_depth_becomes_uf0_points_at_the_observation_joints():
    obs = observation()
    obs["depth"]["wrist"][240, 424] = 400
    points, pixels = GEOMETRY.wrist_points(obs)
    assert points.shape == (1, 3) and tuple(pixels[0]) == (424, 240)
    assert np.isfinite(points).all()


def test_no_table_depth_on_a_box_is_a_port_error():
    with pytest.raises(PerceptionPortError):
        GEOMETRY.table_locate(observation(), (10.0, 10.0, 20.0, 20.0))


def test_zones_come_from_the_cell_layout():
    assert GEOMETRY.work_area[0][0] < GEOMETRY.work_area[0][1]


def test_model_readings_are_validated():
    with pytest.raises(ModelOutputError):
        WristReading("M1_02", 1.5, 0.0)
    with pytest.raises(ModelOutputError):
        PoseFit(np.eye(4), float("nan"))


def test_defect_head_needs_the_t27_model_code():
    with pytest.raises(ModelLoadError, match="T27"):
        DefectHeadClassifier(Path("/nonexistent"), ("M1_02",), 224)
