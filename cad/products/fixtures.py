"""One generator for the six reusable printed fixtures (60 x 50 x 8 plate, top at Z = 0)."""

from __future__ import annotations

import cadquery as cq
from common_geometry import EPS, Part, box_span, cyl, rounded_box
from parameters import (
    FIXTURE_CORNER_R,
    FIXTURE_HOLE_D,
    FIXTURE_HOLE_X,
    FIXTURE_HOLE_Y,
    FIXTURE_X,
    FIXTURE_Y,
    FIXTURE_Z,
    FRONT_STOP_LEG,
    FRONT_STOP_WALL,
    LOCATING_CLEARANCE_PER_SIDE,
    PEG_A_D,
    PEG_A_X,
    PEG_A_Y,
    PEG_B_D,
    PEG_B_X,
    PEG_B_Y,
    PEG_HEIGHT,
    PEG_TOP_CHAMFER,
    SIDE_STOP_CENTER_Y,
    SIDE_STOP_GAP,
    SIDE_STOP_X,
    SIDE_STOP_Y,
    STOP_HEIGHT,
)

FIXTURE_NAMES = {
    1: "F01_Rotary_Fixture",
    2: "F02_Indicator_Fixture",
    3: "F03_Pushbutton_Fixture",
    4: "F04_Sensor_Connector_Fixture",
    5: "F05_IO_Connector_Fixture",
    6: "F06_Power_Connector_Fixture",
}


def _peg(d: float, x: float, y: float) -> cq.Workplane:
    return cyl(d, PEG_HEIGHT + EPS, x, y, -EPS).faces(">Z").chamfer(PEG_TOP_CHAMFER)


def _front_l_stop(side: int, hx: float, hy: float) -> cq.Workplane:
    """L stop hugging one front corner of the housing footprint (side = -1 left, +1 right)."""
    c = LOCATING_CLEARANCE_PER_SIDE
    xi, yi = side * (hx / 2 + c), -(hy / 2 + c)            # inner corner (touches clearance gap)
    xo, yo = side * (hx / 2 + c + FRONT_STOP_WALL), yi - FRONT_STOP_WALL
    x_leg_end = xo - side * FRONT_STOP_LEG
    front_leg = box_span(min(xo, x_leg_end), max(xo, x_leg_end), yo, yi, -EPS, STOP_HEIGHT)
    side_leg = box_span(min(xo, xi), max(xo, xi), yo, yo + FRONT_STOP_LEG, -EPS, STOP_HEIGHT)
    return front_leg.union(side_leg)


def make_fixture(model_no: int, housing_x: float, housing_y: float) -> Part:
    plate = rounded_box(FIXTURE_X, FIXTURE_Y, FIXTURE_Z, FIXTURE_CORNER_R, z0=-FIXTURE_Z)
    for sx in (-1, 1):
        for sy in (-1, 1):
            plate = plate.cut(cyl(FIXTURE_HOLE_D, FIXTURE_Z + 2, sx * FIXTURE_HOLE_X,
                                  sy * FIXTURE_HOLE_Y, -FIXTURE_Z - 1))
    plate = plate.union(_peg(PEG_A_D, PEG_A_X, PEG_A_Y)).union(_peg(PEG_B_D, PEG_B_X, PEG_B_Y))
    for side in (-1, 1):
        plate = plate.union(_front_l_stop(side, housing_x, housing_y))
        sx_c = side * (housing_x / 2 + SIDE_STOP_GAP)
        plate = plate.union(box_span(sx_c - SIDE_STOP_X / 2, sx_c + SIDE_STOP_X / 2,
                                     SIDE_STOP_CENTER_Y - SIDE_STOP_Y / 2,
                                     SIDE_STOP_CENTER_Y + SIDE_STOP_Y / 2, -EPS, STOP_HEIGHT))
    # Fixture keeps the world frame (top surface Z = 0) so its STEP lines up with every housing.
    return Part(FIXTURE_NAMES[model_no], plate.clean(), (0.0, 0.0, 0.0))
