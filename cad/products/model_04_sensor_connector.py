"""Model 4 - Sensor Connector Module: housing, sensor insert, upper guide, top lock."""

from __future__ import annotations

import cadquery as cq
from common_geometry import EPS, Part, box_span, profile_xz
from connector_family import (
    FRONT,
    INSERT_TOP,
    LOCK_BRIDGE_Z1,
    connector_housing,
    connector_insert,
    guide_corner_notch,
    guide_frame,
    left_rib,
    rear_rib,
    top_lock,
)
from fixtures import make_fixture
from parameters import (
    CONN_FLOOR_Z,
    CONN_HOUSING_X,
    CONN_HOUSING_Y,
    CONN_HOUSING_Z,
    LOCATING_CLEARANCE_PER_SIDE,
    MARKER_H,
)

MODEL_DIR = "MODEL_04_SENSOR_CONNECTOR"
CAV_X, CAV_Y = 25.0, 19.0
INSERT_X, INSERT_Y = 24.0, 18.0
# unequal internal guide ribs (projection 1.0): (side, width in Y, centre Y).
# User-approved change: the spec put the right rib at Y -4, which is point-symmetric with the
# left rib, so a 180-degree turned insert still fitted. Both ribs at Y +4 block reversal.
GUIDE_RIBS = ((-1, 3.0, 4.0), (1, 2.0, 4.0))


def _housing_detail(body: cq.Workplane) -> cq.Workplane:
    for side, width, yc in GUIDE_RIBS:
        wall = side * CAV_X / 2
        body = body.union(box_span(*sorted((wall + side * EPS, wall - side * 1.0)),
                                   yc - width / 2, yc + width / 2, CONN_FLOOR_Z - EPS, CONN_HOUSING_Z))
    tri = [(-15.5, 10.0), (-12.5, 10.0), (-14.0, 13.0)]          # raised front-left marker
    return body.union(profile_xz(tri, FRONT - MARKER_H, MARKER_H + EPS))


def housing() -> Part:
    ribs = [left_rib(-6.0), left_rib(6.0), rear_rib(-8.0), rear_rib(8.0)]
    return connector_housing("M4_01_Housing", CAV_X, CAV_Y, 20.0, ribs, _housing_detail)


def _insert_detail(body: cq.Workplane) -> cq.Workplane:
    c = LOCATING_CLEARANCE_PER_SIDE
    for side, width, yc in GUIDE_RIBS:   # open-bottom grooves that ride on the housing ribs
        face = side * INSERT_X / 2
        inner = side * (CAV_X / 2 - 1.0 - c)
        body = body.cut(box_span(*sorted((face + side * 1.0, inner)), yc - width / 2 - c,
                                 yc + width / 2 + c, CONN_FLOOR_Z - 1, INSERT_TOP + 1))
    marker = (cq.Workplane("XY").workplane(offset=INSERT_TOP - EPS)
              .polyline([(-11.0, -2.0), (-8.0, -2.0), (-9.5, 2.0)]).close().extrude(MARKER_H + EPS))
    return body.union(marker)


def sensor_insert() -> Part:
    pockets = [(3.5, 3.0, x, z) for x in (-5.5, 5.5) for z in (4.0, 8.0)]
    return connector_insert("M4_02_Sensor_Insert", INSERT_X, INSERT_Y, pockets, _insert_detail)


def upper_guide() -> Part:
    return guide_frame("M4_03_Upper_Guide", guide_corner_notch(-1, -1))


def _lock_detail(lock: cq.Workplane) -> cq.Workplane:
    for x in (-9.0, 9.0):                 # two parallel ribs, 1 mm high, on the bridge top
        lock = lock.union(box_span(x - 0.5, x + 0.5, -14.0, -9.0, LOCK_BRIDGE_Z1 - EPS, LOCK_BRIDGE_Z1 + 1.0))
    return lock


def top_lock_part() -> Part:
    return top_lock("M4_04_Top_Lock", _lock_detail)


def build() -> tuple[Part, list[Part]]:
    return (make_fixture(4, CONN_HOUSING_X, CONN_HOUSING_Y),
            [housing(), sensor_insert(), upper_guide(), top_lock_part()])
