"""Model 5 - Control I/O Connector Module: housing, I/O insert, alignment guide, service lock.

M5_02 shares its outer body and seating with M6_02 on purpose (wrong-part recovery test).
"""

from __future__ import annotations

import cadquery as cq

from common_geometry import EPS, Part, box_span, profile_xz
from connector_family import (
    FRONT, INSERT_TOP, LOCK_BRIDGE_Z1, connector_housing, connector_insert, guide_corner_notch,
    guide_frame, left_rib, rear_rib, top_lock,
)
from fixtures import make_fixture
from parameters import CONN_HOUSING_X, CONN_HOUSING_Y, MARKER_H

MODEL_DIR = "MODEL_05_CONTROL_IO_CONNECTOR"
# shared M5/M6 insert seating envelope
CAV_X, CAV_Y, OPEN_W = 27.0, 21.0, 22.0
INSERT_X, INSERT_Y = 26.0, 20.0


def _housing_detail(body: cq.Workplane) -> cq.Workplane:
    square = [(13.0, 10.0), (16.0, 10.0), (16.0, 13.0), (13.0, 13.0)]   # front-right marker
    return body.union(profile_xz(square, FRONT - MARKER_H, MARKER_H + EPS))


def housing() -> Part:
    ribs = [left_rib(-1.5), left_rib(1.5), rear_rib(8.0), rear_rib(11.0)]   # two close pairs
    return connector_housing("M5_01_Housing", CAV_X, CAV_Y, OPEN_W, ribs, _housing_detail)


def _insert_detail(body: cq.Workplane) -> cq.Workplane:
    for x in (-11.0, -9.0):               # raised double-line identity marker
        body = body.union(box_span(x - 0.5, x + 0.5, -3.0, 3.0, INSERT_TOP - EPS, INSERT_TOP + MARKER_H))
    return body


def io_insert() -> Part:
    pockets = [(3.2, 3.0, x, z) for x in (-7.0, 0.0, 7.0) for z in (4.0, 8.0)]
    return connector_insert("M5_02_IO_Insert", INSERT_X, INSERT_Y, pockets, _insert_detail)


def alignment_guide() -> Part:
    return guide_frame("M5_03_Alignment_Guide", guide_corner_notch(1, 1))


def _lock_detail(lock: cq.Workplane) -> cq.Workplane:
    rib = box_span(-0.5, 0.5, -3.0, 3.0, LOCK_BRIDGE_Z1 - EPS, LOCK_BRIDGE_Z1 + 1.0)
    return lock.union(rib.rotate((0, 0, 0), (0, 0, 1), 40.0).translate((-9.0, -11.5, 0)))


def service_lock() -> Part:
    return top_lock("M5_04_Service_Lock", _lock_detail)


def build() -> tuple[Part, list[Part]]:
    return (make_fixture(5, CONN_HOUSING_X, CONN_HOUSING_Y),
            [housing(), io_insert(), alignment_guide(), service_lock()])
