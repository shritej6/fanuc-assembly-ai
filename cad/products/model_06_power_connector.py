"""Model 6 - Power Connector Module: housing, power insert, power guide, power lock.

Correct:  Fixture -> Housing -> Power Insert -> Power Guide -> Power Lock
Recovery: if M5_02 was inserted, remove lock, guide, wrong insert; insert M6_02; reinstall.
"""

from __future__ import annotations

from dataclasses import replace

import cadquery as cq

from common_geometry import EPS, Part, box_span, profile_xz
from connector_family import (
    FRONT, INSERT_TOP, LOCK_BRIDGE_Z1, connector_housing, connector_insert, guide_corner_notch,
    guide_frame, left_rib, top_lock,
)
from fixtures import make_fixture
from model_05_io_connector import CAV_X, CAV_Y, INSERT_X, INSERT_Y, OPEN_W
from parameters import CONN_HOUSING_X, CONN_HOUSING_Y, CONN_HOUSING_Z, GRIP_TOP_CHAMFER, LOCK_Y, MARKER_H

MODEL_DIR = "MODEL_06_POWER_CONNECTOR"


def _housing_detail(body: cq.Workplane) -> cq.Workplane:
    z = CONN_HOUSING_Z
    v = [(-16.5, z + 1.0), (-12.5, z + 1.0), (-14.5, z - 2.0)]   # 4 wide x 2 high V notch
    return body.cut(profile_xz(v, FRONT - 1.0, 2.0))


def housing() -> Part:
    ribs = [left_rib(y) for y in (-9.0, -3.0, 3.0, 9.0)]          # four evenly spaced
    return connector_housing("M6_01_Housing", CAV_X, CAV_Y, OPEN_W, ribs, _housing_detail)


def _insert_detail(body: cq.Workplane) -> cq.Workplane:
    return body.union(box_span(-4.0, 4.0, -9.25, -7.75, INSERT_TOP - EPS, INSERT_TOP + MARKER_H))


def power_insert() -> Part:
    pockets = [(9.0, 8.0, x, 6.0) for x in (-6.0, 6.0)]
    return connector_insert("M6_02_Power_Insert", INSERT_X, INSERT_Y, pockets, _insert_detail)


def power_guide() -> Part:
    return guide_frame("M6_03_Power_Guide", guide_corner_notch(1, -1))


def power_lock() -> Part:
    part = top_lock("M6_04_Power_Lock")
    top = LOCK_BRIDGE_Z1 + 8.0
    chevron = [(-2.0, 2.5), (0.0, -1.5), (2.0, 2.5), (1.1, 2.5), (0.0, 0.3), (-1.1, 2.5)]
    cut = (cq.Workplane("XY").workplane(offset=top - 1.0).center(0, LOCK_Y)
           .polyline(chevron).close().extrude(1.0 + GRIP_TOP_CHAMFER))
    return replace(part, solid=part.solid.cut(cut).clean())


def build() -> tuple[Part, list[Part]]:
    return (make_fixture(6, CONN_HOUSING_X, CONN_HOUSING_Y),
            [housing(), power_insert(), power_guide(), power_lock()])
