"""Model 2, Indicator Light Module: housing, diffuser carrier, indicator cap and rear guard.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import cadquery as cq
from common_geometry import EPS, Part, box_span, cyl
from control_family import (
    CAP_Z0,
    cap_grip_zone,
    cap_pad,
    carrier_grip_zone,
    control_housing,
    cut_cap_socket,
    housing_grip_zone,
    rear_guard,
    round_carrier,
    round_pocket,
)
from fixtures import make_fixture
from parameters import CTRL_POCKET_FLOOR_Z, ENTRY_CHAMFER

MODEL_DIR = "MODEL_02_INDICATOR_LIGHT"
HOUSING_X = 34.0
HOUSING_Y = 32.0
POST_H = 5.0
CAP_D = 22.0
CAP_H = 10.0
CAP_TOP = CAP_Z0 + CAP_H          # 28
GUARD_ARM_CX = 10.0


def housing() -> Part:
    body = control_housing(HOUSING_X, 31.0, GUARD_ARM_CX, "M2_01_Housing")
    body = round_pocket(body, 19.0, ENTRY_CHAMFER)
    front = -HOUSING_Y / 2
    # three front ribs, below the grip pad (Z 8) and clear of the fixture L-stops (|X| > 12.5)
    for x in (-8.0, 0.0, 8.0):
        body = body.union(box_span(x - 1.0, x + 1.0, front - 1.0, front + EPS, 0.0, 7.0))
    # 3 (X) x 2 (Z) x 1 (Y) front-left identification notch at the top edge
    body = body.cut(box_span(-14.0, -11.0, front - 1.0, front + 1.0, 14.0, 17.0))
    return Part("M2_01_Housing", body.clean(), (0.0, 0.0, 0.0), (housing_grip_zone(),))


def diffuser_carrier() -> Part:
    body = round_carrier(POST_H, flange_grooves=True)
    return Part("M2_02_Diffuser_Carrier", body.clean(), (0.0, 0.0, CTRL_POCKET_FLOOR_Z),
                (carrier_grip_zone(POST_H),))


def indicator_cap() -> Part:
    body = cyl(CAP_D, CAP_H, z0=CAP_Z0).faces(">Z").edges().fillet(1.0)
    body = cut_cap_socket(body, POST_H)
    body = cap_pad(body, CAP_TOP, "M2_03_Indicator_Cap")
    for r_in, r_out in ((7.6, 8.1), (8.6, 9.1), (9.6, 10.0)):   # optical rings, 0.5 high
        ring = cyl(2 * r_out, 0.5 + EPS, z0=CAP_TOP - EPS).cut(cyl(2 * r_in, 2.0, z0=CAP_TOP - 1))
        body = body.union(ring)
    body = body.union(cyl(2.0, 0.8 + EPS, -5.5, 0.0, CAP_TOP - EPS))   # identification dot
    return Part("M2_03_Indicator_Cap", body.clean(), (0.0, 0.0, CAP_Z0), (cap_grip_zone(CAP_TOP),))


def _guard_detail(guard: cq.Workplane) -> cq.Workplane:
    """R1.5 upper outer corners, then a 2 x 2 notch in the left end of the bridge."""
    guard = guard.edges("|Y").edges(cq.selectors.BoxSelector((-14, 0, 29.9), (14, 30, 30.1))).fillet(1.5)
    return guard.cut(box_span(-14.0, -11.5, 0, 30, 27.0, 29.0))


def rear_guard_part() -> Part:
    return rear_guard("M2_04_Rear_Guard", GUARD_ARM_CX, 27.0, _guard_detail)


def build() -> tuple[Part, list[Part]]:
    return (make_fixture(2, HOUSING_X, HOUSING_Y),
            [housing(), diffuser_carrier(), indicator_cap(), rear_guard_part()])
