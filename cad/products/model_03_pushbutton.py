"""Model 3 - Push Button Module: housing, actuator stem, button cap, rear guard."""

from __future__ import annotations

import cadquery as cq

from common_geometry import (
    EPS, Part, box_span, fillet_edges_in_box, grip_pad, profile_xz, rect_taper, rounded_box,
)
from control_family import (
    CAP_Z0, cap_grip_zone, cap_pad, carrier_grip_zone, control_housing, cut_cap_socket,
    housing_grip_zone, rear_guard,
)
from fixtures import make_fixture
from parameters import (
    CARRIER_BOTTOM_CHAMFER, CARRIER_FLANGE_H, CARRIER_FLANGE_Z0, CTRL_HOUSING_Z,
    CTRL_POCKET_FLOOR_Z, ENTRY_CHAMFER, KEY_POST_X, KEY_POST_Y,
    LOCATING_CLEARANCE_PER_SIDE,
)

MODEL_DIR = "MODEL_03_PUSH_BUTTON"
HOUSING_X = 36.0
HOUSING_Y = 32.0
POCKET_X, POCKET_Y = 17.0, 13.0
STEM_X, STEM_Y = 16.0, 12.0
RELIEF = 2.0                       # anti-180 degree relieved corner on the stem (+X, +Y)
POST_H = 6.0
CAP_X, CAP_Y, CAP_H = 24.0, 20.0, 8.0
CAP_TOP = CAP_Z0 + CAP_H           # 26
GUARD_ARM_CX = 11.0


def housing() -> Part:
    body = control_housing(HOUSING_X, 33.0, GUARD_ARM_CX, "M3_01_Housing")
    depth = CTRL_HOUSING_Z - CTRL_POCKET_FLOOR_Z
    body = body.cut(rounded_box(POCKET_X, POCKET_Y, depth + 1, 1.0, z0=CTRL_POCKET_FLOOR_Z))
    body = body.cut(rect_taper(POCKET_X, POCKET_Y, ENTRY_CHAMFER + EPS, -45,
                               z0=CTRL_HOUSING_Z - ENTRY_CHAMFER, corner_r=1.0))
    # key block filling the pocket's rear-right corner; the stem's relieved corner clears it
    c = LOCATING_CLEARANCE_PER_SIDE
    kx0, ky0 = STEM_X / 2 - RELIEF + c, STEM_Y / 2 - RELIEF + c
    body = body.union(box_span(kx0, POCKET_X / 2 + EPS, ky0, POCKET_Y / 2 + EPS,
                               CTRL_POCKET_FLOOR_Z - EPS, CTRL_HOUSING_Z))
    # 12 (X) x 2 (Z) x 0.8 (Y) front recess for visual identity. Offset left of centre because
    # the grip-pad support wedge occupies the front face at |X| < 3.5.
    front = -HOUSING_Y / 2
    body = body.cut(box_span(-16.0, -4.0, front - 1.0, front + 0.8, 3.0, 5.0))
    return Part("M3_01_Housing", body.clean(), (0.0, 0.0, 0.0), (housing_grip_zone(),))


def actuator_stem() -> Part:
    """Guided body 16 x 12 (Z 7-16) with relieved corner, flange 22 x 18, 7 x 10 x 6 post."""
    z0, c = CTRL_POCKET_FLOOR_Z, CARRIER_BOTTOM_CHAMFER
    outline = [(-STEM_X / 2, -STEM_Y / 2), (STEM_X / 2, -STEM_Y / 2),
               (STEM_X / 2, STEM_Y / 2 - RELIEF), (STEM_X / 2 - RELIEF, STEM_Y / 2 - RELIEF),
               (STEM_X / 2 - RELIEF, STEM_Y / 2), (-STEM_X / 2, STEM_Y / 2)]
    body = (cq.Workplane("XY").workplane(offset=z0).polyline(outline).close()
            .extrude(CARRIER_FLANGE_Z0 - z0 + EPS))
    body = body.faces("<Z").chamfer(c)
    body = body.union(rounded_box(22.0, 18.0, CARRIER_FLANGE_H, 1.0, z0=CARRIER_FLANGE_Z0))
    # 45 degree support under the flange, 0.5 mm inside the pocket lead-in and clear of the key block
    # (starts 0.3 mm lower and smaller so its faces cross the body walls instead of meeting edge-on)
    c_in, sink = ENTRY_CHAMFER, 0.3
    support = rect_taper(STEM_X - 2 * sink, STEM_Y - 2 * sink, c_in + sink + EPS, -45,
                         z0=CARRIER_FLANGE_Z0 - c_in - sink)
    kx, ky = STEM_X / 2 - RELIEF, STEM_Y / 2 - RELIEF
    support = support.cut(box_span(kx, STEM_X, ky, STEM_Y, CARRIER_FLANGE_Z0 - c_in - 1, CARRIER_FLANGE_Z0))
    body = body.union(support)
    body = body.union(grip_pad(KEY_POST_X, KEY_POST_Y, POST_H + EPS, 0, 0, CAP_Z0 - EPS))
    return Part("M3_02_Actuator_Stem", body.clean(), (0.0, 0.0, z0), (carrier_grip_zone(POST_H),))


def button_cap() -> Part:
    body = rounded_box(CAP_X, CAP_Y, CAP_H, 2.0, z0=CAP_Z0)
    body = fillet_edges_in_box(body, (-13, -11, CAP_TOP - 0.1), (13, 11, CAP_TOP + 0.1), 0.8,
                               "M3_03_Button_Cap top edge")
    # socket roof ends exactly at the cap top; the grip pad closes it as a 5 x 8 mm bridge
    body = cut_cap_socket(body, POST_H)
    body = cap_pad(body, CAP_TOP, "M3_03_Button_Cap")
    # 12 x 8 x 0.5 top recess, interrupted where the pad stands; 1.6 mm wall kept to the socket
    for side in (-1, 1):
        body = body.cut(box_span(*sorted((side * 4.6, side * 6.0)), -4.0, 4.0,
                                 CAP_TOP - 0.5, CAP_TOP + 1.0))
    return Part("M3_03_Button_Cap", body.clean(), (0.0, 0.0, CAP_Z0), (cap_grip_zone(CAP_TOP),))


def _guard_detail(guard: cq.Workplane) -> cq.Workplane:
    """Centred triangular notch, 3 wide x 2 high, in the underside of the bridge."""
    tri = profile_xz([(-1.5, 26.0 - EPS), (1.5, 26.0 - EPS), (0.0, 28.0)], 10.0, 15.0)
    return guard.cut(tri)


def rear_guard_part() -> Part:
    return rear_guard("M3_04_Rear_Guard", GUARD_ARM_CX, 29.0, _guard_detail)


def build() -> tuple[Part, list[Part]]:
    return (make_fixture(3, HOUSING_X, HOUSING_Y),
            [housing(), actuator_stem(), button_cap(), rear_guard_part()])
