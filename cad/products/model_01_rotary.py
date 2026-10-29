"""Model 1 - Rotary Knob Module: housing, spindle carrier, rotary knob, rear guard."""

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

MODEL_DIR = "MODEL_01_ROTARY_KNOB"
HOUSING_X = 34.0
HOUSING_Y = 32.0
POST_H = 6.0
KNOB_D = 22.0
KNOB_H = 12.0
KNOB_TOP = CAP_Z0 + KNOB_H        # 30
GUARD_ARM_CX = 10.0


def housing() -> Part:
    body = control_housing(HOUSING_X, 31.0, GUARD_ARM_CX, "M1_01_Housing")
    body = round_pocket(body, 19.0, ENTRY_CHAMFER)
    for side in (-1, 1):  # shallow side ribs, kept above the 3 mm fixture stops
        x_face = side * HOUSING_X / 2
        body = body.union(box_span(min(x_face - side * EPS, x_face + side * 1.0),
                                   max(x_face - side * EPS, x_face + side * 1.0), -1.0, 1.0, 4.0, 12.0))
    return Part("M1_01_Housing", body.clean(), (0.0, 0.0, 0.0), (housing_grip_zone(),))


def spindle_carrier() -> Part:
    body = round_carrier(POST_H, flange_grooves=False)
    return Part("M1_02_Spindle_Carrier", body.clean(), (0.0, 0.0, CTRL_POCKET_FLOOR_Z),
                (carrier_grip_zone(POST_H),))


def rotary_knob() -> Part:
    body = cyl(KNOB_D, KNOB_H, z0=CAP_Z0).faces(">Z").edges().fillet(1.0)
    rib = box_span(KNOB_D / 2 - 0.1, KNOB_D / 2 + 0.6, -0.6, 0.6, CAP_Z0, KNOB_TOP - 1.0)
    for i in range(12):
        body = body.union(rib.rotate((0, 0, 0), (0, 0, 1), i * 30.0))
    body = cut_cap_socket(body, POST_H)
    body = cap_pad(body, KNOB_TOP, "M1_03_Rotary_Knob")
    # raised orientation line pointing to the operator (-Y), clear of the X-facing grip faces
    body = body.union(box_span(-0.5, 0.5, -9.8, -6.6, KNOB_TOP - EPS, KNOB_TOP + 1.0))
    return Part("M1_03_Rotary_Knob", body.clean(), (0.0, 0.0, CAP_Z0), (cap_grip_zone(KNOB_TOP),))


def _guard_detail(guard: cq.Workplane) -> cq.Workplane:
    """Right-side 2 x 2 identification notch in the top corner of the bridge."""
    return guard.cut(box_span(10.5, 13.0, 0, 30, 28.0, 31.0))


def rear_guard_part() -> Part:
    return rear_guard("M1_04_Rear_Guard", GUARD_ARM_CX, 25.0, _guard_detail)


def build() -> tuple[Part, list[Part]]:
    """Return (fixture, product parts in assembly order)."""
    return (make_fixture(1, HOUSING_X, HOUSING_Y),
            [housing(), spindle_carrier(), rotary_knob(), rear_guard_part()])


if __name__ == "__main__":
    fixture, parts = build()
    for p in [fixture, *parts]:
        s = p.solid.val()
        print(f"{p.name:24s} solids={len(p.solid.solids().vals())} vol={s.Volume():8.1f} valid={s.isValid()}")
