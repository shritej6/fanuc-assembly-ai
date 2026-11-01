"""Shared geometry for the control family (Models 1 to 3).
Authored by Shritej Gunda.
"""

from __future__ import annotations

import cadquery as cq
from common_geometry import (
    EPS,
    GripZone,
    Part,
    add_top_pad,
    box,
    cone,
    cut_fixture_sockets,
    cyl,
    edge_fillet_solid,
    grip_pad,
    pad_support_wedge,
    profile_yz,
    rect_taper,
    rounded_box,
    side_pad_root,
    top_grip_zone,
)
from parameters import (
    CARRIER_BOTTOM_CHAMFER,
    CARRIER_FLANGE_H,
    CARRIER_FLANGE_Z0,
    CTRL_BOSS_Y0,
    CTRL_BOSS_Y1,
    CTRL_BOSS_Z,
    CTRL_HOUSING_Z,
    CTRL_POCKET_FLOOR_Z,
    EDGE_FILLET,
    ENTRY_CHAMFER,
    GRIP_HEIGHT,
    GRIP_LENGTH,
    GRIP_ROOT_FILLET,
    GRIP_WIDTH,
    GUARD_ARM_X,
    GUARD_ARM_Y,
    GUARD_ARM_Z,
    GUARD_BRIDGE_Z,
    GUARD_CENTER_Y,
    GUARD_FOOT_CHAMFER,
    GUARD_PAD_Y,
    GUARD_SOCKET_BOTTOM_Z,
    GUARD_SOCKET_X,
    GUARD_SOCKET_Y,
    HOUSING_PAD_CENTER_Y,
    HOUSING_PAD_Z0,
    KEY_POST_X,
    KEY_POST_Y,
    SOCKET_ENTRY_CHAMFER,
    SOCKET_HEADROOM,
    SOCKET_ROOF,
)

HOUSING_Y = 32.0
GUARD_Z0 = GUARD_SOCKET_BOTTOM_Z
GUARD_BRIDGE_Z0 = GUARD_Z0 + GUARD_ARM_Z - GUARD_BRIDGE_Z      # world 26
GUARD_TOP_Z = GUARD_Z0 + GUARD_ARM_Z                            # world 30
CAP_Z0 = CARRIER_FLANGE_Z0 + CARRIER_FLANGE_H                   # world 18
GUARD_PRINT_ROTATION = ((1.0, 0.0, 0.0), -90.0)                 # lay guard on its broad Y face


def control_housing(housing_x: float, boss_x: float, guard_arm_cx: float, label: str) -> cq.Workplane:
    """Return the housing body with guard boss, guard slots, front grip pad and fixture sockets."""
    body = rounded_box(housing_x, HOUSING_Y, CTRL_HOUSING_Z, EDGE_FILLET)
    boss_y = CTRL_BOSS_Y1 - CTRL_BOSS_Y0
    body = body.union(rounded_box(boss_x, boss_y, CTRL_BOSS_Z, EDGE_FILLET,
                                  y=(CTRL_BOSS_Y0 + CTRL_BOSS_Y1) / 2))
    for side in (-1, 1):
        body = body.cut(box(GUARD_SOCKET_X, GUARD_SOCKET_Y, CTRL_HOUSING_Z,
                            side * guard_arm_cx, GUARD_CENTER_Y, GUARD_SOCKET_BOTTOM_Z))
    body = cut_fixture_sockets(body)

    front = -HOUSING_Y / 2
    body = body.union(grip_pad(GRIP_WIDTH, GRIP_LENGTH, GRIP_HEIGHT, 0, HOUSING_PAD_CENTER_Y,
                               HOUSING_PAD_Z0))
    body = body.union(side_pad_root("Y", front, -1, GRIP_WIDTH / 2, HOUSING_PAD_Z0, CTRL_HOUSING_Z))
    return body.union(pad_support_wedge("Y", front, -1, GRIP_WIDTH / 2, HOUSING_PAD_Z0,
                                        front - (HOUSING_PAD_CENTER_Y - GRIP_LENGTH / 2)))


def housing_grip_zone() -> GripZone:
    pad_y0 = HOUSING_PAD_CENTER_Y - GRIP_LENGTH / 2
    return GripZone("X", 0.0, GRIP_WIDTH / 2, (pad_y0, -HOUSING_Y / 2 - GRIP_ROOT_FILLET),
                    CTRL_HOUSING_Z, GRIP_HEIGHT)


def round_pocket(body: cq.Workplane, d: float, chamfer: float) -> cq.Workplane:
    """Cylindrical pocket from the pocket floor to the top, with a conical lead-in."""
    depth = CTRL_HOUSING_Z - CTRL_POCKET_FLOOR_Z
    body = body.cut(cyl(d, depth + 1, z0=CTRL_POCKET_FLOOR_Z))
    r = d / 2
    return body.cut(cone(r, r + chamfer + EPS, chamfer + EPS, z0=CTRL_HOUSING_Z - chamfer))


def key_post(height: float) -> cq.Workplane:
    """7 x 10 keyed post: the cap's anti-rotation key and the carrier's grip feature."""
    return grip_pad(KEY_POST_X, KEY_POST_Y, height + EPS, 0, 0, CAP_Z0 - EPS)


def round_carrier(post_h: float, flange_grooves: bool) -> cq.Workplane:
    """Plug d18 (Z 7-16) with a 1 mm front flat, flange d22 (Z 16-18), keyed post above."""
    c = CARRIER_BOTTOM_CHAMFER
    z0, z1, flat_y = CTRL_POCKET_FLOOR_Z, CARRIER_FLANGE_Z0 + EPS, -8.0
    # Plug = chamfered cylinder intersected with a chamfered half-space (the 1 mm front flat).
    # Both chamfer surfaces meet on a true intersection curve, so no sliver faces are created.
    round_part = cone(9.0 - c, 9.0, c, z0=z0).union(cyl(18.0, z1 - z0 - c, z0=z0 + c))
    flat_side = profile_yz([(flat_y + c + 1, z0 - 1), (20, z0 - 1), (20, z1 + 1),
                            (flat_y, z1 + 1), (flat_y, z0 + c)], 40)
    plug = round_part.intersect(flat_side)
    body = plug.union(cyl(22.0, CARRIER_FLANGE_H, z0=CARRIER_FLANGE_Z0)).union(key_post(post_h))
    # 45 degree cone under the flange (prints without support); it stays 0.5 mm inside the
    # housing's entry chamfer. The carrier seats on its plug bottom, not on the flange.
    lead = ENTRY_CHAMFER
    body = body.union(cone(9.0, 9.0 + lead, lead + EPS, z0=CARRIER_FLANGE_Z0 - lead))
    if flange_grooves:
        for r_in, r_out in ((8.0, 8.6), (9.4, 10.0)):
            ring = cyl(2 * r_out, 1.0, z0=CAP_Z0 - 0.4).cut(cyl(2 * r_in, 1.0, z0=CAP_Z0 - 0.4))
            body = body.cut(ring)
    return body


def carrier_grip_zone(post_h: float) -> GripZone:
    return GripZone("X", 0.0, KEY_POST_X / 2, (-KEY_POST_Y / 2, KEY_POST_Y / 2), CAP_Z0 + post_h, post_h)


def cut_cap_socket(body: cq.Workplane, post_h: float) -> cq.Workplane:
    """Bottom socket for the 7 x 10 post: entry chamfer, straight section, 45 degree roof."""
    sx, sy = KEY_POST_X + 1.0, KEY_POST_Y + 1.0          # 0.5 mm clearance per side
    straight = post_h + SOCKET_HEADROOM
    c = SOCKET_ENTRY_CHAMFER
    body = body.cut(box(sx, sy, straight + EPS, z0=CAP_Z0 - EPS))
    body = body.cut(rect_taper(sx, sy, SOCKET_ROOF, 45, z0=CAP_Z0 + straight))
    return body.cut(rect_taper(sx + 2 * c, sy + 2 * c, c, 45, z0=CAP_Z0 - EPS))


def cap_pad(body: cq.Workplane, z0: float, label: str) -> cq.Workplane:
    return add_top_pad(body, GRIP_WIDTH, GRIP_LENGTH, GRIP_HEIGHT, 0, 0, z0, label)


def cap_grip_zone(z0: float) -> GripZone:
    return top_grip_zone("X", GRIP_WIDTH, GRIP_LENGTH, GRIP_HEIGHT, 0, 0, z0)


def rear_guard(name: str, arm_cx: float, bridge_x: float, detail=None) -> Part:
    """Return the U-shaped rear guard with its grip pad and model-specific feature."""
    guard = box(bridge_x, GUARD_ARM_Y, GUARD_BRIDGE_Z, 0, GUARD_CENTER_Y, GUARD_BRIDGE_Z0)
    for side in (-1, 1):
        arm = box(GUARD_ARM_X, GUARD_ARM_Y, GUARD_ARM_Z, side * arm_cx, GUARD_CENTER_Y, GUARD_Z0)
        guard = guard.union(arm.faces("<Z").chamfer(GUARD_FOOT_CHAMFER))
    if detail is not None:
        guard = detail(guard)
    # Pad is flush with the bridge in Y, so only its two X contact sides get a root fillet.
    guard = add_top_pad(guard, GRIP_WIDTH, GUARD_PAD_Y, GRIP_HEIGHT, 0, GUARD_CENTER_Y,
                        GUARD_TOP_Z, f"{name}", root_fillet=False)
    y0, y1 = GUARD_CENTER_Y - GUARD_PAD_Y / 2, GUARD_CENTER_Y + GUARD_PAD_Y / 2
    for side in (-1, 1):
        x = side * GRIP_WIDTH / 2
        guard = guard.union(edge_fillet_solid((x, y0, GUARD_TOP_Z), (x, y1, GUARD_TOP_Z),
                                              (side, 0, 0), (0, 0, 1), GRIP_ROOT_FILLET))
    zone = GripZone("X", 0.0, GRIP_WIDTH / 2, (y0, y1), GUARD_TOP_Z + GRIP_HEIGHT,
                    GRIP_HEIGHT - GRIP_ROOT_FILLET)
    return Part(name, guard.clean(), (0.0, GUARD_CENTER_Y, GUARD_Z0), (zone,),
                print_rotation=GUARD_PRINT_ROTATION)
