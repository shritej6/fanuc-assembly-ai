"""Shared geometry for the connector family (Models 4 to 6).
Authored by Shritej Gunda.
"""

from __future__ import annotations

from collections.abc import Callable

import cadquery as cq
from common_geometry import (
    EPS,
    GripZone,
    Part,
    add_top_pad,
    box,
    box_span,
    cut_fixture_sockets,
    fillet_edges_in_box,
    grip_pad,
    pad_support_wedge,
    profile_xz,
    profile_yz,
    rect_taper,
    rounded_box,
    side_pad_root,
    top_grip_zone,
)
from parameters import (
    CONN_FLOOR_Z,
    CONN_HOUSING_X,
    CONN_HOUSING_Y,
    CONN_HOUSING_Z,
    CONN_PAD_CENTER_X,
    CONN_PAD_Z0,
    CONN_RIB_PROJECTION,
    CONN_RIB_WIDTH,
    EDGE_FILLET,
    GRIP_EDGE_RADIUS,
    GRIP_HEIGHT,
    GRIP_LENGTH,
    GRIP_ROOT_FILLET,
    GRIP_TOP_CHAMFER,
    GRIP_WIDTH,
    GUIDE_HOLE,
    GUIDE_HOLE_X,
    GUIDE_HOLE_Y,
    GUIDE_OPEN_X,
    GUIDE_OPEN_Y1,
    GUIDE_PAD_CENTER_Y,
    GUIDE_PEG,
    GUIDE_PEG_H,
    GUIDE_T,
    GUIDE_X,
    GUIDE_Y,
    GUIDE_Z0,
    INSERT_BOTTOM_CHAMFER,
    INSERT_H,
    INSERT_Z0,
    LOCK_BRIDGE_CHAMFER,
    LOCK_BRIDGE_X,
    LOCK_BRIDGE_Y,
    LOCK_BRIDGE_Z,
    LOCK_GUSSET_RUN,
    LOCK_HOLE,
    LOCK_LEG,
    LOCK_PAD_TAPER_H,
    LOCK_X,
    LOCK_Y,
    POCKET_DEPTH,
    POCKET_ROOF,
    SMALL_LEAD_IN,
)

Detail = Callable[[cq.Workplane], cq.Workplane]
GUIDE_Z1 = GUIDE_Z0 + GUIDE_T                      # 23
LOCK_BRIDGE_Z1 = GUIDE_Z1 + LOCK_BRIDGE_Z          # 27
INSERT_TOP = INSERT_Z0 + INSERT_H                  # 17
FRONT = -CONN_HOUSING_Y / 2                        # -15
RIGHT = CONN_HOUSING_X / 2                         # +19


# --- housing ----------------------------------------------------------------

def left_rib(y: float) -> cq.Workplane:
    """Vertical rib on the left face (Z 4-17: above the 3 mm fixture stops)."""
    w = CONN_RIB_WIDTH / 2
    return box_span(-RIGHT - CONN_RIB_PROJECTION, -RIGHT + EPS, y - w, y + w, 4.0, 17.0)


def rear_rib(x: float) -> cq.Workplane:
    """Vertical rib on the rear face. The right face is kept clear for the housing gripper."""
    w = CONN_RIB_WIDTH / 2
    back = CONN_HOUSING_Y / 2
    return box_span(x - w, x + w, back - EPS, back + CONN_RIB_PROJECTION, 4.0, 17.0)


def connector_housing(name: str, cav_x: float, cav_y: float, open_w: float,
                      ribs: list[cq.Workplane], detail: Detail | None = None) -> Part:
    body = rounded_box(CONN_HOUSING_X, CONN_HOUSING_Y, CONN_HOUSING_Z, EDGE_FILLET)
    body = body.cut(box(cav_x, cav_y, CONN_HOUSING_Z, z0=CONN_FLOOR_Z))
    body = body.cut(box_span(-open_w / 2, open_w / 2, FRONT - 1, -cav_y / 2 + EPS,
                             CONN_FLOOR_Z, CONN_HOUSING_Z + 1))
    body = cut_fixture_sockets(body)
    for side in (-1, 1):
        peg = box(GUIDE_PEG, GUIDE_PEG, GUIDE_PEG_H + EPS, side * GUIDE_HOLE_X, GUIDE_HOLE_Y,
                  CONN_HOUSING_Z - EPS).faces(">Z").chamfer(SMALL_LEAD_IN)
        body = body.union(peg)
    for rib in ribs:
        body = body.union(rib)
    body = body.union(grip_pad(GRIP_LENGTH, GRIP_WIDTH, GRIP_HEIGHT, CONN_PAD_CENTER_X, 0, CONN_PAD_Z0))
    body = body.union(side_pad_root("X", RIGHT, 1, GRIP_WIDTH / 2, CONN_PAD_Z0, CONN_HOUSING_Z))
    body = body.union(pad_support_wedge("X", RIGHT, 1, GRIP_WIDTH / 2, CONN_PAD_Z0,
                                        CONN_PAD_CENTER_X + GRIP_LENGTH / 2 - RIGHT))
    if detail is not None:
        body = detail(body)
    zone = GripZone("Y", 0.0, GRIP_WIDTH / 2,
                    (RIGHT + GRIP_ROOT_FILLET, CONN_PAD_CENTER_X + GRIP_LENGTH / 2),
                    CONN_HOUSING_Z, GRIP_HEIGHT)
    return Part(name, body.clean(), (0.0, 0.0, 0.0), (zone,))


# --- insert -----------------------------------------------------------------

def front_pocket(width: float, height: float, x: float, z_center_local: float, y_front: float) -> cq.Workplane:
    """Blind front cavity, 4 mm deep, with 45 degree roof chamfers so it prints without support."""
    z0 = INSERT_Z0 + z_center_local - height / 2
    z1, r, w = z0 + height, POCKET_ROOF, width / 2
    pts = [(x - w, z0), (x + w, z0), (x + w, z1 - r), (x + w - r, z1), (x - w + r, z1), (x - w, z1 - r)]
    return profile_xz(pts, y_front - EPS, POCKET_DEPTH + EPS)


def connector_insert(name: str, bx: float, by: float, pockets: list[tuple[float, float, float, float]],
                     detail: Detail | None = None) -> Part:
    """Return the connector insert body with lead-in, front pockets and grip pad."""
    body = box(bx, by, INSERT_H, z0=INSERT_Z0).faces("<Z").chamfer(INSERT_BOTTOM_CHAMFER)
    for w, h, x, zc in pockets:
        body = body.cut(front_pocket(w, h, x, zc, -by / 2))
    if detail is not None:
        body = detail(body)
    body = add_top_pad(body, GRIP_WIDTH, GRIP_LENGTH, GRIP_HEIGHT, 0, 0, INSERT_TOP, name)
    zone = top_grip_zone("X", GRIP_WIDTH, GRIP_LENGTH, GRIP_HEIGHT, 0, 0, INSERT_TOP)
    return Part(name, body.clean(), (0.0, 0.0, INSERT_Z0), (zone,))


# --- U guide frame ------------------------------------------------------------

def _guide_pad() -> cq.Workplane:
    """Return the rear grip pad on a solid block that replaces an unsupported overhang."""
    y0, y1 = GUIDE_PAD_CENTER_Y - GRIP_LENGTH / 2, GUIDE_PAD_CENTER_Y + GRIP_LENGTH / 2
    rear = GUIDE_Y / 2
    pts = [(rear - EPS, GUIDE_Z0), (y1, GUIDE_Z0), (y1, GUIDE_Z1 + GRIP_HEIGHT),
           (y0, GUIDE_Z1 + GRIP_HEIGHT), (y0, GUIDE_Z1 - EPS), (rear - EPS, GUIDE_Z1 - EPS)]
    pad = profile_yz(pts, GRIP_WIDTH).faces(">Z").chamfer(GRIP_TOP_CHAMFER)
    contact_edges = (cq.selectors.BoxSelector((-5, y0 - 0.1, 17), (5, y0 + 0.1, 32))
                     + cq.selectors.BoxSelector((-5, y1 - 0.1, 17), (5, y1 + 0.1, 32)))
    return pad.edges("|Z").edges(contact_edges).fillet(GRIP_EDGE_RADIUS)


def guide_frame(name: str, detail: Detail | None = None) -> Part:
    frame = rounded_box(GUIDE_X, GUIDE_Y, GUIDE_T, EDGE_FILLET, z0=GUIDE_Z0)
    frame = frame.cut(box_span(-GUIDE_OPEN_X / 2, GUIDE_OPEN_X / 2, -GUIDE_Y / 2 - 1, GUIDE_OPEN_Y1,
                               GUIDE_Z0 - 1, GUIDE_Z1 + 1))
    for side in (-1, 1):
        gx = side * GUIDE_HOLE_X
        frame = frame.cut(box(GUIDE_HOLE, GUIDE_HOLE, GUIDE_T + 2, gx, GUIDE_HOLE_Y, GUIDE_Z0 - 1))
        frame = frame.cut(rect_taper(GUIDE_HOLE + 2 * SMALL_LEAD_IN, GUIDE_HOLE + 2 * SMALL_LEAD_IN,
                                     SMALL_LEAD_IN, 45, gx, GUIDE_HOLE_Y, GUIDE_Z0 - EPS))
        # lock hole: inner edge meets the U opening; extended 0.5 mm into it to avoid a seam
        lx0, lx1 = side * (LOCK_X - LOCK_HOLE / 2 - 0.5), side * (LOCK_X + LOCK_HOLE / 2)
        frame = frame.cut(box_span(min(lx0, lx1), max(lx0, lx1), LOCK_Y - LOCK_HOLE / 2,
                                   LOCK_Y + LOCK_HOLE / 2, GUIDE_Z0 - 1, GUIDE_Z1 + 1))
    if detail is not None:
        frame = detail(frame)
    frame = frame.union(_guide_pad())
    lo = (-GRIP_WIDTH / 2 - 0.1, GUIDE_PAD_CENTER_Y - GRIP_LENGTH / 2 - 0.1, GUIDE_Z0 + 0.1)
    hi = (GRIP_WIDTH / 2 + 0.1, GUIDE_Y / 2 + 0.05, GUIDE_Z1 + 0.05)
    frame = fillet_edges_in_box(frame, lo, hi, GRIP_ROOT_FILLET, f"{name} grip-pad root")
    zone = GripZone("X", 0.0, GRIP_WIDTH / 2,
                    (GUIDE_PAD_CENTER_Y - GRIP_LENGTH / 2, GUIDE_PAD_CENTER_Y + GRIP_LENGTH / 2),
                    GUIDE_Z1 + GRIP_HEIGHT, GRIP_HEIGHT - GRIP_ROOT_FILLET)
    return Part(name, frame.clean(), (0.0, 0.0, GUIDE_Z0), (zone,))


def guide_corner_notch(corner_x: int, corner_y: int) -> Detail:
    """3 (X) x 2 (Y) identification notch at one outer corner of the guide frame."""
    def cut(frame: cq.Workplane) -> cq.Workplane:
        x_out, y_out = corner_x * (GUIDE_X / 2 + 1), corner_y * (GUIDE_Y / 2 + 1)
        x_in, y_in = corner_x * (GUIDE_X / 2 - 3), corner_y * (GUIDE_Y / 2 - 2)
        return frame.cut(box_span(min(x_out, x_in), max(x_out, x_in), min(y_out, y_in),
                                  max(y_out, y_in), GUIDE_Z0 - 1, GUIDE_Z1 + 1))
    return cut


# --- top service lock -----------------------------------------------------------

def top_lock(name: str, detail: Detail | None = None) -> Part:
    leg_h = GUIDE_T
    c, hy = LOCK_BRIDGE_CHAMFER, LOCK_BRIDGE_Y / 2
    # bridge underside is chamfered 1.5 mm on both long edges, leaving exactly the 3 mm leg width
    bridge = [(LOCK_Y - hy + c, GUIDE_Z1), (LOCK_Y + hy - c, GUIDE_Z1), (LOCK_Y + hy, GUIDE_Z1 + c),
              (LOCK_Y + hy, LOCK_BRIDGE_Z1), (LOCK_Y - hy, LOCK_BRIDGE_Z1), (LOCK_Y - hy, GUIDE_Z1 + c)]
    lock = profile_yz(bridge, LOCK_BRIDGE_X)
    for side in (-1, 1):
        leg = box(LOCK_LEG, LOCK_LEG, leg_h + EPS, side * LOCK_X, LOCK_Y, GUIDE_Z0)
        lock = lock.union(leg.faces("<Z").chamfer(SMALL_LEAD_IN))
        # 45 degree print gusset from the leg's inner face up to the bridge underside
        xi = side * (LOCK_X - LOCK_LEG / 2)
        xg = xi - side * LOCK_GUSSET_RUN
        tri = [(xi + side * EPS, GUIDE_Z0), (xi + side * EPS, GUIDE_Z1 + EPS), (xg, GUIDE_Z1 + EPS)]
        lock = lock.union(profile_xz(tri, LOCK_Y - LOCK_LEG / 2, LOCK_LEG))
    if detail is not None:
        lock = detail(lock)
    # grip pad: lower 2 mm tapers 45 degrees from the 6 mm bridge to the full 10 mm contact length
    t, top = LOCK_PAD_TAPER_H, LOCK_BRIDGE_Z1 + GRIP_HEIGHT
    pad_pts = [(LOCK_Y - hy, LOCK_BRIDGE_Z1 - EPS), (LOCK_Y + hy, LOCK_BRIDGE_Z1 - EPS),
               (LOCK_Y + GRIP_LENGTH / 2, LOCK_BRIDGE_Z1 + t), (LOCK_Y + GRIP_LENGTH / 2, top),
               (LOCK_Y - GRIP_LENGTH / 2, top), (LOCK_Y - GRIP_LENGTH / 2, LOCK_BRIDGE_Z1 + t)]
    pad = profile_yz(pad_pts, GRIP_WIDTH).faces(">Z").chamfer(GRIP_TOP_CHAMFER)
    lock = lock.union(pad.edges("|Z").fillet(GRIP_EDGE_RADIUS))
    zone = GripZone("X", 0.0, GRIP_WIDTH / 2, (LOCK_Y - GRIP_LENGTH / 2, LOCK_Y + GRIP_LENGTH / 2),
                    top, GRIP_HEIGHT - t)
    return Part(name, lock.clean(), (0.0, LOCK_Y, GUIDE_Z0), (zone,))
