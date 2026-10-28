"""Reusable CadQuery primitives, grip pads and the Part record shared by every model."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cadquery as cq

from parameters import (
    GRIP_EDGE_RADIUS, GRIP_ROOT_FILLET, GRIP_TOP_CHAMFER, PEG_A_X, PEG_A_Y, PEG_B_X,
    PEG_B_Y, SOCKET_A_D, SOCKET_B_D, SOCKET_DEPTH, SOCKET_MOUTH_CHAMFER,
)

log = logging.getLogger("fanuc_cad")

# Geometry operations that fell back (e.g. a fillet OpenCascade could not build).
# build_all.py prints these and validate_outputs.py copies them into BUILD_SUMMARY.txt.
BUILD_NOTES: list[str] = []

EPS = 0.01  # overlap used when uniting touching solids so OCC never sees coplanar seams


@dataclass(frozen=True)
class GripZone:
    """Where the two fingers touch a grip pad. Used only for the digital clearance check.

    axis: closing direction ("X" or "Y"); center: pad centre in the closing axis;
    half_width: half the pad thickness across the closing axis;
    span: (lo, hi) free contact range along the other horizontal axis;
    other: centre of the pad on the other horizontal axis (informational);
    z_top / height: top of the pad and its usable contact height.
    """

    axis: str
    center: float
    half_width: float
    span: tuple[float, float]
    z_top: float
    height: float


@dataclass(frozen=True)
class Part:
    """One printed part, modelled in its installed (world) position.

    origin: world coordinates of the part's local origin (centre of its lowest seating plane).
    print_rotation: optional (axis, degrees) applied to the STL only, to lay the part flat.
    """

    name: str
    solid: cq.Workplane
    origin: tuple[float, float, float]
    grips: tuple[GripZone, ...] = ()
    print_rotation: tuple[tuple[float, float, float], float] | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def local(self) -> cq.Workplane:
        ox, oy, oz = self.origin
        return self.solid.translate((-ox, -oy, -oz))


# --- primitives ------------------------------------------------------------

def box(sx: float, sy: float, sz: float, x: float = 0, y: float = 0, z0: float = 0) -> cq.Workplane:
    """Axis-aligned box centred on (x, y) with its bottom at z0."""
    return cq.Workplane("XY").box(sx, sy, sz, centered=(True, True, False)).translate((x, y, z0))


def box_span(x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> cq.Workplane:
    return box(x1 - x0, y1 - y0, z1 - z0, (x0 + x1) / 2, (y0 + y1) / 2, z0)


def cyl(d: float, h: float, x: float = 0, y: float = 0, z0: float = 0) -> cq.Workplane:
    return cq.Workplane("XY").circle(d / 2).extrude(h).translate((x, y, z0))


def cone(r_bottom: float, r_top: float, h: float, x: float = 0, y: float = 0, z0: float = 0) -> cq.Workplane:
    return cq.Workplane("XY").add(
        cq.Solid.makeCone(r_bottom, r_top, h, cq.Vector(x, y, z0), cq.Vector(0, 0, 1)))


def rect_taper(sx: float, sy: float, h: float, taper: float, x: float = 0, y: float = 0,
               z0: float = 0, corner_r: float = 0.0) -> cq.Workplane:
    """Rectangular prism extruded upward with a draft. taper>0 shrinks, taper<0 grows (degrees)."""
    wp = cq.Workplane("XY").workplane(offset=z0).center(x, y)
    if corner_r > 0:
        sk = cq.Sketch().rect(sx, sy).vertices().fillet(corner_r)
        return wp.placeSketch(sk).extrude(h, taper=taper)
    return wp.rect(sx, sy).extrude(h, taper=taper)


def rounded_box(sx: float, sy: float, sz: float, r: float, x: float = 0, y: float = 0,
                z0: float = 0) -> cq.Workplane:
    """Box with vertical edges rounded by r."""
    return box(sx, sy, sz, x, y, z0).edges("|Z").fillet(r)


def profile_yz(points: list[tuple[float, float]], length_x: float, x: float = 0) -> cq.Workplane:
    """Closed (Y, Z) polygon extruded symmetrically along X, centred on x."""
    plane = cq.Plane(origin=(x - length_x / 2, 0, 0), xDir=(0, 1, 0), normal=(1, 0, 0))
    return cq.Workplane(plane).polyline(points).close().extrude(length_x)


def profile_xz(points: list[tuple[float, float]], y_front: float, depth: float) -> cq.Workplane:
    """Closed (X, Z) polygon extruded from y_front toward +Y by depth (blind front pockets)."""
    plane = cq.Plane(origin=(0, y_front + depth, 0), xDir=(1, 0, 0), normal=(0, -1, 0))
    return cq.Workplane(plane).polyline(points).close().extrude(depth)


def fillet_edges_in_box(wp: cq.Workplane, lo: tuple, hi: tuple, r: float, label: str) -> cq.Workplane:
    """Fillet every edge whose centre lies inside the box lo..hi.

    A failed fillet is cosmetic (strength margin only), so the build continues with sharp
    edges; the failure is logged and recorded in BUILD_NOTES so it is never silent.
    """
    try:
        return wp.edges(cq.selectors.BoxSelector(lo, hi)).fillet(r)
    except Exception as exc:  # OCC raises StdFail_NotDone, which has no narrower Python base
        msg = f"{label}: {r} mm fillet not applied ({type(exc).__name__})"
        log.warning(msg)
        BUILD_NOTES.append(msg)
        return wp


def edge_fillet_solid(start: tuple, end: tuple, n1: tuple, n2: tuple, r: float) -> cq.Workplane:
    """Concave fillet material along a straight inside corner from start to end.

    n1, n2: unit vectors lying in the two faces, pointing away from the corner edge.
    Built explicitly (prism minus cylinder) because OCC's fillet fails on some pad roots.
    """
    s, e = cq.Vector(*start), cq.Vector(*end)
    a, b = cq.Vector(*n1) * r, cq.Vector(*n2) * r
    wire = cq.Wire.makePolygon([s, s + a, s + a + b, s + b], close=True)
    prism = cq.Solid.extrudeLinear(cq.Face.makeFromWires(wire), e - s)
    rod = cq.Solid.makeCylinder(r, (e - s).Length, s + a + b, e - s)
    return cq.Workplane("XY").add(prism.cut(rod))


def side_pad_root(normal_axis: str, face: float, out_sign: int, half_width: float,
                  z0: float, z1: float, r: float = GRIP_ROOT_FILLET) -> cq.Workplane:
    """Root fillets for a pad sticking out of a vertical wall (both sides and the underside).

    normal_axis: "X" or "Y" (wall normal); face: wall coordinate; out_sign: pad direction.
    """
    def vec(n: float, t: float, z: float) -> tuple:
        return (n, t, z) if normal_axis == "X" else (t, n, z)

    out = vec(out_sign, 0, 0)
    strips = None
    for side in (-1, 1):
        t = side * half_width
        strip = edge_fillet_solid(vec(face, t, z0), vec(face, t, z1), vec(0, side, 0), out, r)
        strips = strip if strips is None else strips.union(strip)
    bottom = edge_fillet_solid(vec(face, -half_width - r, z0), vec(face, half_width + r, z0),
                               (0, 0, -1), out, r)
    return strips.union(bottom)


def pad_support_wedge(normal_axis: str, face: float, out_sign: int, half_width: float,
                      z_pad: float, projection: float) -> cq.Workplane:
    """45 degree solid wedge under a pad that sticks out of a vertical wall.

    Without it the pad underside is a 9 mm cantilever that cannot print unsupported. The wedge
    reaches to 1 mm short of the pad tip (a 1 mm overhang prints fine) and stays inside the pad
    width, so it never enters the gripper finger zones beside the contact faces.
    """
    run = projection - 1.0
    z0 = max(0.0, z_pad - run)
    n_tip = face + out_sign * (z_pad - z0)
    pts = [(face - out_sign * EPS, z0), (face - out_sign * EPS, z_pad + EPS), (n_tip, z_pad + EPS)]
    if normal_axis == "Y":
        return profile_yz(pts, 2 * half_width)
    return profile_xz(pts, -half_width, 2 * half_width)


# --- grip pads --------------------------------------------------------------

def grip_pad(sx: float, sy: float, sz: float, x: float, y: float, z0: float) -> cq.Workplane:
    """Grip pad: flat vertical contact faces, R0.4 vertical edges, 0.5 top chamfer."""
    pad = box(sx, sy, sz, x, y, z0).faces(">Z").chamfer(GRIP_TOP_CHAMFER)
    return pad.edges("|Z").fillet(GRIP_EDGE_RADIUS)


def add_top_pad(body: cq.Workplane, sx: float, sy: float, sz: float, x: float, y: float,
                z0: float, label: str, root_fillet: bool = True) -> cq.Workplane:
    """Unite a grip pad standing on a horizontal face at z0 and fillet its root."""
    body = body.union(grip_pad(sx, sy, sz + EPS, x, y, z0 - EPS))
    if not root_fillet:
        return body
    lo = (x - sx / 2 - 0.1, y - sy / 2 - 0.1, z0 - 0.05)
    hi = (x + sx / 2 + 0.1, y + sy / 2 + 0.1, z0 + 0.05)
    return fillet_edges_in_box(body, lo, hi, GRIP_ROOT_FILLET, f"{label} grip-pad root")


def top_grip_zone(axis: str, sx: float, sy: float, sz: float, x: float, y: float, z0: float,
                  root: float = GRIP_ROOT_FILLET) -> GripZone:
    """GripZone for a pad standing on a top face; the lowest `root` mm is fillet, not contact."""
    if axis == "X":
        return GripZone("X", x, sx / 2, (y - sy / 2, y + sy / 2), z0 + sz, sz - root)
    return GripZone("Y", y, sy / 2, (x - sx / 2, x + sx / 2), z0 + sz, sz - root)


# --- fixture interface -------------------------------------------------------

def cut_fixture_sockets(body: cq.Workplane) -> cq.Workplane:
    """Two different-diameter locating sockets in the housing underside (anti-180 degree)."""
    for d, x, y in ((SOCKET_A_D, PEG_A_X, PEG_A_Y), (SOCKET_B_D, PEG_B_X, PEG_B_Y)):
        r = d / 2
        body = body.cut(cyl(d, SOCKET_DEPTH, x, y, -EPS))
        body = body.cut(cone(r + SOCKET_MOUTH_CHAMFER, r, SOCKET_MOUTH_CHAMFER, x, y, -EPS))
    return body
