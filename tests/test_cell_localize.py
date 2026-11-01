"""table-depth localisation and cell zones on the real calibration, against ray-cast synthetic depth.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np
import pytest

from perception.cell.localize import CellZones, LocalizeError, Zone, above_table_mm, locate, table_points, wrist_bbox
from robot.cell.calibration import geometry

CAMS = geometry.load()
SURFACE_M = geometry.SURFACE_Z


def _render_depth(boxes: list[tuple[float, float, float, float, float]]) -> np.ndarray:
    """Depth (uint16 mm) the table depth camera sees."""
    cam = CAMS["table_depth"]
    w, h = cam["size"]
    v, u = np.mgrid[0:h, 0:w].astype(float)
    rays_c = np.stack([(u - cam["cx"]) / cam["fx"], (v - cam["cy"]) / cam["fy"], np.ones_like(u)], -1)
    rot, origin = cam["T"][:3, :3], cam["T"][:3, 3]
    rays = rays_c @ rot.T
    t = (SURFACE_M - origin[2]) / rays[..., 2]
    for x, y, hx, hy, height in boxes:
        top = SURFACE_M + height / 1000.0
        tt = (top - origin[2]) / rays[..., 2]
        px = (origin[0] + rays[..., 0] * tt) * 1000.0
        py = (origin[1] + rays[..., 1] * tt) * 1000.0
        hit = (np.abs(px - x) <= hx) & (np.abs(py - y) <= hy) & (tt < t)
        t = np.where(hit, tt, t)
    return np.round(t * 1000.0).astype(np.uint16)          # depth along the optical axis: rays_c has z = 1


def _bbox(x: float, y: float, hx: float, hy: float, height: float) -> tuple[float, float, float, float]:
    corners = [(x + sx * hx, y + sy * hy, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, height)]
    uv = geometry.project(CAMS["table"], [(cx / 1000, cy / 1000, SURFACE_M + cz / 1000) for cx, cy, cz in corners])
    return (*uv.min(0), *uv.max(0))


@pytest.mark.parametrize("part", [(350.0, -100.0, 15.0, 12.0, 8.0), (300.0, 150.0, 25.0, 25.0, 40.0),
                                  (440.0, 60.0, 10.0, 20.0, 20.0)])
def test_locates_part_centre_and_top(part):
    pts, uv = table_points(CAMS, _render_depth([part]), above_table_mm())
    found = locate(_bbox(*part), pts, uv)
    assert abs(found.x_mm - part[0]) < 3.0 and abs(found.y_mm - part[1]) < 3.0
    assert abs(found.top_z_mm - (geometry.SURFACE_Z * 1000 + geometry.UF0_Z_MM + part[4])) < 1.5


def test_neighbour_in_the_box_corner_does_not_move_the_centre():
    part, neighbour = (350.0, 0.0, 20.0, 20.0, 30.0), (385.0, 35.0, 12.0, 12.0, 30.0)
    pts, uv = table_points(CAMS, _render_depth([part, neighbour]), above_table_mm())
    found = locate(_bbox(*part), pts, uv)
    assert abs(found.x_mm - part[0]) < 4.0 and abs(found.y_mm - part[1]) < 4.0


def test_depth_hole_raises_instead_of_guessing():
    depth = _render_depth([(350.0, 0.0, 20.0, 20.0, 30.0)])
    depth[:] = 0
    pts, uv = table_points(CAMS, depth, above_table_mm())
    with pytest.raises(LocalizeError, match="depth points on the part"):
        locate(_bbox(350.0, 0.0, 20.0, 20.0, 30.0), pts, uv)


def test_empty_box_raises():
    pts, uv = table_points(CAMS, _render_depth([]), above_table_mm())
    with pytest.raises(LocalizeError, match="empty box"):
        locate((10, 10, 10, 40), pts, uv)


def test_bare_table_gives_no_points():
    pts, _ = table_points(CAMS, _render_depth([]), above_table_mm())
    assert len(pts) == 0


@pytest.mark.parametrize(("xy", "zone", "slot"), [((250, 350), Zone.REJECT, None), ((195, -330), Zone.SPARE, "S1"),
                                                  ((296, -378), Zone.SPARE, "S6"), ((350, 0), Zone.TABLE, None),
                                                  ((100, 0), Zone.OUTSIDE, None), ((470, 270), Zone.OUTSIDE, None)])
def test_zones_from_cell_layout(xy, zone, slot):
    assert CellZones.load().zone(*xy) == (zone, slot)


def test_coarse_accuracy_over_random_parts():
    """The numbers in the localize docstring: 300 seeded random parts over the work area."""
    rng = np.random.default_rng(1)
    errors = []
    for _ in range(300):
        part = (rng.uniform(230, 460), rng.uniform(-250, 250), rng.uniform(8, 25), rng.uniform(8, 25), rng.uniform(8, 45))
        pts, uv = table_points(CAMS, _render_depth([part]), above_table_mm())
        found = locate(_bbox(*part), pts, uv)
        errors.append(np.hypot(found.x_mm - part[0], found.y_mm - part[1]))
    assert np.percentile(errors, 95) < 2.5 and max(errors) < 4.0


def test_part_under_the_wrist_at_the_inspect_stop_projects_into_the_lower_middle():
    """Set O renders put the target low in the wrist image (e.g."""
    from sdg.wrist_ik import _starter_robot, solve_tcp

    ik, robot = _starter_robot()
    joints = np.degrees(solve_tcp(ik, [350.0, 0.0, -168.0, -180.0, 0.0, 0.0], **robot))
    part = np.array([[350.0 + dx, dy, z] for dx in (-10, 10) for dy in (-10, 10) for z in (-335.0, -315.0)])
    x0, y0, x1, y1 = wrist_bbox(CAMS, part, joints)
    w, h = CAMS["wrist"]["size"]
    assert abs((x0 + x1) / 2 - w / 2) < w * 0.2 and h / 2 < (y0 + y1) / 2 < h
    assert 40 < x1 - x0 < w and 40 < y1 - y0 < h


def test_part_far_from_the_wrist_view_is_refused():
    from sdg.wrist_ik import _starter_robot, solve_tcp

    ik, robot = _starter_robot()
    joints = np.degrees(solve_tcp(ik, [350.0, 0.0, -168.0, -180.0, 0.0, 0.0], **robot))
    with pytest.raises(LocalizeError):
        wrist_bbox(CAMS, np.array([[350.0, 400.0, -335.0]]), joints)
