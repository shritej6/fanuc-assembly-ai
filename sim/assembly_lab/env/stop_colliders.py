"""Build exact box colliders for the sharp-cornered stop features on top of a base.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np

STEP_MM = 0.1           # height-map pitch; grid cell centres sit half a step off the 0.5 mm CAD grid
FLAT_TOL_MM = 0.05      # a sharp feature: every cell within this of its top height (a chamfer makes a sloped rim)
SNAP_MM = 0.1           # box edges snap to an STL vertex coordinate this close
EMBED_MM = 0.5          # boxes start this far below the plate top so no seam is left at the root


def top_height_map(tris: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Highest surface z (mm) over each (y, x) grid point, NaN where the mesh has none; tris (T, 3, 3) mm."""
    top = np.full((len(ys), len(xs)), np.nan)
    for a, b, c in tris:
        lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
        ix = np.where((xs >= lo[0]) & (xs <= hi[0]))[0]
        iy = np.where((ys >= lo[1]) & (ys <= hi[1]))[0]
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if not len(ix) or not len(iy) or abs(d) < 1e-12:
            continue
        gx, gy = np.meshgrid(xs[ix], ys[iy])
        l1 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / d
        l2 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / d
        inside = (l1 >= 0) & (l2 >= 0) & (1 - l1 - l2 >= 0)
        z = l1 * a[2] + l2 * b[2] + (1 - l1 - l2) * c[2]
        sub = top[np.ix_(iy, ix)]
        top[np.ix_(iy, ix)] = np.where(inside & ~(z <= sub), z, sub)
    return top


def _largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int, int]:
    """(area, row0, row1, col0, col1) of the largest all-True axis-aligned rectangle (inclusive indices)."""
    best = (0, 0, 0, 0, 0)
    heights = np.zeros(mask.shape[1], int)
    for r in range(mask.shape[0]):
        heights = np.where(mask[r], heights + 1, 0)
        stack: list[int] = []
        for c in range(len(heights) + 1):
            h = heights[c] if c < len(heights) else 0
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                area = heights[top] * (c - left)
                if area > best[0]:
                    best = (int(area), r - heights[top] + 1, r, left, c - 1)
            stack.append(c)
    return best


def _snap(v: float, coords: np.ndarray) -> float:
    k = np.abs(coords - v).argmin() if len(coords) else None
    return float(coords[k]) if k is not None and abs(coords[k] - v) <= SNAP_MM else round(v, 3)


def sharp_boxes(tris: np.ndarray, plate_z: float = 0.0) -> list[tuple[float, ...]]:
    """[(xmin, xmax, ymin, ymax, zmin, zmax) mm] boxes covering every raised, flat-topped, unchamfered top feature."""
    from scipy import ndimage

    lo, hi = tris.reshape(-1, 3).min(0), tris.reshape(-1, 3).max(0)
    xs = np.arange(np.floor(lo[0]), np.ceil(hi[0]), STEP_MM) + STEP_MM / 2
    ys = np.arange(np.floor(lo[1]), np.ceil(hi[1]), STEP_MM) + STEP_MM / 2
    top = np.nan_to_num(top_height_map(tris, xs, ys), nan=-1e9)
    labels, n = ndimage.label(top > plate_z + FLAT_TOL_MM)
    verts = tris.reshape(-1, 3)
    boxes = []
    for i in range(1, n + 1):
        feature = labels == i
        h = top[feature]
        if np.mean(h >= h.max() - FLAT_TOL_MM) < 0.999:          # chamfered or domed: leave it to the SDF
            continue
        z_top = float(h.max())
        rim = verts[np.abs(verts[:, 2] - z_top) < 1e-3]
        left = feature.copy()
        while left.sum() > 0.001 * feature.sum():
            area, r0, r1, c0, c1 = _largest_rectangle(left)
            if area == 0:
                break
            left[r0:r1 + 1, c0:c1 + 1] = False
            half = STEP_MM / 2
            boxes.append((_snap(xs[c0] - half, rim[:, 0]), _snap(xs[c1] + half, rim[:, 0]),
                          _snap(ys[r0] - half, rim[:, 1]), _snap(ys[r1] + half, rim[:, 1]),
                          plate_z - EMBED_MM, round(z_top, 3)))
    return boxes
