"""layout.mesh_mass_properties: volume, centroid and inertia of a closed mesh (used for the MHZL2 body mass).
Authored by Shritej Gunda.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sim" / "assembly_lab" / "env"))
import layout as L


def box_mesh(sx: float, sy: float, sz: float, offset=(0.0, 0.0, 0.0)) -> tuple[np.ndarray, np.ndarray]:
    pts = np.array([[x, y, z] for x in (0, sx) for y in (0, sy) for z in (0, sz)], float) + offset
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = np.array([t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))])
    return pts, tris


def test_box_matches_closed_form():
    pts, tris = box_mesh(1.0, 2.0, 3.0, offset=(5.0, -2.0, 1.0))
    vol, com, inertia = L.mesh_mass_properties(pts, tris)
    assert np.isclose(vol, 6.0)
    assert np.allclose(com, [5.5, -1.0, 2.5])
    assert np.allclose(inertia, np.diag([6.0 * (4 + 9) / 12, 6.0 * (1 + 9) / 12, 6.0 * (1 + 4) / 12]))


def test_inside_out_mesh_gives_negative_volume():
    pts, tris = box_mesh(1.0, 1.0, 1.0)
    vol, _, _ = L.mesh_mass_properties(pts, tris[:, ::-1])
    assert np.isclose(vol, -1.0)
