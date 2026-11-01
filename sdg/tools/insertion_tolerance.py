"""Measure from CAD the yaw and position tolerance each part needs to drop into place.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from sdg.scene_sampler import Inputs
from sdg.transforms import about_point, rot_axis, translation, world_aabb

if TYPE_CHECKING:
    from manifold3d import Manifold

# The mesh libraries (trimesh, manifold3d) are imported where the solids are built, so the pure helpers below stay
# importable (and tested) without them.

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sdg" / "data" / "insertion_tolerance.json"
YAW_GRID_DEG, YAW_MAX_DEG, YAW_RES_DEG = 5, 20.0, 0.05
SHIFT_DIRS, SHIFT_MAX_MM, SHIFT_RES_MM = 16, 4.0, 0.02


def drops_in(part: Manifold, tf: np.ndarray, obstacle: Manifold) -> bool:
    from sdg.tools.build_clearance import OVERLAP_MM3, _posed

    return (_posed(part, tf) ^ obstacle).volume() <= OVERLAP_MM3


def largest_ok(ok, hi: float, res: float) -> float:
    """Largest x in [0, hi] with ok(x), assuming ok holds up to some point and fails beyond it."""
    if ok(hi):
        return hi
    lo = 0.0
    while hi - lo > res:
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return round(lo, 3)


def symmetry(fit_yaws: list[int]) -> str:
    """Return the insertion symmetry from the spacing of the fit-window centres."""
    if len(fit_yaws) == 360 // YAW_GRID_DEG:
        return "any"
    start = next(i for i, y in enumerate(fit_yaws) if (y - fit_yaws[i - 1]) % 360 != YAW_GRID_DEG) if \
        len(fit_yaws) > 1 else 0
    ordered, windows = fit_yaws[start:] + fit_yaws[:start], []
    for y in ordered:
        if windows and (y - windows[-1][-1]) % 360 == YAW_GRID_DEG:
            windows[-1].append(y)
        else:
            windows.append([y])
    centres = [(w[0] + ((w[-1] - w[0]) % 360) / 2) % 360 for w in windows]
    steps = {round((b - a) % 360) for a, b in zip(centres, centres[1:] + centres[:1])} if len(centres) > 1 else {360}
    return f"every {steps.pop()} deg" if len(steps) == 1 else "irregular"


def part_tolerance(args: tuple[str, dict]) -> tuple[str, dict]:
    import trimesh
    from manifold3d import Manifold, OpType

    from sdg.tools.build_clearance import _manifold

    part_id, geometry = args
    load = lambda k: trimesh.load(REPO / geometry[k]["stl"], force="mesh")
    geo = geometry[part_id]
    product = part_id[1]
    stack = [load(f"F0{product}")] + [
        load(k).apply_transform(np.array(geometry[k]["installed_in_base_frame"]))
        for k in geometry if k.startswith(f"M{product}_") and geometry[k]["install_step"] < geo["install_step"]]
    obstacle = Manifold.batch_boolean([_manifold(m) for m in stack], OpType.Add)
    installed = np.array(geo["installed_in_base_frame"])
    part = _manifold(load(part_id))
    box = world_aabb(installed, geo["bounds_mm"])
    centre = np.array([*(box[0, :2] + box[1, :2]) / 2, box[0, 2]])
    if not drops_in(part, installed, obstacle):
        raise RuntimeError(f"{part_id}: does not drop in even at its CAD pose")

    def turned(deg: float) -> np.ndarray:
        return about_point(rot_axis((0, 0, 1), deg), centre) @ installed

    fit_yaws = [y for y in range(0, 360, YAW_GRID_DEG) if drops_in(part, turned(y), obstacle)]
    yaw_tol = min(largest_ok(lambda d, s=s: drops_in(part, turned(s * d), obstacle), YAW_MAX_DEG, YAW_RES_DEG)
                  for s in (1, -1))
    shift_tol = min(
        largest_ok(lambda m, a=a: drops_in(part, translation(m * math.cos(a), m * math.sin(a), 0) @ installed,
                                           obstacle), SHIFT_MAX_MM, SHIFT_RES_MM)
        for a in (2 * math.pi * d / SHIFT_DIRS for d in range(SHIFT_DIRS)))
    return part_id, {"fit_yaws_deg": fit_yaws, "symmetry": symmetry(fit_yaws), "yaw_tol_deg": yaw_tol,
                     "shift_tol_mm": shift_tol}


def main() -> None:
    inp = Inputs.load()
    parts = sorted(k for k in inp.geometry if k.startswith("M"))
    with ProcessPoolExecutor() as pool:
        table = dict(pool.map(part_tolerance, [(p, inp.geometry) for p in parts]))
    OUT.write_text(json.dumps({"source": "sdg/tools/insertion_tolerance.py", "units": "deg / mm",
                               "yaw_grid_deg": YAW_GRID_DEG, "parts": table}, indent=1) + "\n")
    for p, t in table.items():
        print(f"{p}: fits {t['symmetry']:>14}  yaw tol +-{t['yaw_tol_deg']:5.2f} deg  shift tol {t['shift_tol_mm']:.2f} mm")


if __name__ == "__main__":
    main()
