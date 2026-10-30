"""Precompute how far a mis-built part must rise to rest ON the fixture stack instead of passing through it.

    uv run --with trimesh --with manifold3d --with scipy python -m sdg.tools.build_clearance

Writes sdg/data/build_clearance.json, which the scene sampler reads (no mesh library at sampling time).
For every part in its installed pose, against the fixture plus every part installed before it:
  tilt   lift (mm) per tilt azimuth x tilt angle, pivot = bottom centre of the part's box (as the sampler tilts)
  shift  lift per shift direction x shift distance
  yaw    lift per yaw (90/180/270) and how visible the turn is (p99 surface deviation, mm); a turn of a symmetric
         part looks seated, so the sampler only uses yaws whose deviation is at least YAW_VISIBLE_MM
  wrong  lift of each look-alike swapped in at this part's installed pose
Lift = smallest rise with no interpenetration (bisection on intersection volume, 0.01 mm resolution).
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import trimesh
from manifold3d import Manifold, Mesh, OpType
from scipy.spatial import cKDTree

from sdg.scene_sampler import Inputs
from sdg.transforms import about_point, rot_axis, translation, world_aabb

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sdg" / "data" / "build_clearance.json"
TILT_AZ_STEPS, TILT_DEG = 32, [round(0.5 * i, 1) for i in range(1, 25)]   # 0.5 .. 12 deg
SHIFT_DIRS, SHIFT_MM = 16, [round(0.5 * i, 1) for i in range(1, 9)]      # 0.5 .. 4 mm
YAWS = (90, 180, 270)
YAW_VISIBLE_MM = 1.0
OVERLAP_MM3, LIFT_RES_MM, LIFT_MAX_MM = 0.05, 0.01, 40.0


def _manifold(mesh: trimesh.Trimesh) -> Manifold:
    return Manifold(Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                         tri_verts=np.asarray(mesh.faces, np.uint32)))


def _posed(part: Manifold, tf: np.ndarray) -> Manifold:
    return part.transform(tf[:3, :])


def lift(part: Manifold, tf: np.ndarray, obstacle: Manifold) -> float:
    """Smallest upward shift of `part` posed by `tf` that clears `obstacle`."""
    def clear(h):
        return (_posed(part, translation(0, 0, h) @ tf) ^ obstacle).volume() <= OVERLAP_MM3
    if clear(0.0):
        return 0.0
    hi = 1.0
    while not clear(hi):
        hi *= 2
        if hi > LIFT_MAX_MM:
            raise RuntimeError(f"no clearance within {LIFT_MAX_MM} mm")
    lo = hi / 2 if hi > 1.0 else 0.0
    while hi - lo > LIFT_RES_MM:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if clear(mid) else (mid, hi)
    return round(hi, 3)


def _yaw_deviation(mesh: trimesh.Trimesh, turn: np.ndarray) -> float:
    pts = mesh.sample(4000, seed=0)
    turned = trimesh.transform_points(pts, turn)
    return float(np.percentile(cKDTree(mesh.sample(20000, seed=1)).query(turned)[0], 99))


def part_table(args: tuple[str, dict, list[str]]) -> tuple[str, dict]:
    part_id, geometry, mates = args
    load = lambda k: trimesh.load(REPO / geometry[k]["stl"], force="mesh")
    geo = geometry[part_id]
    product = part_id[1]
    stack = [load(f"F0{product}")] + [
        load(k).apply_transform(np.array(geometry[k]["installed_in_base_frame"]))
        for k in geometry if k.startswith(f"M{product}_") and geometry[k]["install_step"] < geo["install_step"]]
    obstacle = Manifold.batch_boolean([_manifold(m) for m in stack], OpType.Add)
    installed = np.array(geo["installed_in_base_frame"])
    mesh = load(part_id)
    part = _manifold(mesh)
    box = world_aabb(installed, geo["bounds_mm"])
    centre = np.array([*(box[0, :2] + box[1, :2]) / 2, box[0, 2]])

    tilt = []
    for a in range(TILT_AZ_STEPS):
        az = math.radians(360 * a / TILT_AZ_STEPS)
        tilt.append([lift(part, about_point(rot_axis((math.cos(az), math.sin(az), 0), t), centre) @ installed,
                          obstacle) for t in TILT_DEG])
    shift = []
    for d in range(SHIFT_DIRS):
        az = math.radians(360 * d / SHIFT_DIRS)
        shift.append([lift(part, translation(s * math.cos(az), s * math.sin(az), 0) @ installed, obstacle)
                      for s in SHIFT_MM])
    yaw = {}
    for y in YAWS:
        turn = about_point(rot_axis((0, 0, 1), y), centre)
        yaw[str(y)] = {"lift_mm": lift(part, turn @ installed, obstacle),
                       "deviation_mm": round(_yaw_deviation(mesh.copy().apply_transform(installed), turn), 3)}
    wrong = {m: lift(_manifold(load(m)), installed, obstacle) for m in mates}
    return part_id, {"tilt_mm": tilt, "shift_mm": shift, "yaw": yaw, "wrong_mm": wrong}


def main() -> None:
    inp = Inputs.load()
    parts = sorted(k for k in inp.geometry if k.startswith("M"))
    with ProcessPoolExecutor() as pool:
        tables = dict(pool.map(part_table, [(p, inp.geometry, inp.family_mates(p)) for p in parts]))
    out = {"source": "sdg/tools/build_clearance.py", "units": "mm / deg",
           "tilt_az_steps": TILT_AZ_STEPS, "tilt_deg": TILT_DEG, "shift_dirs": SHIFT_DIRS, "shift_mm": SHIFT_MM,
           "yaw_visible_mm": YAW_VISIBLE_MM, "parts": tables}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    worst = max(max(max(r) for r in t["tilt_mm"]) for t in tables.values())
    print(f"wrote {OUT} for {len(tables)} parts; largest tilt lift {worst} mm")


if __name__ == "__main__":
    main()
