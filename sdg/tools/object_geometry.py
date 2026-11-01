"""Measure from CAD how every object rests on the table and where it sits when installed.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
STL_DIR = REPO / "cad" / "stl"
OUT = REPO / "sdg" / "data" / "object_geometry.json"
PRODUCT_MODULES = [f"model_0{i}_{n}" for i, n in enumerate(
    ["rotary", "indicator", "pushbutton", "sensor_connector", "io_connector", "power_connector"], start=1)]
MIN_POSE_PROB = 0.02
MAX_POSES = 6
MAX_RESIDUAL_MM = 0.05


def axis_rotations() -> list[np.ndarray]:
    """The 24 proper rotations that map coordinate axes onto coordinate axes."""
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            m = np.zeros((3, 3))
            for row, (col, s) in enumerate(zip(perm, signs)):
                m[row, col] = s
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats


def assembled_mesh(part) -> trimesh.Trimesh:
    shape = part.solid.val() if hasattr(part.solid, "val") else part.solid
    verts, tris = shape.tessellate(0.02)
    return trimesh.Trimesh([(v.x, v.y, v.z) for v in verts], tris)


def register(stl: trimesh.Trimesh, target: trimesh.Trimesh) -> tuple[np.ndarray, float]:
    """Return the best rigid transform taking the STL onto the assembled solid, and its residual."""
    tree = cKDTree(target.vertices)
    lo, hi = target.bounds
    best = (None, np.inf)
    for rot in axis_rotations():
        pts = stl.vertices @ rot.T
        shift = lo - pts.min(axis=0)
        if not np.allclose(pts.max(axis=0) + shift, hi, atol=0.2):
            continue
        score = float(np.mean(tree.query(pts + shift)[0]))
        if score < best[1]:
            tf = np.eye(4)
            tf[:3, :3], tf[:3, 3] = rot, shift
            best = (tf, score)
    if best[0] is None:
        raise RuntimeError("no axis rotation matches the assembled bounding box")
    moved = trimesh.transform_points(stl.vertices, best[0])
    sample = moved[np.random.default_rng(0).choice(len(moved), min(len(moved), 3000), replace=False)]
    return best[0], float(np.mean(target.nearest.on_surface(sample)[1]))


def footprint(mesh: trimesh.Trimesh, tf: np.ndarray) -> dict:
    """Table footprint of the mesh posed by `tf`: smallest circle about the xy box centre, plus height."""
    pts = trimesh.transform_points(mesh.vertices, tf)
    if abs(pts[:, 2].min()) > 1e-3:
        raise RuntimeError(f"pose does not rest on z=0 (min z {pts[:, 2].min():.4f})")
    centre = (pts[:, :2].min(axis=0) + pts[:, :2].max(axis=0)) / 2
    return {"footprint_xy": np.round(centre, 3).tolist(),
            "footprint_r": round(float(np.linalg.norm(pts[:, :2] - centre, axis=1).max()), 3),
            "height_mm": round(float(pts[:, 2].max()), 3)}


def installed_rest(mesh: trimesh.Trimesh, installed: np.ndarray) -> dict:
    """The installed rotation lowered onto z=0, with its footprint, and whether it stands on its own."""
    tf = np.eye(4)
    tf[:3, :3] = np.asarray(installed)[:3, :3]
    tf[2, 3] = -trimesh.transform_points(mesh.vertices, tf)[:, 2].min()
    return {"transform": np.round(tf, 6).tolist(), **footprint(mesh, tf), "stands": stands(mesh, tf)}


def stands(mesh: trimesh.Trimesh, tf: np.ndarray, contact_mm: float = 0.3, margin_mm: float = 0.5) -> bool:
    """Centre of mass inside the contact polygon (shrunk by margin_mm): the part does not tip over on its own."""
    from shapely.geometry import MultiPoint, Point

    posed = mesh.copy()
    posed.apply_transform(tf)
    contact = posed.vertices[posed.vertices[:, 2] < contact_mm][:, :2]
    if len(contact) < 3:
        return False
    return bool(MultiPoint(contact).convex_hull.buffer(-margin_mm).contains(Point(posed.center_mass[:2])))


def add_installed_rest(path: Path = OUT) -> None:
    """Add `installed_rest` to every object of an existing geometry file; every other value stays byte-identical."""
    data = json.loads(path.read_text())
    for object_id, obj in data["objects"].items():
        obj["installed_rest"] = installed_rest(trimesh.load(REPO / obj["stl"]), np.array(obj["installed_in_base_frame"]))
        print(f"{object_id:6s} installed rest: height {obj['installed_rest']['height_mm']:6.2f} mm  "
              f"stands {obj['installed_rest']['stands']}")
    path.write_text(json.dumps(data, indent=1))


def stable_poses(mesh: trimesh.Trimesh) -> list[dict]:
    transforms, probs = trimesh.poses.compute_stable_poses(mesh, n_samples=4, threshold=0.0)
    keep = [(t, p) for t, p in zip(transforms, probs) if p >= MIN_POSE_PROB][:MAX_POSES]
    total = sum(p for _, p in keep)
    return [{"transform": np.round(t, 6).tolist(), "probability": round(float(p / total), 4), **footprint(mesh, t)}
            for t, p in keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--cad-dir", type=Path, help="folder with the product CAD scripts (rebuilds the whole file)")
    mode.add_argument("--add-installed-rest", action="store_true", help="add installed_rest to the existing file")
    args = ap.parse_args()
    if args.add_installed_rest:
        add_installed_rest()
        return
    sys.path.insert(0, str(args.cad_dir))
    objects = {}
    for name in PRODUCT_MODULES:
        fixture, parts = importlib.import_module(name).build()
        for step, part in enumerate([fixture] + parts):
            object_id = "_".join(part.name.split("_")[:2]) if step else part.name.split("_")[0]
            mesh = trimesh.load(STL_DIR / f"{part.name}.stl")
            installed, residual = register(mesh, assembled_mesh(part))
            if residual > MAX_RESIDUAL_MM:
                raise RuntimeError(f"{part.name}: registration residual {residual:.3f} mm > {MAX_RESIDUAL_MM}")
            objects[object_id] = {
                "stl": f"cad/stl/{part.name}.stl",
                "install_step": step,
                "bounds_mm": np.round(mesh.bounds, 3).tolist(),
                "installed_in_base_frame": np.round(installed, 6).tolist(),
                "registration_residual_mm": round(residual, 4),
                "upright": footprint(mesh, trimesh.transformations.translation_matrix([0, 0, -mesh.bounds[0][2]])),
                "stable_poses": stable_poses(mesh),
                "installed_rest": installed_rest(mesh, installed),
            }
            print(f"{object_id:6s} step {step}  residual {residual:.4f} mm  poses {len(objects[object_id]['stable_poses'])}")
    OUT.write_text(json.dumps({"units": "mm", "source": "cad/stl + CAD build()", "objects": objects}, indent=1))
    print(f"wrote {OUT.relative_to(REPO)} ({len(objects)} objects)")


if __name__ == "__main__":
    main()
