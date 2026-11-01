"""Generate the defect meshes the renderer needs for every part and defect type.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[2]
STL_DIR = REPO / "cad" / "stl"
OUT = REPO / "cad" / "defects"
LAYER_MM = 0.2
REMOVED_BAND = {"snapped_tab": (0.02, 0.30), "missing_feature": (0.02, 0.30), "layer_split": (0.005, 0.12)}
DEFECTS = ("snapped_tab", "missing_feature", "layer_split", "warp", "scale_97", "scale_103")


class DefectGeometryError(RuntimeError):
    """A variant failed its geometry checks; nothing is written for that part."""


def _rng(part_id: str, defect: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(f"{part_id}/{defect}".encode()))


def _cutter(size: float, centre) -> trimesh.Trimesh:
    box = trimesh.creation.box((size, size, size))
    box.apply_translation(centre)
    return box


def _halfspace(point, normal, size: float) -> trimesh.Trimesh:
    """A box whose -normal face is the plane through `point`: everything on the `normal` side gets cut."""
    n = np.asarray(normal, dtype=float) / np.linalg.norm(normal)
    box = trimesh.creation.box((size, size, size))
    box.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], n))
    box.apply_translation(np.asarray(point) + n * size / 2)
    return box


def _one_body(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Drop slivers a cut can leave behind (a broken-off piece falls away)."""
    parts = mesh.split(only_watertight=True)
    if len(parts) == 0:
        raise DefectGeometryError("cut left no watertight body")
    return max(parts, key=lambda p: p.volume)


def snapped_tab(mesh: trimesh.Trimesh, rng) -> trimesh.Trimesh:
    lo, hi = mesh.bounds
    axis, sign = int(rng.integers(0, 2)), float(rng.choice([-1.0, 1.0]))
    normal = np.zeros(3)
    normal[axis], normal[2] = sign * np.sin(np.radians(30.0)), np.cos(np.radians(30.0))
    size = 4 * float(np.max(mesh.extents))
    for inset, z_frac in ((0.3, 0.65), (0.25, 0.75), (0.35, 0.55), (0.2, 0.85), (0.4, 0.45)):
        point = (lo + hi) / 2
        point[axis] = np.where(sign > 0, hi, lo)[axis] - sign * inset * mesh.extents[axis]
        point[2] = lo[2] + z_frac * mesh.extents[2]
        cut = _one_body(mesh.difference(_halfspace(point, normal, size)))
        if _in_band(mesh, cut, "snapped_tab"):
            return cut
    raise DefectGeometryError("no snapped-corner cut inside the volume band")


def missing_feature(mesh: trimesh.Trimesh, rng) -> trimesh.Trimesh:
    lo, hi = mesh.bounds
    axis = int(np.argmax(mesh.extents[:2]))           # the long end is where tabs and pins stick out
    sign = float(rng.choice([-1.0, 1.0]))
    size = 4 * float(np.max(mesh.extents))
    for keep in (0.85, 0.8, 0.9, 0.75, 0.7):
        centre = (lo + hi) / 2
        start = lo[axis] + keep * mesh.extents[axis] if sign > 0 else hi[axis] - keep * mesh.extents[axis]
        centre[axis] = start + sign * size / 2
        cut = _one_body(mesh.difference(_cutter(size, centre)))
        if _in_band(mesh, cut, "missing_feature"):
            return cut
    raise DefectGeometryError("no end cut inside the volume band")


def layer_split(mesh: trimesh.Trimesh, rng) -> trimesh.Trimesh:
    lo, hi = mesh.bounds
    gap, depth = 0.5, 0.65
    for z_frac in (0.5, 0.4, 0.6, 0.3, 0.7):
        z = lo[2] + round(z_frac * mesh.extents[2] / LAYER_MM) * LAYER_MM
        slab = trimesh.creation.box((depth * mesh.extents[0] * 2, mesh.extents[1] + 2, gap))
        slab.apply_translation((lo[0], (lo[1] + hi[1]) / 2, z + gap / 2))
        cut = mesh.difference(slab)
        if cut.is_watertight and len(cut.split()) == 1 and _in_band(mesh, cut, "layer_split"):
            return cut
    raise DefectGeometryError("no layer-split slab keeps the part in one piece")


def warp(mesh: trimesh.Trimesh, rng, lift_mm: float = 0.8) -> trimesh.Trimesh:
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=1.5)
    lo, hi = mesh.bounds
    centre = (lo[:2] + hi[:2]) / 2
    r = np.linalg.norm((v[:, :2] - centre) / (np.linalg.norm(hi[:2] - lo[:2]) / 2), axis=1)
    v = v.copy()
    v[:, 2] += lift_mm * r**2
    out = trimesh.Trimesh(v, f, process=True)
    out.apply_translation((0, 0, lo[2] - out.bounds[0][2]))
    return out


def scaled(mesh: trimesh.Trimesh, factor: float) -> trimesh.Trimesh:
    lo, hi = mesh.bounds
    pivot = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
    out = mesh.copy()
    out.apply_transform(trimesh.transformations.scale_matrix(factor, pivot))
    return out


def _in_band(ok: trimesh.Trimesh, cut: trimesh.Trimesh, defect: str) -> bool:
    lo, hi = REMOVED_BAND[defect]
    return cut.is_watertight and lo <= 1 - cut.volume / ok.volume <= hi


def make(mesh: trimesh.Trimesh, part_id: str, defect: str) -> trimesh.Trimesh:
    rng = _rng(part_id, defect)
    out = {"snapped_tab": snapped_tab, "missing_feature": missing_feature, "layer_split": layer_split,
           "warp": warp}.get(defect)
    out = out(mesh, rng) if out else scaled(mesh, 0.97 if defect == "scale_97" else 1.03)
    if not out.is_watertight or len(out.split()) != 1:
        raise DefectGeometryError(f"{part_id}/{defect}: not a single watertight body")
    return out


def main() -> None:
    """Regenerate every variant; fail if any differs from the committed manifest (unless --write-manifest)."""
    write_manifest = "--write-manifest" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "cad/stl (print frame, mm)", "defects": list(DEFECTS), "variants": {}}
    for stl in sorted(STL_DIR.glob("M*.stl")):
        part_id = "_".join(stl.stem.split("_")[:2])
        ok = trimesh.load(stl)
        for defect in DEFECTS:
            out = make(ok, part_id, defect)
            name = f"{part_id}__{defect}"
            out.export(OUT / f"{name}.stl")
            manifest["variants"][name] = {
                "volume_ratio": round(float(out.volume / ok.volume), 4),
                "extents_mm": np.round(out.extents, 2).tolist(),
            }
    path = OUT / "manifest.json"
    if write_manifest:
        path.write_text(json.dumps(manifest, indent=1) + "\n")
    elif json.loads(path.read_text())["variants"] != manifest["variants"]:
        raise DefectGeometryError(f"regenerated variants differ from {path}; check trimesh / manifold3d versions")
    print(f"{len(manifest['variants'])} variants in {OUT.relative_to(REPO)} match the manifest")


if __name__ == "__main__":
    main()
