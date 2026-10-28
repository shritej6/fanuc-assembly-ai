"""CAD risk ranking v0 for the 24 product parts and 6 bases (PRELIMINARY: design factors only, no test data).

    ~/cad-workspace/.venv/bin/python cad/risk_ranking.py

Measured from the delivered STLs, in their print pose (Z = build direction):
  wall_p05_mm      5th-percentile local wall thickness (inward ray from the surface)
  thin_area_pct    share of surface area whose wall is thinner than 1.6 mm (4 perimeters of 0.4 mm)
  weakest_layer    smallest horizontal cross-section (mm^2) over the part height: the layer most likely to split
  slenderness      tallest extent / wall_p05: long thin features bend and snap
Insertion clearance (0.5 mm per side) and the 7.0 mm grip pad are the same for every part, so they do not rank.
Each factor is turned into a rank score in [0, 1] (1 = worst of the 30, ties share a rank); the risk score is
the mean of the four.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
STL_DIR = Path.home() / "Desktop" / "FANUC_FINAL"
THIN_MM = 1.6
SAMPLES = 3000
SECTION_STEP_MM = 0.25


def wall_thickness(mesh: trimesh.Trimesh) -> np.ndarray:
    """Inward ray length at area-uniform surface samples (NaN where the ray found no exit)."""
    pts, faces = trimesh.sample.sample_surface(mesh, SAMPLES, seed=7)
    normals = mesh.face_normals[faces]
    origins = pts - normals * 1e-3
    locs, idx, _ = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
    thick = np.full(len(pts), np.nan)
    thick[idx] = np.linalg.norm(locs - origins[idx], axis=1)
    return thick


def weakest_layer(mesh: trimesh.Trimesh) -> float:
    z0, z1 = mesh.bounds[0][2] + 0.3, mesh.bounds[1][2] - 0.3
    areas = []
    for z in np.arange(z0, z1, SECTION_STEP_MM):
        sec = mesh.section(plane_origin=(0, 0, z), plane_normal=(0, 0, 1))
        if sec is not None:
            planar, _ = sec.to_2D()
            areas.append(planar.area)
    return float(min(areas)) if areas else float("nan")


def measure(path: str) -> dict:
    mesh = trimesh.load_mesh(path)
    thick = wall_thickness(mesh)
    ok = ~np.isnan(thick)
    p05 = float(np.percentile(thick[ok], 5))
    return {"object_id": Path(path).stem[:5] if Path(path).stem.startswith("M") else Path(path).stem[:3],
            "name": Path(path).stem, "wall_p05_mm": round(p05, 2),
            "thin_area_pct": round(100 * float(np.mean(thick[ok] < THIN_MM)), 1),   # samples are area-uniform
            "weakest_layer_mm2": round(weakest_layer(mesh), 1),
            "slenderness": round(float(mesh.extents.max()) / p05, 1)}


def rank_scores(rows: list[dict]) -> None:
    factors = {"wall_p05_mm": False, "thin_area_pct": True, "weakest_layer_mm2": False, "slenderness": True}
    n = len(rows)
    for key, higher_is_worse in factors.items():
        values = np.array([row[key] for row in rows], dtype=float)
        ranks = rankdata(values if higher_is_worse else -values, method="average")   # ties share a rank
        for row, r in zip(rows, ranks):
            row[f"rank_{key}"] = round(float((r - 1) / (n - 1)), 3)
    for row in rows:
        row["risk_score"] = round(float(np.mean([row[f"rank_{k}"] for k in factors])), 3)


def main() -> None:
    rows = [measure(p) for p in sorted(glob.glob(str(STL_DIR / "*" / "*.stl")))]
    rank_scores(rows)
    rows.sort(key=lambda r: -r["risk_score"])
    out = {"status": "PRELIMINARY - design factors only; to be updated by cycle tests (U05) and real runs",
           "method": __doc__.strip().splitlines()[2:], "thin_wall_mm": THIN_MM, "parts": rows}
    (ROOT / "cad" / "risk_ranking_v0.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in rows:
        print(f"{r['name']:32s} risk {r['risk_score']:.3f}  wall_p05 {r['wall_p05_mm']:5.2f}  thin {r['thin_area_pct']:5.1f}%"
              f"  layer {r['weakest_layer_mm2']:7.1f}  slender {r['slenderness']:5.1f}")


if __name__ == "__main__":
    main()
