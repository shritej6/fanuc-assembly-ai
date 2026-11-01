"""CAD risk ranking v0 for the 24 product parts and 6 bases (PRELIMINARY: design factors only, no test data).

    uv run python cad/risk_ranking.py --stl-dir ~/Desktop/FANUC_FINAL

Measured from the delivered STLs, in their print pose (Z = build direction):
  wall_p05_mm      5th-percentile local wall thickness (inward ray from the surface)
  thin_area_pct    share of surface area whose wall is thinner than 1.6 mm (4 perimeters of 0.4 mm)
  weakest_layer    smallest cross-section (mm^2) at the mid-height of each printed 0.20 mm layer, over the interior
                   layers: the 5 solid top and 5 solid bottom layers are left out (there every part tapers into the same
                   edge chamfers, so they would rank nothing); the layer most likely to split
  first_layer      cross-section of the first layer on the bed (reported, not ranked)
  slenderness      tallest extent / wall_p05: long thin features bend and snap
Insertion clearance (0.5 mm per side) and the 7.0 mm grip pad are the same for every part, so they do not rank.
Each factor is turned into a rank score in [0, 1] (1 = worst of the 30, ties share a rank); the risk score is
the mean of the four.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
TAXONOMY = ROOT / "contracts" / "taxonomy.json"
OUT = ROOT / "cad" / "risk_ranking_v0.json"
THIN_MM = 1.6
SAMPLES = 3000
LAYER_MM = 0.20   # PRINT_INSTRUCTIONS.txt delivered with the STLs: "Layer height: 0.20 mm"
SKIN_LAYERS = 5   # same file: "Top/bottom: 5 layers"
FACTORS = {"wall_p05_mm": False, "thin_area_pct": True, "weakest_layer_mm2": False, "slenderness": True}


class RiskInputError(ValueError):
    """The STL set or a measurement cannot support a ranking; nothing is written."""


def wall_thickness(mesh: trimesh.Trimesh) -> np.ndarray:
    """Inward ray length at area-uniform surface samples (NaN where the ray found no exit)."""
    pts, faces = trimesh.sample.sample_surface(mesh, SAMPLES, seed=7)
    normals = mesh.face_normals[faces]
    origins = pts - normals * 1e-3
    locs, idx, _ = mesh.ray.intersects_location(origins, -normals, multiple_hits=False)
    thick = np.full(len(pts), np.nan)
    thick[idx] = np.linalg.norm(locs - origins[idx], axis=1)
    return thick


def layer_areas(mesh: trimesh.Trimesh) -> list[float]:
    """Cross-section of every printed layer, bottom up, cut at the layer's mid-height (NaN where nothing is cut)."""
    z_min, z_max = mesh.bounds[0][2], mesh.bounds[1][2]
    layers = math.floor((z_max - z_min) / LAYER_MM + 1e-9)   # 1.0 // 0.2 is 4.0 in floating point, not 5
    areas = []
    for k in range(layers):
        sec = mesh.section(plane_origin=(0, 0, z_min + (k + 0.5) * LAYER_MM), plane_normal=(0, 0, 1))
        areas.append(float(sec.to_2D()[0].area) if sec is not None else float("nan"))
    return areas


def weakest_layer(areas: list[float]) -> float:
    interior = areas[SKIN_LAYERS:-SKIN_LAYERS]
    return float(np.nanmin(interior)) if interior and not np.isnan(interior).all() else float("nan")


def object_id(path: Path) -> str:
    return path.stem[:5] if path.stem.startswith("M") else path.stem[:3]


def measure(path: Path) -> dict:
    mesh = trimesh.load_mesh(path)
    if not mesh.is_watertight:
        raise RiskInputError(f"{path.name} is not watertight: wall and section measurements are undefined")
    thick = wall_thickness(mesh)
    ok = ~np.isnan(thick)
    if not ok.any():
        raise RiskInputError(f"{path.name}: no inward ray found the opposite wall")
    p05 = float(np.percentile(thick[ok], 5))
    areas = layer_areas(mesh)
    row = {"object_id": object_id(path), "name": path.stem, "wall_p05_mm": round(p05, 2),
           "thin_area_pct": round(100 * float(np.mean(thick[ok] < THIN_MM)), 1),   # samples are area-uniform
           "weakest_layer_mm2": round(weakest_layer(areas), 1), "first_layer_mm2": round(areas[0], 1),
           "slenderness": round(float(mesh.extents.max()) / p05, 1)}
    # a share of 0 % thin wall is a measurement; a wall, layer or slenderness of 0 is not
    bad = [key for key in FACTORS if not math.isfinite(row[key]) or row[key] < 0 or (row[key] == 0 and key != "thin_area_pct")]
    if bad:
        raise RiskInputError(f"{path.name}: impossible {', '.join(bad)}")
    return row


def find_stls(stl_dir: Path, expected: list[str]) -> list[Path]:
    """The one STL per taxonomy ID, or RiskInputError naming what is missing, extra or duplicated."""
    by_id: dict[str, list[Path]] = {}
    for path in sorted(stl_dir.glob("*/*.stl")):
        by_id.setdefault(object_id(path), []).append(path)
    missing = sorted(set(expected) - set(by_id))
    extra = sorted(set(by_id) - set(expected))
    duplicated = sorted(oid for oid, paths in by_id.items() if len(paths) > 1)
    if missing or extra or duplicated:
        raise RiskInputError(f"{stl_dir}: missing {missing}, not in taxonomy {extra}, more than one STL {duplicated}")
    return [by_id[oid][0] for oid in expected]


def rank_scores(rows: list[dict]) -> None:
    n = len(rows)
    for key, higher_is_worse in FACTORS.items():
        values = np.array([row[key] for row in rows], dtype=float)
        ranks = rankdata(values if higher_is_worse else -values, method="average")   # ties share a rank
        for row, r in zip(rows, ranks, strict=True):
            row[f"rank_{key}"] = round(float((r - 1) / (n - 1)), 3)
    for row in rows:
        row["risk_score"] = round(float(np.mean([row[f"rank_{k}"] for k in FACTORS])), 3)


def ranking(stl_dir: Path) -> dict:
    expected = [obj["object_id"] for obj in json.loads(TAXONOMY.read_text())["objects"]]
    rows = [measure(path) for path in find_stls(stl_dir, expected)]
    rank_scores(rows)
    rows.sort(key=lambda r: (-r["risk_score"], r["object_id"]))
    return {"status": "PRELIMINARY - design factors only; to be updated by cycle tests (U05) and real runs",
            "method": __doc__.strip().splitlines()[4:], "thin_wall_mm": THIN_MM, "layer_mm": LAYER_MM, "parts": rows}


def write_atomically(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stl-dir", type=Path, required=True, help="delivered STLs, one folder per product")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    out = ranking(args.stl_dir)   # every check passes before anything is written
    write_atomically(args.out, json.dumps(out, indent=1) + "\n")
    for r in out["parts"]:
        print(f"{r['name']:32s} risk {r['risk_score']:.3f}  wall_p05 {r['wall_p05_mm']:5.2f}  thin {r['thin_area_pct']:5.1f}%"
              f"  layer {r['weakest_layer_mm2']:7.1f}  slender {r['slenderness']:5.1f}")


if __name__ == "__main__":
    main()
