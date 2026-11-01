"""Find where look-alike objects differ and whether each camera can see that difference.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from sim.assembly_lab.objects import lookalikes as L
from sim.assembly_lab.objects import meshes as M

HERE = Path(__file__).resolve().parent
CAMERAS = M.ROOT / "robot" / "cell" / "calibration" / "cameras.json"
SURFACE_Z_MM = -5.1            # calibration/geometry.py SURFACE_Z (base frame, m) -> mm
PART_XY_MM = (400.0, 0.0)      # UF0 x, y = base frame x, y (UF0 is 330 mm above the base, same x, y)
YAWS = range(0, 360, 45)
WRIST_FX_848 = 619.6399636921526 * 848 / 1280
STANDOFF_MM = 150.0
TILT_DEG = 30.0
DIFF_MM = 0.2
SPACING_MM = 0.25
MAX_POINTS = 1500
VISIBLE_PX2, CUE_PX = 12.0, 2.0
GROUPS = [["F01", "F02", "F03", "F04", "F05", "F06"], ["M4_03", "M5_03", "M6_03"], ["M4_04", "M5_04", "M6_04"],
          ["M5_02", "M6_02"], ["M1_02", "M2_02"], ["M1_01", "M3_01"], ["M1_01", "M2_01"], ["M4_01", "M5_01", "M6_01"],
          ["M1_04", "M2_04", "M3_04"]]


def sample(tri: np.ndarray, spacing: float = SPACING_MM) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface points, their face normals and the area each point stands for."""
    pts, nrm, wts = [], [], []
    edges = np.linalg.norm(tri[:, [1, 2, 0]] - tri, axis=2).max(axis=1)
    levels = np.maximum(1, np.ceil(edges / spacing)).astype(int)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1) / 2
    normal = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    for n in np.unique(levels):
        sel = np.nonzero(levels == n)[0]
        u, v = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        keep = (u + v) <= n - 1
        u, v = (u[keep] + 1 / 3) / n, (v[keep] + 1 / 3) / n         # one point per sub-triangle, inside it
        w = 1 - u - v
        t = tri[sel]
        p = w[None, :, None] * t[:, None, 0] + u[None, :, None] * t[:, None, 1] + v[None, :, None] * t[:, None, 2]
        pts.append(p.reshape(-1, 3))
        nrm.append(np.repeat(normal[sel], len(u), axis=0))
        wts.append(np.repeat(area[sel] / len(u), len(u)))
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(wts)


def differing(a_tri: np.ndarray, b_tri: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """A's surface points further than DIFF_MM from B's surface (subsampled, weights rescaled)."""
    pa, na, wa = sample(a_tri)
    pb, _, _ = sample(b_tri, 0.1)
    far = cKDTree(pb).query(pa, distance_upper_bound=DIFF_MM)[0] > DIFF_MM
    pa, na, wa = pa[far], na[far], wa[far]
    if len(pa) > MAX_POINTS:
        idx = rng.choice(len(pa), MAX_POINTS, replace=False)
        wa = wa[idx] * (wa.sum() / wa[idx].sum())
        pa, na = pa[idx], na[idx]
    return pa, na, wa


def occluded(points: np.ndarray, cam: np.ndarray, tri: np.ndarray, chunk: int = 96) -> np.ndarray:
    """Moller-Trumbore: does the segment point -> camera hit any triangle of the part?"""
    out = np.zeros(len(points), bool)
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    for s in range(0, len(points), chunk):
        o = points[s:s + chunk]
        d = cam - o
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        pvec = np.cross(d[:, None, :], e2[None])
        det = np.einsum("ptk,tk->pt", pvec, e1)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tvec = o[:, None, :] - v0[None]
        u = np.einsum("ptk,ptk->pt", tvec, pvec) * inv
        qvec = np.cross(tvec, e1[None])
        v = np.einsum("pk,ptk->pt", d, qvec) * inv
        t = np.einsum("ptk,tk->pt", qvec, e2) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4)
        out[s:s + chunk] = hit.any(axis=1)
    return out


def visible_px2(pts, nrm, wts, tri, cam: np.ndarray, fx: float) -> float:
    """Projected visible area of the points, in pixels^2 (pinhole, fx)."""
    to_cam = cam - pts
    dist = np.linalg.norm(to_cam, axis=1)
    facing = np.einsum("ij,ij->i", nrm, to_cam) / dist
    cand = (facing > 0.05) & (pts[:, 2] > SURFACE_Z_MM + 0.05)
    if not cand.any():
        return 0.0
    idx = np.nonzero(cand)[0]
    vis = idx[~occluded(pts[idx] + nrm[idx] * 0.01, cam, tri)]
    return float(np.sum(wts[vis] * facing[vis] / (dist[vis] / fx) ** 2))


def place(local: np.ndarray, yaw: float, z0: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rot.T + np.array([PART_XY_MM[0], PART_XY_MM[1], z0])


def wrist_cams(top_centre: np.ndarray) -> dict[str, np.ndarray]:
    out = {"wrist_top_down": top_centre + [0, 0, STANDOFF_MM]}
    for name, az in (("front", -90), ("right", 0), ("back", 90), ("left", 180)):
        d = np.array([math.sin(math.radians(TILT_DEG)) * math.cos(math.radians(az)),
                      math.sin(math.radians(TILT_DEG)) * math.sin(math.radians(az)), math.cos(math.radians(TILT_DEG))])
        out[f"wrist_oblique30_{name}"] = top_centre + STANDOFF_MM * d
    return out


def fixed_cams() -> dict[str, tuple[np.ndarray, float]]:
    raw = json.loads(CAMERAS.read_text())
    return {k: (np.array(raw[k]["T_base_cam"])[:3, 3] * 1000.0, raw[k]["fx"]) for k in ("table", "side")}


def pair_views(a: str, b: str, catalog: dict, rng: np.random.Generator) -> dict:
    """Visible differing area of A-vs-B on A and on B, per camera; summed over both parts' own differences."""
    frames = M.grasp_frames()
    tris = {k: M.local_triangles(frames[k]["name"], frames[k]["T_local_from_stl"]) for k in (a, b)}
    diffs = {a: differing(tris[a], tris[b], rng), b: differing(tris[b], tris[a], rng)}
    out = {}
    for oid in (a, b):
        z0 = SURFACE_Z_MM - catalog[oid]["bbox_mm"]["min"][2]
        bb = catalog[oid]["bbox_mm"]
        per = {}
        for name, (cam, fx) in fixed_cams().items():
            vals = []
            for yaw in YAWS:
                pts, nrm, w = diffs[oid]
                rot_n = place(nrm, yaw, 0.0) - [*PART_XY_MM, 0.0]
                vals.append(visible_px2(place(pts, yaw, z0), rot_n, w, place(tris[oid].reshape(-1, 3), yaw, z0)
                                        .reshape(-1, 3, 3), cam, fx))
            dist = float(np.linalg.norm(cam - np.array([*PART_XY_MM, z0])))
            per[name] = {"px2_worst_median_best_yaw": [round(min(vals), 1), round(float(np.median(vals)), 1),
                                                       round(max(vals), 1)], "mm_per_px": round(dist / fx, 3)}
        top = np.array([(bb["min"][0] + bb["max"][0]) / 2, (bb["min"][1] + bb["max"][1]) / 2, bb["max"][2]])
        world_tri = place(tris[oid].reshape(-1, 3), 0.0, z0).reshape(-1, 3, 3)
        pts, nrm, w = diffs[oid]
        for name, cam in wrist_cams(place(top[None], 0.0, z0)[0]).items():
            px2 = visible_px2(place(pts, 0.0, z0), nrm, w, world_tri, cam, WRIST_FX_848)
            per[name] = {"px2": round(px2, 1), "mm_per_px": round(STANDOFF_MM / WRIST_FX_848, 3)}
        out[oid] = {"differing_area_mm2": round(float(diffs[oid][2].sum()), 1), "views": per}
    return out


def verdict(views_a: dict, views_b: dict, cue_mm: float) -> dict:
    """Best camera for the pair: the one where both parts' differing geometry projects largest."""
    res = {}
    for cam, va in views_a.items():
        vb = views_b[cam]
        area = (va.get("px2") if "px2" in va else va["px2_worst_median_best_yaw"][0]) + \
               (vb.get("px2") if "px2" in vb else vb["px2_worst_median_best_yaw"][0])
        cue_px = cue_mm / va["mm_per_px"]
        res[cam] = {"px2_both_parts_worst_yaw": round(area, 1), "cue_px": round(cue_px, 1),
                    "visible": bool(area >= VISIBLE_PX2 and cue_px >= CUE_PX)}
    return res


def main() -> None:
    doc = yaml.safe_load((HERE / "objects.yaml").read_text())
    catalog, rng, out, maps = doc["objects"], np.random.default_rng(0), {}, _maps()
    for group in GROUPS:
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                pv = pair_views(a, b, catalog, rng)
                cue = L.pair_report(a, b, catalog, maps)["best_cue_mm"]
                out[f"{a}~{b}"] = {"best_cue_mm": cue, "parts": pv,
                                   "cameras": verdict(pv[a]["views"], pv[b]["views"], cue)}
                print(a, b, cue, {k: v["visible"] for k, v in out[f"{a}~{b}"]["cameras"].items()})
    (HERE / "visibility.yaml").write_text("# GENERATED by sim/assembly_lab/objects/visibility.py - PRELIMINARY "
                                          "(geometry only)\n" + yaml.safe_dump(out, sort_keys=False, width=120))


def _maps() -> dict:
    data = np.load(HERE / "heightmaps.npz")
    ids = {k.split("/")[0] for k in data.files}
    return {i: {"origin": tuple(data[f"{i}/origin"]), "res": float(data[f"{i}/res"]),
                "top": data[f"{i}/top"].astype(float), "bottom": data[f"{i}/bottom"].astype(float)} for i in ids}


if __name__ == "__main__":
    main()
