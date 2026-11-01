"""Build objects.yaml and heightmaps.npz with everything the planner, skills and eval need about the printed objects.
Authored by Shritej Gunda.

    PYTHONPATH=. uv run python sim/assembly_lab/objects/build_objects.py      # same CAD -> same files

Sources (all the printed CAD, nothing measured on the cell):
- cad/stl/*.stl (printed meshes) moved into each installed local frame with cad/grasp_frames.json T_local_from_stl:
  bbox, volume -> mass (PLA 1.24 g/cm3, cad/products/parameters.py), centre of mass, top/bottom height maps.
- cad/grasp_frames.json: grip pad, closing axis, TCP, roll offset, installed_origin_mm (Team V fingers, 24/24 PASS).
- cad/products (parameters.py + model constants, the inputs of build()): pegs, holes, mates, clearances, chamfers;
  each feature is checked against the printed mesh (features.check_features).
- build() order = assembly-sheet order (cad/assembly_sheets.py): base 0, then 01..04.
- contracts/taxonomy.json: family, look-alike groups. sdg/data/insertion_tolerance.json: simulated yaw sweep.
- physics/limits.yaml: max_force.
Precision (INTERFACES.md "Precision gates"): per part, the yaw a seated part can be off before a locating feature
binds = clearance / R_eff, with R_eff the largest radius of a locating feature corner (or peg centre) from the part
origin; the xy capture = clearance + the chamfers on both sides of the tightest mate.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import yaml

from sim.assembly_lab.objects import features as F
from sim.assembly_lab.objects import lookalikes as L
from sim.assembly_lab.objects import meshes as M
from sim.assembly_lab.objects import planner

ROOT = M.ROOT
HERE = Path(__file__).resolve().parent
TAXONOMY = ROOT / "contracts" / "taxonomy.json"
INSERTION_TOL = ROOT / "sdg" / "data" / "insertion_tolerance.json"
LIMITS = HERE.parent / "physics" / "limits.yaml"
PLA = F.P.PLA_DENSITY_G_PER_CM3
BASE_THICK = F.P.FIXTURE_Z


def product_ids(n: int) -> dict[int, str]:
    return {0: f"F0{n}", **{i: f"M{n}_0{i}" for i in range(1, 5)}}


def _r3(v) -> list[float]:
    return [round(float(x), 3) for x in v]


def feature_radius(f: dict) -> float | None:
    """Largest distance from the part origin that a locating feature moves per radian of yaw (None: yaw-free)."""
    x, y = f["xy_mm"]
    if f["shape"] == "round":
        r = math.hypot(x, y)
        return r if r > 1e-6 else None
    a, b = f["size_mm"][0] / 2, f["size_mm"][1] / 2
    if abs(x) < 1e-6 and abs(y) < 1e-6:
        return max(a, b)
    return max(math.hypot(x + sa * a, y + sb * b) for sa in (-1, 1) for sb in (-1, 1))


def locating_features(n: int, role: int, feats: dict) -> list[dict]:
    """The moving part's own features that locate it on the parts below (its insertion)."""
    if role == 1:
        hx, hy = F.housing_dims(n)
        body = {"name": "footprint_in_stops", "shape": "rect", "size_mm": [hx, hy], "xy_mm": [0, 0]}
        return [h for h in feats["holes"] if h["name"].startswith("socket")] + [body]
    if role == 3:
        return [h for h in feats["holes"] if h["name"] == "socket" or h["name"].startswith("guide_hole")]
    located = [p for p in feats["pegs"] if p["name"] in ("plug", "insert_body") or p["name"][:3] in ("arm", "leg")]
    return located


# xy capture from a CAD column sweep: the upper part dropped at 0-4 mm offsets in 8 directions
# onto the base + earlier steps; capture = min over directions of the end of the lead-in ramp (slope >= 0.35, steeper
# than PLA friction). Overrides the declared-mate value where a side has no chamfer (run10: every M4_02 landed on
# M4_01's flat top at the declared 2.0 mm).
CAPTURE_SWEEP_MM = {(3, 2): 1.4, (4, 2): 0.5, (4, 4): 0.5, (5, 4): 0.5, (6, 4): 0.5}
CAPTURE_SWEEP_NOTE = "CAD column sweep (agent 1, 2026-10-30): min over 8 directions of the chamfer ramp end, was {}"


def insertion(n: int, role: int, feats: dict, sweep: dict) -> dict | None:
    if role == 0:
        return None
    lower = [m for m in feats["mates_with"] if m["clearance_per_side_mm"] is not None and
             (m["with"].startswith("base_type") or int(m["with"][-1]) < role)]
    radii = [r for r in (feature_radius(f) for f in locating_features(n, role, feats) if "xy_mm" in f) if r]
    if n == 4 and role == 2:      # the insert's grooves ride on ribs at x +-11.5, y +4 (widths 3 / 2 + clearance)
        radii += [math.hypot(F.importlib.import_module(F.MODULES[4]).CAV_X / 2 - 1.0, y + w / 2 + 0.5)
                  for _, w, y in F.importlib.import_module(F.MODULES[4]).GUIDE_RIBS]
    clearance = min(m["clearance_per_side_mm"] for m in lower)
    capture = min(m["capture_mm"] for m in lower)
    source = None
    if (n, role) in CAPTURE_SWEEP_MM:
        source, capture = CAPTURE_SWEEP_NOTE.format(capture), CAPTURE_SWEEP_MM[(n, role)]
    r_eff = max(radii) if radii else None
    cad_tol = math.degrees(clearance / r_eff) if r_eff else None
    tols = [t for t in (cad_tol, (sweep or {}).get("yaw_tol_deg")) if t is not None and r_eff]
    out = {"onto": [m["with"] for m in lower], "axis": [0.0, 0.0, -1.0],
            "engagement_mm": max(m["engagement_mm"] for m in lower),
            "xy_clearance_mm": clearance, "xy_capture_mm": capture,
            "r_eff_mm": round(r_eff, 3) if r_eff else None,
            "yaw_tol_deg": round(min(tols), 3) if tols else None,
            "yaw_tol_cad_deg": round(cad_tol, 3) if cad_tol else None,
            "yaw_capture_deg": round(math.degrees(capture / r_eff), 3) if r_eff else None,
            "yaw_tol_rule": "min(clearance / r_eff from the CAD features, the simulated sweep); None = yaw-free (round)",
            "combined_rule": "|xy error| + r_eff * |yaw error (rad)| <= xy_clearance_mm to seat; <= xy_capture_mm to "
                             "be caught by the chamfer",
            "sim_sweep": sweep}
    if source:
        out["xy_capture_source"] = source
    return out


def mesh_entry(oid: str, frame: dict) -> tuple[dict, M.HeightMap]:
    tri = M.local_triangles(frame["name"], frame["T_local_from_stl"])
    vol, com = M.mesh_volume_com(tri)
    pts = tri.reshape(-1, 3)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    entry = {"bbox_mm": {"min": _r3(lo), "max": _r3(hi), "size": _r3(hi - lo)}, "volume_mm3": round(vol, 1),
             "mass_g": round(vol / 1000.0 * PLA, 2), "com_mm": _r3(com)}
    return entry, M.height_map(tri)


def max_force(oid: str, limits: dict) -> dict:
    """Global calculated limits, with a SIM_MEASURED per-part axial override where limits.yaml has one."""
    out = {k: limits[k] for k in ("axial_N", "lateral_N", "torque_Nm", "status")}
    override = limits.get("per_part_axial_N", {}).get(oid)
    if override:
        out.update(axial_N=override["axial_N"], axial_status=override["status"])
    return out


def object_entry(n: int, role: int, frames: dict, tax: dict, sweeps: dict, limits: dict) -> tuple[dict, M.HeightMap]:
    ids = product_ids(n)
    oid, frame = ids[role], frames[ids[role]]
    entry, hm = mesh_entry(oid, frame)
    feats = F.object_features(n, role, ids)
    F.check_features(oid, M.local_triangles(frame["name"], frame["T_local_from_stl"]), feats)
    origin = frame["installed_origin_mm"]
    top_local = entry["bbox_mm"]["max"][2]
    entry.update({
        "id": oid, "name": frame["name"], "product": n, "role": "base" if role == 0 else f"0{role}",
        "sheet_order": role, "family": tax[oid]["family"], "base_type": F.BASE_TYPE[n],
        "lookalikes": tax[oid]["lookalikes"], "installed_origin_mm": origin,
        "T_local_from_stl": frame["T_local_from_stl"], "grips": frame["grips"],
        "pegs": feats["pegs"], "holes": feats["holes"], "mates_with": feats["mates_with"],
        "seated_origin_z_mm": origin[2],
        "seated_top_z_mm": round(origin[2] + top_local, 3),
        "seated_height_mm": round(BASE_THICK + origin[2] + top_local, 3) if role else round(BASE_THICK + top_local, 3),
        "insert_axis": [0.0, 0.0, -1.0] if role else None,
        "pull_out_axis": [0.0, 0.0, 1.0] if role else None,
        "insertion": insertion(n, role, feats, sweeps.get(oid)),
        "max_force": max_force(oid, limits),
    })
    if role == 0:
        entry["housing_opening_mm"] = feats["housing_opening_mm"]
    return entry, hm


def _maps_dict(hms: dict[str, M.HeightMap]) -> dict:
    return {k: {"origin": h.origin_xy, "res": h.res, "top": h.top, "bottom": h.bottom} for k, h in hms.items()}


def allowed_rolls(catalog: dict, maps: dict) -> None:
    """Both CAD grips (roll_offset and +180) checked for finger clearance on the seated product at yaw 0, with the
    lower steps installed. J6 depends on the live pose and is checked by the planner at run time."""
    for oid, e in catalog.items():
        if e["sheet_order"] == 0:
            e["allowed_rolls_deg"] = []
            continue
        ids = product_ids(e["product"])
        scene = {"objects": [{"id": ids[0], "x": 400.0, "y": 0.0, "yaw": 0.0, "state": "installed"}]
                 + [{"id": ids[k], "state": f"on:{ids[0]}"} for k in range(1, e["sheet_order"] + 1)]}
        rolls = []
        for k in (0, 1):
            plan = planner.plan_pick(scene, oid, current_r=e["grips"][0]["roll_offset_deg"] + 180 * k, flip=k,
                                     catalog=catalog, maps=maps)
            if plan.feasible:
                rolls.append(round((e["grips"][0]["roll_offset_deg"] + 180.0 * k + 180.0) % 360.0 - 180.0, 3))
            elif plan.blocked_by.get(k) not in (None, ["J6_LIMIT"]):
                raise ValueError(f"{oid}: CAD grip k={k} collides on its own product: {plan.blocked_by[k]}")
        e["allowed_rolls_deg"] = rolls


def build() -> tuple[dict, dict]:
    logging.getLogger("fanuc_cad").setLevel(logging.ERROR)
    frames = M.grasp_frames()
    taxonomy = json.loads(TAXONOMY.read_text())
    tax = {o["object_id"]: o for o in taxonomy["objects"]}
    sweeps = json.loads(INSERTION_TOL.read_text())["parts"]
    sweeps = {k: {"yaw_tol_deg": v["yaw_tol_deg"], "shift_tol_mm": v["shift_tol_mm"], "symmetry": v["symmetry"],
                  "source": "sdg/data/insertion_tolerance.json"} for k, v in sweeps.items()}
    limits = yaml.safe_load(LIMITS.read_text())
    catalog, hms = {}, {}
    for n in range(1, 7):
        for role in range(5):
            entry, hm = object_entry(n, role, frames, tax, sweeps, limits)
            catalog[entry["id"]], hms[entry["id"]] = entry, hm
    maps = _maps_dict(hms)
    allowed_rolls(catalog, maps)
    groups = L.groups(taxonomy)
    for g in groups:
        for oid in g:
            catalog[oid]["distinguishing_dims_mm"] = L.distinguishing(oid, g, catalog, maps)
    for oid, e in catalog.items():
        e.setdefault("distinguishing_dims_mm", {"group": [oid], "note": "no look-alike in contracts/taxonomy.json"})
    summary = {"lookalike_groups": [L.group_summary(g, catalog, maps) for g in groups],
               "size_twins_within_1mm": L.size_twins(catalog, maps, groups)}
    return {"source": __doc__.split("\n\n")[2].strip(), "units": "mm, deg, g", **summary,
            "objects": {k: catalog[k] for k in sorted(catalog)}}, hms


def _clean(o):
    """numpy scalars / tuples -> plain YAML types."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main() -> None:
    doc, hms = build()
    (HERE / "objects.yaml").write_text("# GENERATED by sim/assembly_lab/objects/build_objects.py - do not edit.\n"
                                       + yaml.safe_dump(_clean(doc), sort_keys=False, width=120))
    arrays = {}
    for k, h in hms.items():
        arrays[f"{k}/origin"] = np.array(h.origin_xy)
        arrays[f"{k}/res"] = np.array(h.res)
        arrays[f"{k}/top"] = h.top.astype(np.float32)
        arrays[f"{k}/bottom"] = h.bottom.astype(np.float32)
    np.savez_compressed(HERE / "heightmaps.npz", **arrays)
    print(f"wrote {len(doc['objects'])} objects to objects.yaml and heightmaps.npz")


if __name__ == "__main__":
    main()
