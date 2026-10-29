"""Check every exported file and every assembly, then write BUILD_SUMMARY.txt.

    uv run python cad/products/validate_outputs.py        (run after build_all.py)

Per file : exists, STEP re-opens (CadQuery and FreeCAD), STL re-opens, watertight,
           positive volume, one connected body, bounding box, mass estimate.
Per assembly (exported STEP files re-placed at their installed offsets):
           no static intersections, 1.0 mm sampled vertical insertion/removal paths,
           finger clearance around each grip pad when that part is next to be removed.
Exit code 0 only if everything passes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cadquery as cq
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_geometry import GripZone, box_span
from parameters import MAX_PART_MASS_G, PLA_DENSITY_G_PER_CM3

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
FREECADCMD = Path("/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd")
VOL_TOL = 0.01          # mm^3; intersections smaller than this are face contact, not overlap
PATH_STEP = 1.0         # mm
FINGER_DEPTH = 10.0     # free space required beyond each contact face
APPROACH = 20.0         # clear vertical approach above the pad
LATERAL_PROBE = 0.3     # every fit must allow this sideways play (catches zero-clearance contact)
MIN_CONTACT = 6.0       # finger proxy covers the upper 6 mm of the pad (or the full pad if shorter)
# Keyed interfaces besides housing-on-fixture: (receiving part, keyed part)
KEYED_PAIRS = (("M3_01_Housing", "M3_02_Actuator_Stem"), ("M4_01_Housing", "M4_02_Sensor_Insert"))


def file_checks(registry: dict) -> tuple[list[dict], dict]:
    rows, shapes = [], {}
    for name, info in registry["parts"].items():
        step = OUT / "step" / info["folder"] / f"{name}.step"
        stl = OUT / "stl" / info["folder"] / f"{name}.stl"
        row = {"name": name, "fixture": info["folder"] == "FIXTURES", "step": str(step),
               "exists": step.is_file() and stl.is_file()}
        if row["exists"]:
            shape = cq.importers.importStep(str(step)).val()
            shapes[name] = shape
            mesh = trimesh.load_mesh(str(stl))
            bb = mesh.bounds
            row.update(step_valid=shape.isValid(), step_solids=len(shape.Solids()),
                       step_volume=shape.Volume(), stl_watertight=bool(mesh.is_watertight),
                       stl_volume=float(mesh.volume), stl_bodies=len(mesh.split(only_watertight=False)),
                       bbox=[round(float(v), 2) for v in (bb[1] - bb[0])])
            row["mass_g"] = row["step_volume"] / 1000.0 * PLA_DENSITY_G_PER_CM3
        rows.append(row)
    return rows, shapes


def freecad_checks(rows: list[dict]) -> dict:
    paths = [r["step"] for r in rows if r["exists"]]
    if not FREECADCMD.is_file():
        return {"error": f"FreeCAD not found at {FREECADCMD}"}
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = Path(tmp) / "in.json", Path(tmp) / "out.json"
        fin.write_text(json.dumps(paths))
        script = Path(__file__).resolve().parent / "freecad_check.py"
        env = dict(os.environ, FC_CHECK_IN=str(fin), FC_CHECK_OUT=str(fout))
        proc = subprocess.run([str(FREECADCMD), "-c", f"exec(open({str(script)!r}).read())"],
                              env=env, capture_output=True, text=True, timeout=600, check=False)   # exit code read below
        if not fout.is_file():
            return {"error": f"freecadcmd exit {proc.returncode}: {proc.stderr[-400:]}"}
        return json.loads(fout.read_text())


def placed(shapes: dict, registry: dict, name: str) -> cq.Shape:
    return shapes[name].translate(cq.Vector(*registry["parts"][name]["origin"]))


def overlap(a: cq.Shape, b: cq.Shape) -> float:
    ba, bb = a.BoundingBox(), b.BoundingBox()
    if (ba.xmin >= bb.xmax or bb.xmin >= ba.xmax or ba.ymin >= bb.ymax or bb.ymin >= ba.ymax
            or ba.zmin >= bb.zmax or bb.zmin >= ba.zmax):
        return 0.0
    return a.intersect(b).Volume()


def static_check(world: dict, order: list[str]) -> list[str]:
    fails = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            v = overlap(world[a], world[b])
            if v > VOL_TOL:
                fails.append(f"{a} x {b}: {v:.3f} mm^3")
    return fails


def path_check(world: dict, order: list[str]) -> tuple[list[str], int]:
    """Lift each part straight up in 1 mm steps past everything installed before it.

    Insertion is the same straight line travelled downward, so one sweep covers both directions.
    """
    fails, samples = [], 0
    for i, name in enumerate(order[1:], start=1):
        below = [world[n] for n in order[:i]]
        part = world[name]
        clear_z = max(s.BoundingBox().zmax for s in below) - part.BoundingBox().zmin + PATH_STEP
        dz = 0.0
        while dz <= clear_z:
            moved = part.translate(cq.Vector(0, 0, dz))
            samples += 1
            for n, other in zip(order[:i], below, strict=True):
                v = overlap(moved, other)
                if v > VOL_TOL:
                    fails.append(f"{name} vs {n} at +{dz:.0f} mm: {v:.3f} mm^3")
            dz += PATH_STEP
    return fails, samples


def clearance_check(world: dict, order: list[str]) -> list[str]:
    """Nudge each installed part sideways; any hit means a fit has under LATERAL_PROBE of play."""
    fails = []
    for i, name in enumerate(order[1:], start=1):
        for dx, dy in ((LATERAL_PROBE, 0), (-LATERAL_PROBE, 0), (0, LATERAL_PROBE), (0, -LATERAL_PROBE)):
            moved = world[name].translate(cq.Vector(dx, dy, 0))
            for n in order[:i]:
                v = overlap(moved, world[n])
                if v > VOL_TOL:
                    fails.append(f"{name} vs {n} shifted ({dx:+.1f},{dy:+.1f}): {v:.3f} mm^3")
    return fails


def finger_proxies(z: GripZone) -> list[cq.Shape]:
    z_lo, z_hi = z.z_top - min(MIN_CONTACT, z.height), z.z_top + APPROACH
    lo, hi = z.span
    c, w = z.center, z.half_width
    zones = ((c + w, c + w + FINGER_DEPTH, z_lo),      # finger beside the + contact face
             (c - w - FINGER_DEPTH, c - w, z_lo),      # finger beside the - contact face
             (c - w, c + w, z.z_top))                  # vertical approach above the pad
    boxes = []
    for a0, a1, zb in zones:
        spans = (a0, a1, lo, hi) if z.axis == "X" else (lo, hi, a0, a1)
        boxes.append(box_span(*spans, zb, z_hi).val())
    return boxes


def grip_check(world: dict, order: list[str], registry: dict) -> list[str]:
    """When part k is next to remove, parts 0..k are present; fingers must not hit any of them."""
    fails = []
    for k, name in enumerate(order):
        for g in registry["parts"][name]["grips"]:
            for proxy in finger_proxies(GripZone(**{**g, "span": tuple(g["span"])})):
                for n in order[:k + 1]:
                    v = overlap(proxy, world[n])
                    if v > VOL_TOL:
                        fails.append(f"{name} grip: finger zone hits {n} ({v:.2f} mm^3)")
    return fails


def anti_rotation_check(shapes: dict, registry: dict) -> tuple[list[str], int]:
    """A keyed part turned 180 degrees about its own vertical axis must collide with its seat."""
    pairs = [(order[0], order[1]) for label, order in registry["assemblies"].items() if "wrong" not in label]
    pairs += [p for p in KEYED_PAIRS if p[0] in shapes and p[1] in shapes]
    fails = []
    for seat, keyed in pairs:
        ox, oy, _ = registry["parts"][keyed]["origin"]
        turned = placed(shapes, registry, keyed).rotate(cq.Vector(ox, oy, 0), cq.Vector(ox, oy, 1), 180)
        if overlap(turned, placed(shapes, registry, seat)) <= VOL_TOL:
            fails.append(f"{keyed} fits {seat} when turned 180 degrees")
    return fails, len(pairs)


def verdict(fails: list[str]) -> str:
    return "PASS" if not fails else "FAIL - " + "; ".join(fails)


def tally(rows: list[dict], fc: dict, asm: dict) -> dict:
    """Reduce per-file and per-assembly results to the counts the summary reports."""
    ok = lambda r, k: r.get("exists") and r.get(k)
    same_volume = lambda r: abs(fc[r["step"]]["volume"] - r["step_volume"]) < 1e-3 * r["step_volume"]
    wrong = [a for a in asm.values() if a["wrong"]]
    return {
        "prod": [r for r in rows if not r["fixture"]],
        "fix": [r for r in rows if r["fixture"]],
        "stl_ok": [r for r in rows if ok(r, "stl_watertight") and r["stl_volume"] > 0 and r["stl_bodies"] == 1],
        "step_ok": [r for r in rows if ok(r, "step_valid") and r["step_solids"] == 1 and r["step_volume"] > 0],
        "fc_ok": [r for r in rows if r["exists"] and fc.get(r["step"], {}).get("valid")
                  and fc[r["step"]]["solids"] == 1 and same_volume(r)],
        "heavy": [r for r in rows if not r["fixture"] and r.get("mass_g", 0) > MAX_PART_MASS_G],
        "n_asm": sum(1 for a in asm.values() if not a["wrong"]),
        "static_ok": sum(1 for a in asm.values() if not a["wrong"] and not a["static"]),
        "path": [f for a in asm.values() for f in a["path"]],
        "grip": [f for a in asm.values() for f in a["grip"]],
        "play": [f for a in asm.values() for f in a["play"]],
        "n_wrong": len(wrong),
        "wrong_ok": all(not (a["static"] or a["path"]) for a in wrong),
        "samples": sum(a["samples"] for a in asm.values()),
    }


def summarise(rows: list[dict], fc: dict, asm: dict, notes: list[str], anti: tuple) -> tuple[str, bool]:
    t = tally(rows, fc, asm)
    n_prod, n_fix, n = sum(r["exists"] for r in t["prod"]), sum(r["exists"] for r in t["fix"]), len(rows)
    heaviest = max(r.get("mass_g", 0) for r in t["prod"])
    lines = [
        f"{n_prod}/24 product STL files generated", f"{n_prod}/24 product STEP files generated",
        f"{n_fix}/6 fixture STL files generated", f"{n_fix}/6 fixture STEP files generated",
        f"{len(t['stl_ok'])}/{n} STL files watertight, positive volume, one connected body",
        f"{len(t['step_ok'])}/{n} STEP files re-open in CadQuery as one valid solid",
        f"{len(t['fc_ok'])}/{n} STEP files re-open in FreeCAD as one valid solid (volume agrees)",
        f"{t['static_ok']}/{t['n_asm']} assemblies free of unintended static intersections",
        f"Insertion/removal path check ({t['samples']} samples at {PATH_STEP:.0f} mm): {verdict(t['path'])}",
        f"Sliding-fit play >= {LATERAL_PROBE} mm in X and Y for every installed part: {verdict(t['play'])}",
        f"Grip-pad finger clearance (10 mm each side, 20 mm above): {verdict(t['grip'])}",
        f"Wrong-part cases ({t['n_wrong']}) fit, insert and remove cleanly: {'PASS' if t['wrong_ok'] else 'FAIL'}",
        f"Anti-180 degree keying ({anti[1]} interfaces) blocks reversed installation: {verdict(anti[0])}",
        f"Heaviest product part (100% solid PLA estimate): {heaviest:.1f} g (limit {MAX_PART_MASS_G:.0f} g): "
        + verdict([r["name"] for r in t["heavy"]]),
    ]
    files_ok = n == 30 and n_prod == 24 and n_fix == 6
    stl_pass = files_ok and len(t["stl_ok"]) == len(t["step_ok"]) == len(t["fc_ok"]) == 30
    motion_pass = not t["path"] and t["wrong_ok"] and t["static_ok"] == t["n_asm"] == 6
    passed = stl_pass and motion_pass and not (t["grip"] or t["play"] or t["heavy"] or anti[0])
    status = ["", f"CAD generation: {'PASS' if files_ok else 'FAIL'}",
              f"STL printability checks: {'PASS' if stl_pass else 'FAIL'}",
              f"Digital insertion/removal: {'PASS' if motion_pass else 'FAIL'}",
              "Physical print fit: NOT TESTED", "Physical FANUC gripper compatibility: NOT TESTED",
              "Camera error detection: NOT TESTED", "Physical gripper test: NOT PERFORMED",
              "Physical print-fit test: NOT PERFORMED"]
    bboxes = ["", "Bounding boxes as exported (X x Y x Z mm) / mass estimate at 100% solid:"]
    bboxes += [f"  {r['name']:30s} {' x '.join(f'{v:.1f}' for v in r['bbox'])}   {r['mass_g']:.1f} g"
               for r in rows if r["exists"]]
    extra = ["", "Build notes:"] + [f"  {note}" for note in notes] if notes else []
    return "\n".join(lines + status + bboxes + extra) + "\n", passed


def main() -> int:
    registry = json.loads((OUT / "part_registry.json").read_text())
    rows, shapes = file_checks(registry)
    fc = freecad_checks(rows)
    if "error" in fc:
        print("FreeCAD check failed:", fc["error"])
        fc = {}
    asm_results = {}
    for label, order in registry["assemblies"].items():
        if not all(n in shapes for n in order):
            asm_results[label] = {"wrong": "wrong" in label, "static": ["missing files"], "path": ["missing files"],
                                  "grip": [], "samples": 0, "play": []}
            continue
        world = {n: placed(shapes, registry, n) for n in order}
        path_fails, samples = path_check(world, order)
        asm_results[label] = {"wrong": "wrong" in label, "static": static_check(world, order),
                              "path": path_fails, "grip": grip_check(world, order, registry), "samples": samples,
                              "play": clearance_check(world, order)}
        print(f"{label}: static={asm_results[label]['static'] or 'ok'} path={path_fails or 'ok'} "
              f"grip={asm_results[label]['grip'] or 'ok'} play={asm_results[label]['play'] or 'ok'}")
    anti = anti_rotation_check(shapes, registry)
    text, passed = summarise(rows, fc, asm_results, registry.get("build_notes", []), anti)
    (ROOT / "BUILD_SUMMARY.txt").write_text(text)
    print(text)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
