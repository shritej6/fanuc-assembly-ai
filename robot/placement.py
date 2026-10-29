"""Write robot/cell/submission/placement.yaml: where the operator puts every object in each real-cell rollout.

    uv run python -m robot.placement            # same seed, same file

Three rollouts of three products each (PLAN 7: 3 products per trial; ~17 objects and 12 inserts do not fit ten
rollouts into a 30-minute slot). Each rollout has its planned faults and the spares they need. Positions are
x, y in mm in the robot frame (UF0, as obs["tcp_pose"]); no yaw is given because parts stand in their installed
orientation at any yaw and the policy measures each pose. Every footprint stays inside the reachable work area and
its gaps (sdg/configs/cell_layout.yaml), spares go to the fixed store slots, and the reject zone stays empty.
Every object's straight-down grasp and approach pose, at four gripper rolls, must pass the arm's IK
(sdg/wrist_ik.py), or nothing is written.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import yaml

from sdg.wrist_ik import reachable

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "robot" / "cell" / "submission" / "placement.yaml"
SEED = 2026_10_29
BASE_GAP_MM = 60.0       # between base edges: the gripper works around a product that grows on each base
BASE_PART_GAP_MM = 25.0
PART_GAP_MM = 20.0       # finger clearance around every loose part
FALLEN_SCALE = 1.5       # a part lying on its side covers more table than its upright footprint
TRIES = 20_000
GRASP_Z, APPROACH_Z = -325.0, -215.0   # UF0: 10 mm above the table (-335) and 120 mm above it
ROLLS = (-90.0, 0.0, 90.0, 180.0)      # any part yaw: the gripper must reach it at every roll

# products built, faults planned and spares stocked; a fault entry is (kind, object_id, operator note)
ROLLOUTS = [
    {"products": [1, 2, 3], "faults": [], "spares": ["M1_03", "M2_03", "M2_02", "M3_03"]},
    {"products": [4, 5, 6], "faults": [("DEFECT", "M4_04", "snapped lock arm; replaces the good M4_04")],
     "spares": ["M4_04", "M6_02"]},
    {"products": [1, 3, 5], "faults": [("TRAP", "M4_04", "look-alike of M5_04; product 4 is not built here"),
                                       ("FALLEN", "M3_03", "lying on its side; replaces the standing M3_03")],
     "spares": ["M3_03", "M1_03"]},
]


class PlacementError(ValueError):
    """No layout satisfies the zones and gaps; nothing is written."""


def load() -> tuple[dict, dict, dict]:
    geometry = json.loads((ROOT / "sdg" / "data" / "object_geometry.json").read_text())["objects"]
    layout = yaml.safe_load((ROOT / "sdg" / "configs" / "cell_layout.yaml").read_text())
    names = {oid: Path(g["stl"]).stem.split("_", 2 if oid.startswith("M") else 1)[-1].replace("_", " ")
             for oid, g in geometry.items()}
    return geometry, layout, names


def _inside(x: float, y: float, r: float, layout: dict) -> bool:
    (x0, x1), (y0, y1) = layout["work_area"]["x"], layout["work_area"]["y"]
    d = math.hypot(x, y)
    return (x0 + r <= x <= x1 - r and y0 + r <= y <= y1 - r
            and layout["reach_radius_min"] + r <= d <= layout["reach_radius_max"] - r)


def _gap(a: str, b: str) -> float:
    if a == b == "base":
        return BASE_GAP_MM
    return PART_GAP_MM if a == b == "part" else BASE_PART_GAP_MM


def _place(r: float, kind: str, placed: list[tuple[float, float, float, str]], layout: dict,
           rng: random.Random) -> tuple[int, int]:
    """A random spot whose footprint keeps its gap to everything already placed."""
    (x0, x1), (y0, y1) = layout["work_area"]["x"], layout["work_area"]["y"]
    for _ in range(TRIES):
        x, y = round(rng.uniform(x0, x1)), round(rng.uniform(y0, y1))
        if _inside(x, y, r, layout) and all(math.hypot(x - px, y - py) >= r + pr + _gap(kind, pk)
                                            for px, py, pr, pk in placed):
            placed.append((x, y, r, kind))
            return x, y
    raise PlacementError(f"no free {kind} spot of radius {r} mm after {TRIES} tries")


def rollout(spec: dict, geometry: dict, layout: dict, names: dict, rng: random.Random) -> dict:
    placed: list = []
    objects: dict[str, list[int]] = {}
    for p in spec["products"]:
        base = f"F0{p}"
        objects[f"{base} base ({names[base]}, product {p})"] = list(_place(
            geometry[base]["upright"]["footprint_r"], "base", placed, layout, rng))
    replaced = {oid for kind, oid, _ in spec["faults"] if kind in ("DEFECT", "FALLEN")}
    loose = [(f"{oid} {names[oid]}", oid, 1.0) for p in spec["products"] for oid in (f"M{p}_0{i}" for i in range(1, 5))
             if oid not in replaced]
    loose += [(f"{kind} {oid} {names[oid]} ({note})", oid, FALLEN_SCALE if kind == "FALLEN" else 1.0)
              for kind, oid, note in spec["faults"]]
    for label, oid, scale in sorted(loose, key=lambda item: -geometry[item[1]]["upright"]["footprint_r"]):
        objects[label] = list(_place(geometry[oid]["upright"]["footprint_r"] * scale, "part", placed, layout, rng))
    slots = layout["spare_store"]["slots"]
    if len(spec["spares"]) > len(slots):
        raise PlacementError(f"{len(spec['spares'])} spares for {len(slots)} store slots")
    for slot, oid in zip(slots, spec["spares"], strict=False):
        objects[f"spare {oid} {names[oid]} (store slot {slot['id']})"] = [round(slot["x"]), round(slot["y"])]
    built = ", ".join(str(p) for p in spec["products"][:-1]) + f" and {spec['products'][-1]}"
    return {"objects": objects,
            "prompt": f"Assemble products {built} on their bases. Reject any part that is defective, fallen or not "
                      f"in these products, and use a spare from the store instead."}


def unreachable(out: dict) -> list[str]:
    """Objects with a straight-down grasp or approach pose the arm's IK cannot reach."""
    return [f"rollout {i} {name}" for i, r in enumerate(out["rollouts"], start=1) for name, (x, y) in r["objects"].items()
            if not all(reachable([x, y, z, -180.0, 0.0, roll]) for z in (GRASP_Z, APPROACH_Z) for roll in ROLLS)]


def placement(seed: int = SEED) -> dict:
    geometry, layout, names = load()
    rng = random.Random(seed)
    out = {"rollouts": [rollout(spec, geometry, layout, names, rng) for spec in ROLLOUTS]}
    bad = unreachable(out)
    if bad:
        raise PlacementError(f"unreachable: {bad}")
    return out


def main() -> None:
    text = yaml.safe_dump(placement(), sort_keys=False, allow_unicode=True, width=200, default_flow_style=None)
    OUT.write_text("# Generated by robot/placement.py (do not edit by hand): 3 rollouts x 3 products, x y in mm (UF0).\n"
                   "# Parts stand in their installed orientation at any yaw unless marked FALLEN.\n" + text)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
