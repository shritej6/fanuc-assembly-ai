"""Generate a one-product-per-rollout real-cell placement layout.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import copy
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate as V

PRODUCTS = {1: 5, 2: 5, 3: 5}            # rollout -> product (product 5 gives the best demo yield)
SEED0 = {1: 1100, 2: 2200, 3: 3300}
WORK_X, WORK_Y = (220.0, 470.0), (-270.0, 270.0)   # sim/assembly_lab/env/sampler.py (training spawn)
SPACING = {"F": 40.0, "M": 26.0}
PROMPT = ("Assemble product {} on its base. Reject any part that is defective, fallen or not in this product, "
          "and use a spare from the store instead.")
MAX_TRIES = 4000


def sample(rng: np.random.Generator, prod: int) -> list[dict] | None:
    ids = [V.base_of(prod), *V.parts_of(prod)]
    placed: list[tuple[float, float, float]] = []
    objs = []
    for oid in ids:
        rad = SPACING[oid[0]]
        for _ in range(400):
            x, y = rng.uniform(*WORK_X), rng.uniform(*WORK_Y)
            if 180.0 <= math.hypot(x, y) <= 490.0 and all(math.hypot(x - a, y - b) >= rad + rb for a, b, rb in placed):
                break
        else:
            return None
        placed.append((x, y, rad))
        objs.append({"id": oid, "spot": "", "cx": round(x), "cy": round(y), "yaw": round(rng.uniform(-180, 180))})
    for o in objs:
        o["ox"], o["oy"] = V.origin_from_centre(o["id"], o["cx"], o["cy"], o["yaw"])
    return objs


def box_of(o: dict) -> tuple[float, float, float, float]:
    h = V.SLOT_HALF_MM
    return o["cx"] - h, o["cx"] + h, o["cy"] - h, o["cy"] + h


def overlaps(a: tuple, b: tuple) -> bool:
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3]


def checks(objs: list[dict], prod: int) -> list[str]:
    """Every validate.py check for one product; returns the failures (empty = pass)."""
    pts = V.footprint_points(objs)
    bad = list(V.zone_report(objs, pts))
    g = V.gap_report(objs, pts)
    need = {"base_part": V.GAP["base_to_part_gap_mm"], "part_part": V.PART_GAP_MM}
    for kind, n in need.items():
        if min(g["true"][kind][0], g["circle"][kind][0]) < n:
            bad.append(f"gap {kind}")
    base = {prod: V.base_of(prod)}
    for r in V.run_sequence(objs, [prod], base):
        pk, pl = r["pick"], r["place"]
        band = pk[3] is not None and pl and pl[3] is not None and V.roll_off(pk[3]) <= V.ROLL_BAND \
            and V.roll_off(pl[3]) <= V.ROLL_BAND
        if not (r["ok"] and band):
            bad.append(f"plan {r['part']}")
    if bad:
        return bad
    for o in objs:
        if o["id"].startswith("F"):
            continue
        x0, x1, y0, y1 = box_of(o)
        if not (x0 <= o["ox"] <= x1 and y0 <= o["oy"] <= y1):
            bad.append(f"{o['id']} CAD origin outside its box")
        for d in [(10, 0, 0), (-10, 0, 0), (0, 10, 0), (0, -10, 0), (0, 0, 10), (0, 0, -10)]:
            rr = V.run_sequence(objs, [prod], base, perturb={o["id"]: d})
            if not next(x for x in rr if x["part"] == o["id"])["ok"]:
                bad.append(f"tolerance {o['id']} {d}")
    return bad


def choose(n: int, prod: int, taken: list[tuple]) -> tuple[list[dict], int, int]:
    for k in range(MAX_TRIES):
        objs = sample(np.random.default_rng(SEED0[n] + k), prod)
        if objs is None:
            continue
        parts = [o for o in objs if not o["id"].startswith("F")]
        own = [box_of(o) for o in parts]
        if any(overlaps(a, b) for a, b in itertools.combinations(own, 2)) or any(
                overlaps(a, b) for a in own for b in taken):
            continue
        if not checks(objs, prod):
            return objs, SEED0[n] + k, k + 1
    raise SystemExit(f"rollout {n}: no layout passed in {MAX_TRIES} seeds")


def main() -> int:
    parse = V.instruction_parser()
    log: list[str] = []
    rollouts: dict[int, list[dict]] = {}
    taken: list[tuple] = []
    spots: dict[str, tuple[float, float]] = {}
    first, seed, tries = choose(1, PRODUCTS[1], taken)
    for i, o in enumerate(first):
        o["spot"] = "B" if o["id"].startswith("F") else f"P{i}"
        if not o["id"].startswith("F"):
            spots[o["spot"]] = (o["cx"], o["cy"])
    for n, prod in PRODUCTS.items():
        if prod != PRODUCTS[1]:
            raise SystemExit("every rollout repeats rollout 1's product and layout")
        objs = copy.deepcopy(first)
        rollouts[n] = objs
        got = parse(PROMPT.format(prod))
        log.append(f"=== Rollout {n}: product {prod}, training-like layout from seed {seed} ({tries} tries), "
                   f"{len(objs)} objects")
        log.append(f"prompt -> parse_requested_products = {got}  {'OK' if list(got) == [prod] else 'MISMATCH'}")
        if list(got) != [prod]:
            return 1
    # Reuse validate.py's full report + renderer on the chosen layouts.
    V.DESIGN = {n: {"products": [p]} for n, p in PRODUCTS.items()}
    V.PART_SPOTS = spots
    V.PROMPT = PROMPT
    V.build_rollout = lambda n: rollouts[n]
    V.yaml_out = yaml_out
    V.positions_md = positions_md
    code = V.main()
    out = (V.OUT / "validate_output.txt").read_text()
    (V.OUT / "validate_output.txt").write_text("\n".join(log) + "\n\n" + out)
    print("\n".join(log))
    return code


def yaml_out(rollouts: dict[int, list[dict]]) -> str:
    head = [
        ("# CSI real-cell slot 2026-10-31 14:00 CT. Generated by single_product.py (do not edit by hand): 3 rollouts x 1 "
        "product."),
        ("# One product per rollout, laid out like the GR00T training scenes (base + its 4 parts, nothing else on the "
        "table)."),
        "# Positions: [x, y, yaw] in the robot frame UF0 (same x, y as obs[\"tcp_pose\"]), mm and degrees.",
        "# x, y = centre of the part's footprint seen from above; yaw = direction of the part's CAD local +x axis,",
        "# measured from robot +x (towards the front edge) turning towards +y (table_plan.png yaw cross).",
        "# Every part stands in its installed orientation.",
    ]
    data = {"rollouts": []}
    for n, objs in rollouts.items():
        entry = {V.pretty(o["id"]): [round(o["cx"]), round(o["cy"]), round(o["yaw"])] for o in objs}
        data["rollouts"].append({"objects": entry, "prompt": PROMPT.format(PRODUCTS[n])})
    return "\n".join(head) + "\n" + yaml.safe_dump(data, sort_keys=False, width=200, allow_unicode=True,
                                                    default_flow_style=None)


def positions_md(rollouts: dict[int, list[dict]], rolls: dict[int, dict]) -> str:
    out = ["# CSI real-cell placement, 2026-10-31 14:00 CT (one product per rollout)", "",
           ("x, y = centre of the part's footprint (mm, robot frame UF0, as obs[\"tcp_pose\"]). x grows towards the front "
           "edge (away from the robot); +y is to the right in table_plan.png and in the PNGs here."),
           ("yaw = direction of the part's CAD local +x axis (red arrow in rollout_N.png), degrees from +x turning towards "
           "+y. Every part stands in its installed orientation. Nothing else on the table."), ""]
    for n, objs in rollouts.items():
        out += [f"## Rollout {n}: product {PRODUCTS[n]}", "", f"Prompt: `{PROMPT.format(PRODUCTS[n])}`", "",
                "| # | object | x | y | yaw | pick roll r | insert roll r |", "|---|---|---|---|---|---|---|"]
        for i, o in enumerate(objs, 1):
            pr = rolls[n].get(o["id"])
            pick = f"{pr[0]:.0f}" if pr else "-"
            ins = f"{pr[1]:.0f}" if pr else "-"
            out.append(f"| {i} | {V.pretty(o['id'])} | {o['cx']:.0f} | {o['cy']:.0f} | {o['yaw']:.0f} | {pick} | {ins} |")
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
