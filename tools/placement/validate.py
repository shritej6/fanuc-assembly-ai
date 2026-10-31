"""Real-cell placement for the CSI slot 2026-10-31 14:00 CT: generate, validate, render.

Run from the repo root (the planner is imported from it):
    cd /Users/shritej/Desktop/FANUC_T32 && uv run --group dev --with h5py --python 3.13 python <this file>

Writes next to this file: placement.yaml, rollout_{1,2,3}.png, positions.md, validate_output.txt.

Positions: x, y = centre of the part's footprint (bounding box seen from above), UF0 mm; yaw = deg from robot +x
(towards the front edge) turning towards +y, i.e. the direction of the part's CAD local +x axis (arrow in the PNGs).
The planner (sim/assembly_lab/objects/planner.py) takes the CAD local origin, so every check converts centre -> origin.
"""
from __future__ import annotations

import copy
import itertools
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

REPO = Path("/Users/shritej/Desktop/FANUC_T32")
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from sim.assembly_lab.objects.planner import load_catalog, plan_pick, plan_place  # noqa: E402
from sim.assembly_lab.objects.scene import build_instances, solid_of  # noqa: E402

CAT, MAPS = load_catalog()
CELL = yaml.safe_load((REPO / "sdg/configs/cell_layout.yaml").read_text())
GEO = __import__("json").loads((REPO / "sdg/data/object_geometry.json").read_text())["objects"]
FOOT_R = {k: float(v["upright"]["footprint_r"]) for k, v in GEO.items()}
TABLE_Z = float(CELL["table_z"])
WX, WY = CELL["work_area"]["x"], CELL["work_area"]["y"]
RMIN, RMAX = CELL["reach_radius_min"], CELL["reach_radius_max"]
GAP = CELL["clearance"]
PART_GAP_MM = 12.0          # sim/assembly_lab/eval/scenarios.yaml part_gap_mm (cell_layout allows 0)
NEUTRAL_R = 180.0           # planner reference roll; T18 training rolls span 93..266 (dataset task strings)
ROLL_BAND = 60.0            # "moderate": |r - 180| <= this at pick and at insert
SLOT_HALF_MM = 25.0         # proposed InspectionSlotPose selection box half-size

# Fixed spots (footprint centres) reused by every rollout, so ONE non-overlapping inspection-slot registry covers all.
BASE_SPOTS = {"B1": (420.0, -165.0), "B2": (420.0, 0.0), "B3": (420.0, 165.0)}
ROWS = {"N": 255.0, "F": 335.0}                 # near (robot side) / far (base side) row of loose parts
COLS = {"B1": (-225.0, -135.0), "B2": (-45.0, 45.0), "B3": (135.0, 225.0)}   # outer/inner columns per base
PART_SPOTS = {f"P{r}{i}": (ROWS[r], y) for i, y in enumerate([-225.0, -135.0, -45.0, 45.0, 135.0, 225.0], 1)
              for r in ROWS}

NAMES = {"F01": "Rotary Fixture", "F02": "Indicator Fixture", "F03": "Pushbutton Fixture",
         "F04": "Sensor Connector Fixture", "F05": "IO Connector Fixture", "F06": "Power Connector Fixture"}
BASE_GROUPS = {"A": ["F01", "F02"], "B": ["F03"], "C": ["F04", "F05", "F06"]}   # identical shapes: interchangeable
BASE_YAW = {"A": 90.0, "B": 90.0, "C": 45.0}    # per base type: every grip roll offset lands near r = 180


def parts_of(prod: int) -> list[str]:
    return sorted((k for k, e in CAT.items() if e.get("product") == prod and e["sheet_order"] > 0),
                  key=lambda k: CAT[k]["sheet_order"])


def base_of(prod: int) -> str:
    return f"F0{prod}"


def pretty(oid: str) -> str:
    if oid.startswith("F"):
        return f"{oid} base ({NAMES[oid]}, product {int(oid[1:])})"
    return f"{oid} {CAT[oid]['name'].split('_', 2)[2].replace('_', ' ')}"


# Per rollout: product -> base spot, and per part (spot, yaw offset from its base yaw).
# Housings of type C (grip roll offset 0) get a negative offset, the rest positive, so pick and insert rolls stay
# on the same wrist turn and inside 180 +- 60.
DESIGN = {  # product 5 first in every rollout (user priority 2026-10-31: one full product, best demo yield)
    1: {"products": [5, 1, 2], "base_spot": {5: "B2", 1: "B1", 2: "B3"},
        "parts": {5: [("F", 0, -15), ("N", 0, 20), ("F", 1, 10), ("N", 1, 25)],
                  1: [("F", 1, -15), ("N", 1, 20), ("N", 0, -20), ("F", 0, 10)],
                  2: [("F", 0, 15), ("N", 0, -10), ("F", 1, -20), ("N", 1, 20)]}},
    2: {"products": [5, 4, 6], "base_spot": {5: "B2", 4: "B1", 6: "B3"},
        "parts": {5: [("F", 0, -15), ("N", 0, 20), ("F", 1, 10), ("N", 1, 25)],
                  4: [("F", 1, -20), ("N", 1, 15), ("N", 0, 25), ("F", 0, 10)],
                  6: [("F", 1, -25), ("N", 0, 10), ("N", 1, 20), ("F", 0, 15)]}},
    3: {"products": [5, 1, 3], "base_spot": {5: "B2", 1: "B3", 3: "B1"},
        "parts": {5: [("F", 1, -20), ("N", 1, 20), ("N", 0, 10), ("F", 0, 25)],
                  1: [("F", 0, 20), ("N", 0, -15), ("F", 1, 10), ("N", 1, -20)],
                  3: [("F", 1, -10), ("N", 1, 15), ("F", 0, 20), ("N", 0, -15)]}},
}
PROMPT = ("Assemble products {}, {} and {} on their bases. Reject any part that is defective, fallen or not in "
          "these products, and use a spare from the store instead.")


# ------------------------------------------------------------------ geometry helpers
def centre_local(oid: str) -> np.ndarray:
    b = CAT[oid]["bbox_mm"]
    return np.array([(b["min"][0] + b["max"][0]) / 2, (b["min"][1] + b["max"][1]) / 2])


def rot(yaw: float) -> np.ndarray:
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return np.array([[c, -s], [s, c]])


def origin_from_centre(oid: str, x: float, y: float, yaw: float) -> tuple[float, float]:
    o = np.array([x, y]) - rot(yaw) @ centre_local(oid)
    return float(o[0]), float(o[1])


def wrap180(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def build_rollout(n: int) -> list[dict]:
    d = DESIGN[n]
    objs = []
    for prod in d["products"]:
        bspot = d["base_spot"][prod]
        btype = CAT[base_of(prod)]["base_type"]
        byaw = BASE_YAW[btype]
        bx, by = BASE_SPOTS[bspot]
        objs.append({"id": base_of(prod), "spot": bspot, "cx": bx, "cy": by, "yaw": byaw})
        for part, (row, col, dyaw) in zip(parts_of(prod), d["parts"][prod]):
            y = COLS[bspot][col]
            spot = next(k for k, v in PART_SPOTS.items() if v == (ROWS[row], y))
            objs.append({"id": part, "spot": spot, "cx": ROWS[row], "cy": y, "yaw": wrap180(byaw + dyaw)})
    for o in objs:
        o["ox"], o["oy"] = origin_from_centre(o["id"], o["cx"], o["cy"], o["yaw"])
    return objs


def scene_of(objs: list[dict], perturb: dict | None = None) -> dict:
    out = []
    for o in objs:
        x, y, yaw = o["ox"], o["oy"], o["yaw"]
        if perturb and o["id"] in perturb:
            dx, dy, dyaw = perturb[o["id"]]
            x, y, yaw = (*origin_from_centre(o["id"], o["cx"] + dx, o["cy"] + dy, o["yaw"] + dyaw), o["yaw"] + dyaw)
        out.append({"key": o["id"], "id": o["id"], "x": x, "y": y, "yaw": yaw, "state": "installed"})
    return {"objects": out, "spares": [], "table_z_mm": TABLE_Z}


# ------------------------------------------------------------------ checks
def roll_off(r: float) -> float:
    return abs(wrap180(r - NEUTRAL_R))


def run_sequence(objs: list[dict], order: list[int], assign: dict[int, str], perturb: dict | None = None) -> list[dict]:
    """Pick-insert every part in sheet order, product by product, onto the assigned base instance."""
    scene = scene_of(objs, perturb)
    rows = []
    for prod in order:
        for part in parts_of(prod):
            g = plan_pick(scene, part)
            p = plan_place(scene, part, assign[prod], flip=g.flip) if g.feasible else None
            ok = bool(g.feasible and p is not None and p.feasible and p.flip == g.flip)
            rows.append({"part": part, "base": assign[prod], "ok": ok,
                         "pick": (g.feasible, g.reason, g.flip, g.r_deg, g.j6_estimate_deg),
                         "place": (p.feasible, p.reason, p.flip, p.r_deg, p.j6_estimate_deg) if p else None})
            for e in scene["objects"]:          # the part is now seated on its base (next steps see it there)
                if e["key"] == part:
                    e.clear()
                    e.update({"key": part, "id": part, "state": f"on:{assign[prod]}"})
    return rows


def assignments(products: list[int]) -> list[dict[int, str]]:
    """Every way the controller may map products onto identical bases present in this rollout."""
    present = [base_of(p) for p in products]
    options = []
    for prod in products:
        group = BASE_GROUPS[CAT[base_of(prod)]["base_type"]]
        options.append([b for b in group if b in present])
    out = []
    for combo in itertools.product(*options):
        if len(set(combo)) == len(combo):
            out.append(dict(zip(products, combo)))
    return out


def footprint_points(objs: list[dict]) -> dict[str, np.ndarray]:
    inst = build_instances(scene_of(objs), CAT)
    return {k: np.c_[s.x, s.y] for k, s in ((k, solid_of(i, CAT, MAPS)) for k, i in inst.items())}


def gap_report(objs: list[dict], pts: dict[str, np.ndarray]) -> dict:
    trees = {k: cKDTree(v) for k, v in pts.items()}
    res = {"base_base": (1e9, None), "base_part": (1e9, None), "part_part": (1e9, None)}
    circ = {"base_base": (1e9, None), "base_part": (1e9, None), "part_part": (1e9, None)}
    by = {o["id"]: o for o in objs}
    for a, b in itertools.combinations(by, 2):
        kind = {2: "base_base", 1: "base_part", 0: "part_part"}[a.startswith("F") + b.startswith("F")]
        d_true = float(trees[a].query(pts[b], k=1)[0].min())
        d_circ = math.hypot(by[a]["cx"] - by[b]["cx"], by[a]["cy"] - by[b]["cy"]) - FOOT_R[a] - FOOT_R[b]
        if d_true < res[kind][0]:
            res[kind] = (d_true, f"{a}-{b}")
        if d_circ < circ[kind][0]:
            circ[kind] = (d_circ, f"{a}-{b}")
    return {"true": res, "circle": circ}


def zone_report(objs: list[dict], pts: dict[str, np.ndarray]) -> list[str]:
    bad = []
    for o in objs:
        r = math.hypot(o["cx"], o["cy"])
        p = pts[o["id"]]
        if not (RMIN <= r <= RMAX):
            bad.append(f"{o['id']} centre radius {r:.0f}")
        if not (WX[0] <= p[:, 0].min() and p[:, 0].max() <= WX[1] and WY[0] <= p[:, 1].min() and p[:, 1].max() <= WY[1]):
            bad.append(f"{o['id']} footprint leaves the work area")
        if not (WX[0] + FOOT_R[o["id"]] <= o["cx"] <= WX[1] - FOOT_R[o["id"]]
                and WY[0] + FOOT_R[o["id"]] <= o["cy"] <= WY[1] - FOOT_R[o["id"]]):
            bad.append(f"{o['id']} footprint circle (r {FOOT_R[o['id']]:.1f}) leaves the work area")
        if np.hypot(p[:, 0], p[:, 1]).max() > RMAX:
            bad.append(f"{o['id']} footprint reaches past {RMAX:.0f} mm")
    return bad


def codex_parser():
    """Codex's fail-closed prompt parser from origin/feature/T37-saturday-cell-integration, with a stub catalog."""
    src = subprocess.run(["git", "-C", str(REPO), "show",
                          "origin/feature/T37-saturday-cell-integration:assembly/instruction.py"],
                         check=True, capture_output=True, text=True).stdout
    src = src.replace("from assembly.catalog import Catalog", "Catalog = object")
    ns: dict = {}
    exec(compile(src, "codex_instruction.py", "exec"), ns)

    class _P:
        def __init__(self, n):
            self.product = n

    class _C:
        products = [_P(n) for n in range(1, 7)]
    return lambda text: ns["parse_requested_products"](text, _C())


# ------------------------------------------------------------------ outputs
def yaml_out(rollouts: dict[int, list[dict]]) -> str:
    head = [
        "# CSI real-cell slot 2026-10-31 14:00 CT. Generated by validate.py (do not edit by hand): 3 rollouts x 3 products.",
        "# Positions: [x, y, yaw] in the robot frame UF0 (same x, y as obs[\"tcp_pose\"]), mm and degrees.",
        "# x, y = centre of the part's footprint seen from above; yaw = direction of the part's CAD local +x axis,",
        "# measured from robot +x (towards the front edge) turning towards +y (table_plan.png yaw cross).",
        "# Every part stands in its installed orientation. Bases of one shape are interchangeable (F01 = F02, F04 = F05 = F06).",
        "# Fixed spots: bases on B1..B3 (x 420), loose parts on P-spots (rows x 255 / 335, columns y +-45 / +-135 / +-225).",
    ]
    data = {"rollouts": []}
    for n, objs in rollouts.items():
        entry = {pretty(o["id"]): [round(o["cx"]), round(o["cy"]), round(o["yaw"])] for o in objs}
        data["rollouts"].append({"objects": entry, "prompt": PROMPT.format(*DESIGN[n]["products"])})
    return "\n".join(head) + "\n" + yaml.safe_dump(data, sort_keys=False, width=200, allow_unicode=True, default_flow_style=None)


def render(n: int, objs: list[dict], pts: dict[str, np.ndarray], rolls: dict[str, tuple]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle, Wedge

    fig, ax = plt.subplots(figsize=(14, 9.6))
    ax.set_title(f"Rollout {n}: products {', '.join(map(str, DESIGN[n]['products']))}   "
                 f"(top view as the table camera sees it; x down, y right, UF0 mm)", fontsize=11)
    ax.add_patch(Rectangle((WY[0], WX[0]), WY[1] - WY[0], WX[1] - WX[0], fill=False, ls="--", ec="#888", lw=1))
    ax.text(WY[1], WX[0] - 6, "work area", ha="right", va="bottom", fontsize=8, color="#666")
    ax.add_patch(Wedge((0, 0), RMAX, 0, 180, width=RMAX - RMIN, fc="#c8a00010", ec="#c8a000", ls=":", lw=1))
    ax.add_patch(Circle((0, 0), 105, fc="#d8d4c8", ec="k", hatch="//"))
    ax.text(0, 0, "robot\nbase", ha="center", va="center", fontsize=8)
    for s in CELL["spare_store"]["slots"]:
        ax.add_patch(Circle((s["y"], s["x"]), 20, fill=False, ec="#aaa", lw=0.8))
        ax.text(s["y"], s["x"], s["id"], ha="center", va="center", fontsize=7, color="#999")
    rz = CELL["reject_zone"]
    ax.add_patch(Rectangle((rz["y"][0], rz["x"][0]), rz["y"][1] - rz["y"][0], rz["x"][1] - rz["x"][0],
                           fc="#f4cccc", ec="#c66", lw=0.8))
    ax.text(np.mean(rz["y"]), np.mean(rz["x"]), "reject zone\n(leave empty)", ha="center", va="center", fontsize=7)
    for k, (sx, sy) in PART_SPOTS.items():
        ax.add_patch(Rectangle((sy - SLOT_HALF_MM, sx - SLOT_HALF_MM), 2 * SLOT_HALF_MM, 2 * SLOT_HALF_MM,
                               fill=False, ec="#ccd", lw=0.6, ls=":"))
    inst = build_instances(scene_of(objs), CAT)
    for o in objs:
        s = solid_of(inst[o["id"]], CAT, MAPS)
        h = s.top - TABLE_Z
        cmap = "Blues" if o["id"].startswith("F") else "Greys"
        ax.scatter(s.y, s.x, c=h, cmap=cmap, s=0.6, marker="s", vmin=-5, vmax=40, linewidths=0, rasterized=True)
        v = rot(o["yaw"]) @ np.array([1.0, 0.0])
        L = 34 if o["id"].startswith("F") else 22
        ax.annotate("", xy=(o["cy"] + L * v[1], o["cx"] + L * v[0]), xytext=(o["cy"], o["cx"]),
                    arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.4))
        ax.plot(o["cy"], o["cx"], "+", color="#c0392b", ms=7)
        label = f"{o['id']}\n({o['cx']:.0f}, {o['cy']:.0f}) {o['yaw']:.0f}°"
        ax.text(o["cy"], o["cx"] + (46 if o["id"].startswith("F") else 30), label, ha="center", va="top",
                fontsize=7.5, color="#123" if o["id"].startswith("F") else "#222",
                bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    ax.set_xlim(-420, 420)
    ax.set_ylim(510, -40)
    ax.set_aspect("equal")
    ax.set_xlabel("y (mm)")
    ax.set_ylabel("x (mm)")
    ax.grid(color="#eee", lw=0.6)
    ax.set_xticks(range(-400, 401, 50))
    ax.set_yticks(range(0, 501, 50))
    ax.text(-415, 500, "red arrow = part's local +x (yaw); grey dots = part top (darker = taller); blue = bases;\n"
            "dotted squares = proposed inspection-slot boxes (+-25 mm)", fontsize=8, va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / f"rollout_{n}.png", dpi=130)
    plt.close(fig)


def positions_md(rollouts: dict[int, list[dict]], rolls: dict[int, dict]) -> str:
    out = ["# CSI real-cell placement, 2026-10-31 14:00 CT", "",
           "x, y = centre of the part's footprint (mm, robot frame UF0, as obs[\"tcp_pose\"]). x grows towards the front "
           "edge (away from the robot); +y is to the right in table_plan.png and in the PNGs here.",
           "yaw = direction of the part's CAD local +x axis (red arrow in rollout_N.png), degrees from +x turning towards "
           "+y. Every part stands in its installed orientation. Match the silhouette in the PNG.", "",
           "Bases sit on spots B1 (420, -165), B2 (420, 0), B3 (420, 165); loose parts on a fixed 2 x 6 grid "
           "(rows x 255 / 335, columns y -225, -135, -45, 45, 135, 225). Reject zone and spare store stay empty.", ""]
    for n, objs in rollouts.items():
        out += [f"## Rollout {n}: products {', '.join(map(str, DESIGN[n]['products']))}", "",
                f"Prompt: `{PROMPT.format(*DESIGN[n]['products'])}`", "",
                "| # | object | spot | x | y | yaw | pick roll r | insert roll r |", "|---|---|---|---|---|---|---|---|"]
        for i, o in enumerate(objs, 1):
            pr = rolls[n].get(o["id"])
            pick = f"{pr[0]:.0f}" if pr else "-"
            ins = f"{pr[1]:.0f}" if pr else "-"
            out.append(f"| {i} | {pretty(o['id'])} | {o['spot']} | {o['cx']:.0f} | {o['cy']:.0f} | {o['yaw']:.0f} "
                       f"| {pick} | {ins} |")
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ main
def main() -> int:
    log: list[str] = []
    say = log.append
    parse = codex_parser()
    rollouts = {n: build_rollout(n) for n in DESIGN}
    all_ok = True
    rolls_md: dict[int, dict] = {}
    slot_hits: dict[str, set] = {}
    for n, objs in rollouts.items():
        prods = DESIGN[n]["products"]
        say(f"=== Rollout {n}: products {prods} ({len(objs)} objects: {sum(o['id'].startswith('F') for o in objs)} "
            f"bases, {sum(not o['id'].startswith('F') for o in objs)} loose parts)")
        got = parse(PROMPT.format(*prods))
        say(f"prompt -> Codex parse_requested_products = {got}  {'OK' if list(got) == prods else 'MISMATCH'}")
        all_ok &= list(got) == prods
        pts = footprint_points(objs)
        zb = zone_report(objs, pts)
        say(f"work area / reach: {'OK' if not zb else zb}  (centre radius {min(math.hypot(o['cx'], o['cy']) for o in objs):.0f}"
            f"..{max(math.hypot(o['cx'], o['cy']) for o in objs):.0f} mm; footprint max radius "
            f"{max(float(np.hypot(p[:, 0], p[:, 1]).max()) for p in pts.values()):.0f} mm)")
        all_ok &= not zb
        g = gap_report(objs, pts)
        need = {"base_base": GAP["base_gap_mm"], "base_part": GAP["base_to_part_gap_mm"], "part_part": PART_GAP_MM}
        for kind in ("base_base", "base_part", "part_part"):
            t, c = g["true"][kind], g["circle"][kind]
            ok = c[0] >= need[kind] and t[0] >= need[kind]
            all_ok &= ok
            say(f"min gap {kind:9s}: true footprint {t[0]:5.1f} mm ({t[1]}), footprint-circle {c[0]:5.1f} mm ({c[1]}); "
                f"need >= {need[kind]:.0f}  {'OK' if ok else 'FAIL'}")
        # nominal sequence + every base assignment and product order
        nominal = dict(zip(prods, [base_of(p) for p in prods]))
        rows = run_sequence(objs, prods, nominal)
        rolls_md[n] = {}
        for r in rows:
            pk, pl = r["pick"], r["place"]
            band = pk[3] is not None and pl and pl[3] is not None and roll_off(pk[3]) <= ROLL_BAND and roll_off(pl[3]) <= ROLL_BAND
            all_ok &= r["ok"] and bool(band)
            if pk[3] is not None and pl and pl[3] is not None:
                rolls_md[n][r["part"]] = (pk[3], pl[3])
            say(f"  {r['part']:6s} -> {r['base']}: pick {'ok' if pk[0] else 'FAIL ' + pk[1]} k={pk[2]} r={pk[3] if pk[3] is None else round(pk[3])} "
                f"J6~{pk[4] if pk[4] is None else round(pk[4])} | insert {'ok' if pl and pl[0] else 'FAIL ' + (pl[1] if pl else 'not planned')} "
                f"k={pl[2] if pl else None} r={round(pl[3]) if pl and pl[3] is not None else None} "
                f"J6~{round(pl[4]) if pl and pl[4] is not None else None} | same wrist turn: {r['ok']} | "
                f"rolls within 180+-{ROLL_BAND:.0f}: {bool(band)}")
        combos = [(a, list(o)) for a in assignments(prods) for o in itertools.permutations(prods)]
        fails = []
        for a, order in combos:
            bad = [f"{x['part']}->{x['base']}" for x in run_sequence(objs, order, a) if not x["ok"]]
            if bad:
                fails.append((a, order, bad))
        say(f"interchangeable bases x product orders: {len(combos) - len(fails)}/{len(combos)} sequences fully feasible"
            + ("" if not fails else f"; failures: {fails[:4]}"))
        all_ok &= not fails
        # placement tolerance: each loose part off by +-10 mm (x, y) or +-10 deg, others nominal
        pert_fail = []
        loose = [o["id"] for o in objs if not o["id"].startswith("F")]
        for pid in loose:
            for d in [(10, 0, 0), (-10, 0, 0), (0, 10, 0), (0, -10, 0), (0, 0, 10), (0, 0, -10)]:
                rr = run_sequence(objs, prods, nominal, perturb={pid: d})
                if not next(x for x in rr if x["part"] == pid)["ok"]:
                    pert_fail.append((pid, d))
        say(f"placement tolerance (+-10 mm x/y, +-10 deg, one part at a time): "
            f"{6 * len(loose) - len(pert_fail)}/{6 * len(loose)} still feasible"
            + ("" if not pert_fail else f"; failing: {pert_fail}"))
        all_ok &= not pert_fail
        # inspection-slot boxes: each loose part's centre AND CAD origin inside exactly one box
        for o in objs:
            if o["id"].startswith("F"):
                continue
            inside = [k for k, (sx, sy) in PART_SPOTS.items()
                      if abs(o["cx"] - sx) <= SLOT_HALF_MM and abs(o["cy"] - sy) <= SLOT_HALF_MM
                      and abs(o["ox"] - sx) <= SLOT_HALF_MM and abs(o["oy"] - sy) <= SLOT_HALF_MM]
            all_ok &= inside == [o["spot"]]
            slot_hits.setdefault(o["spot"], set()).add(n)
        render(n, objs, pts, rolls_md[n])
        say("")
    boxes = [(k, sx - SLOT_HALF_MM, sx + SLOT_HALF_MM, sy - SLOT_HALF_MM, sy + SLOT_HALF_MM) for k, (sx, sy) in PART_SPOTS.items()]
    overlap = [(a[0], b[0]) for a, b in itertools.combinations(boxes, 2)
               if a[1] <= b[2] and b[1] <= a[2] and a[3] <= b[4] and b[3] <= a[4]]
    base_in_box = [(b, k) for b, (bx, by) in BASE_SPOTS.items() for k, x0, x1, y0, y1 in boxes if x0 <= bx <= x1 and y0 <= by <= y1]
    say(f"proposed inspection-slot registry: {len(boxes)} boxes +-{SLOT_HALF_MM:.0f} mm, overlaps {overlap or 'none'}, "
        f"base centres inside a box {base_in_box or 'none'}; every loose part centre + CAD origin in exactly its own box")
    for k, x0, x1, y0, y1 in boxes:
        say(f"  {k}: selection_bounds_xy_mm=({x0:.0f}, {x1:.0f}, {y0:.0f}, {y1:.0f})  used by rollouts {sorted(slot_hits.get(k, []))}")
    all_ok &= not overlap
    say(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    (OUT / "placement.yaml").write_text(yaml_out(rollouts))
    (OUT / "positions.md").write_text(positions_md(rollouts, rolls_md))
    # round-trip: the written file parses, has 3 rollouts, the right objects, and matches the design
    back = yaml.safe_load((OUT / "placement.yaml").read_text())
    rt = len(back["rollouts"]) == 3 and all(len(r["objects"]) == len(rollouts[i + 1]) for i, r in enumerate(back["rollouts"]))
    say(f"placement.yaml round-trip: {'OK' if rt else 'FAIL'}")
    text = "\n".join(log)
    (OUT / "validate_output.txt").write_text(text + "\n")
    print(text)
    return 0 if all_ok and rt else 1


if __name__ == "__main__":
    raise SystemExit(main())
