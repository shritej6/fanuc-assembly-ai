"""Scenario -> explicit unseen layouts (INTERFACES.md layout dicts) plus truth labels and expectations (numpy only).

Each layout is reproducible from (seed_base, scenario index, layout index). Every entry names its pool `uid`
(loose part = type id, spare = type id + "#2"), so the expectations can name objects exactly.
The extra key `eval` is ignored by the env (env/layout.py reads objects / spares / defects only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import cad

HERE = Path(__file__).resolve().parent
LOOKALIKES = {  # PLAN section 3.2: same shape within ~1-1.5 mm
    "M4_04": ["M5_04", "M6_04"], "M5_04": ["M4_04", "M6_04"], "M6_04": ["M4_04", "M5_04"],
    "M4_03": ["M5_03", "M6_03"], "M5_03": ["M4_03", "M6_03"], "M6_03": ["M4_03", "M5_03"],
    "M1_02": ["M2_02"], "M2_02": ["M1_02"],
}
SPARE_SLOTS = ("S1", "S2", "S3", "S4", "S5", "S6")
CASES = ("nominal", "foreign", "damaged", "fallen", "unsure", "lookalike", "missing", "crowded", "wont_seat",
         "dropped", "fix_reverse", "final_check", "intrusion")


class ScenarioError(ValueError):
    """scenarios.yaml is inconsistent (unknown case, seed overlap, impossible layout)."""


@dataclass
class Scenario:
    index: int
    id: str
    case: str
    products: list[str]
    layouts: int
    time_limit_s: float
    requires: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)


@dataclass
class EvalLayout:
    scenario: str
    case: str
    seed: int
    layout: dict      # passed to env.reset (objects, spares, defects, eval)
    expect: dict      # see expectations() below


def load(path: Path = HERE / "scenarios.yaml") -> tuple[dict, list[Scenario]]:
    raw = yaml.safe_load(Path(path).read_text())
    base, n = int(raw["seed_base"]), int(raw["layouts_per_scenario"])
    if base <= int(raw["mimic_seed"]) < base + 100 * (len(raw["scenarios"]) + 1):
        raise ScenarioError(f"mimic seed {raw['mimic_seed']} falls inside the eval seed range from {base}")
    out, seen = [], set()
    for i, s in enumerate(raw["scenarios"]):
        if s["case"] not in CASES:
            raise ScenarioError(f"{s['id']}: unknown case {s['case']!r}")
        if s["id"] in seen:
            raise ScenarioError(f"duplicate scenario id {s['id']}")
        seen.add(s["id"])
        bad = [p for p in s["products"] if p not in cad.PRODUCTS]
        if bad:
            raise ScenarioError(f"{s['id']}: unknown products {bad}")
        opts = {k: v for k, v in s.items() if k not in ("id", "case", "products", "layouts", "time_limit_s",
                                                         "requires")}
        out.append(Scenario(i, s["id"], s["case"], list(s["products"]), int(s.get("layouts", n)),
                            float(s.get("time_limit_s", raw["time_limit_s"])), list(s.get("requires", [])), opts))
    return raw, out


class _Placer:
    """Rejection-samples non-overlapping footprints inside the work area and the reach ring."""

    def __init__(self, rng: np.random.Generator, gap_mm: float):
        c = cad.cell_layout()
        self.rng, self.gap = rng, gap_mm
        self.area, self.rmin, self.rmax = c["work_area"], c["reach_radius_min"], c["reach_radius_max"]
        self.taken: list[tuple[float, float, float]] = []
        self.where: dict[str, tuple[float, float, float]] = {}

    @staticmethod
    def radius(tid: str) -> float:
        return float(cad.geometry()[cad.type_id(tid)]["upright"]["footprint_r"])

    def free(self, x: float, y: float, r: float, gap: float) -> bool:
        if not (self.rmin <= math.hypot(x, y) <= self.rmax):
            return False
        return all(math.hypot(x - tx, y - ty) >= r + tr + gap for tx, ty, tr in self.taken)

    def place(self, tid: str, gap: float | None = None) -> tuple[float, float, float]:
        r, g = self.radius(tid), self.gap if gap is None else gap
        if tid.startswith("F"):
            g = max(g, cad.cell_layout()["clearance"]["base_gap_mm"])
        for _ in range(4000):
            x = self.rng.uniform(self.area["x"][0] + r, self.area["x"][1] - r)
            y = self.rng.uniform(self.area["y"][0] + r, self.area["y"][1] - r)
            if self.free(x, y, r, g):
                self.taken.append((x, y, r))
                self.where[tid] = (x, y, r)
                return round(x, 1), round(y, 1), round(float(self.rng.uniform(-180.0, 180.0)), 1)
        raise ScenarioError(f"no free spot for {tid} after 4000 tries")

    def next_to(self, tid: str, other: tuple[float, float, float], touch_gap: float) -> tuple[float, float, float]:
        """Place tid just touch_gap away from another footprint (a crowded neighbour), any free direction."""
        r = self.radius(tid)
        ox, oy, orad = other
        for _ in range(720):
            a = self.rng.uniform(0, 2 * math.pi)
            d = r + orad + touch_gap
            x, y = ox + d * math.cos(a), oy + d * math.sin(a)
            inside = (self.area["x"][0] + r <= x <= self.area["x"][1] - r
                      and self.area["y"][0] + r <= y <= self.area["y"][1] - r)
            if inside and all(math.hypot(x - tx, y - ty) >= r + tr + touch_gap - 1e-6 for tx, ty, tr in self.taken):
                self.taken.append((x, y, r))
                return round(x, 1), round(y, 1), round(float(self.rng.uniform(-180.0, 180.0)), 1)
        raise ScenarioError(f"no crowded spot for {tid}")


def build(raw: dict, sc: Scenario, k: int) -> EvalLayout:
    seed = int(raw["seed_base"]) + 100 * sc.index + k
    rng = np.random.default_rng(seed)
    gap = float(sc.options.get("part_gap_mm", raw["part_gap_mm"]))
    placer = _Placer(rng, gap)
    lay: dict = {"objects": [], "spares": [], "defects": [], "eval": {"labels": {}, "events": []}}
    labels = lay["eval"]["labels"]
    bases = {}
    for prod in sc.products:  # a free base of the product's group (identical bases are interchangeable)
        group = [b for b in cad.base_group(cad.base_type(prod)) if b not in bases.values()]
        bid = str(rng.choice(group))
        bases[prod] = bid
        x, y, yaw = placer.place(bid)
        lay["objects"].append({"id": bid, "uid": bid, "x": x, "y": y, "yaw": yaw, "state": "installed"})
    loose = [p for prod in sc.products for p in cad.product_parts(prod)]
    case_parts = _case_setup(sc, rng, loose, bases, lay, labels)
    for pid in loose:
        if pid in case_parts["skip"]:
            continue
        x, y, yaw = placer.place(pid)
        entry = {"id": pid, "uid": pid, "x": x, "y": y, "yaw": yaw,
                 "state": "fallen" if pid in labels.get("fallen", []) else "installed"}
        lay["objects"].append(entry)
    for pid in case_parts["decoys"]:
        x, y, yaw = placer.place(pid)
        lay["objects"].append({"id": pid, "uid": pid, "x": x, "y": y, "yaw": yaw, "state": "installed"})
    if sc.case == "crowded":
        entry = None
        for blocker in labels.pop("blocker_candidates"):  # the first one that truly crowds the target
            try:
                entry = _crowd(placer, lay, labels["crowded_target"], blocker)
                break
            except ScenarioError:
                continue
        if entry is None:
            raise ScenarioError(f"{sc.id} seed {seed}: no part crowds {labels['crowded_target']}")
        labels["blocker"] = entry["uid"]
        lay["objects"].append(entry)
    for slot, pid in zip(SPARE_SLOTS, case_parts["spares"]):
        lay["spares"].append({"id": pid, "uid": f"{pid}#2", "slot": slot, "yaw": 0.0, "state": "installed"})
    for req in sc.requires:
        if req.startswith("env:"):
            lay["eval"]["events"].append({"type": req[4:], "case": sc.case})
    return EvalLayout(sc.id, sc.case, seed, lay, expectations(sc, bases, labels, case_parts))


def _overlaps(scene: dict, key: str, margin_mm: float = 0.5) -> bool:
    """True if `key`'s footprint cells (agent 2's height maps) come within ~margin of any other object's cells."""
    from ..objects.planner import load_catalog
    from ..objects.scene import build_instances, solid_of

    catalog, maps = load_catalog()
    inst = build_instances(scene, catalog)
    mine = solid_of(inst[key], catalog, maps)
    q = 1.0 / margin_mm
    ix, iy = np.round(mine.x * q).astype(np.int64), np.round(mine.y * q).astype(np.int64)
    dil = {int(a + dx) * 1_000_003 + int(b + dy) for a, b in zip(ix, iy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    keys = np.fromiter(dil, dtype=np.int64)
    reach = float(np.hypot(mine.x - mine.x.mean(), mine.y - mine.y.mean()).max()) + 60.0
    for k, i in inst.items():
        if k == key or math.hypot(i.x - mine.x.mean(), i.y - mine.y.mean()) > reach:
            continue
        other = solid_of(i, catalog, maps)
        ok = np.round(other.x * q).astype(np.int64) * 1_000_003 + np.round(other.y * q).astype(np.int64)
        if np.isin(ok, keys).any():
            return True
    return False


def _crowd(placer: _Placer, lay: dict, target: str, blocker: str) -> dict:
    """Put the blocker where agent 2's finger model (objects/planner.py blockers()) says the target cannot be gripped
    at any wrist turn until the blocker moves (a genuinely crowded part), without touching any object's real shape."""
    from ..objects.planner import blockers

    tx, ty, tr = placer.where[target]
    br = placer.radius(blocker)
    for _ in range(150):
        d = float(placer.rng.uniform(0.3, 1.0)) * (tr + br)
        a = float(placer.rng.uniform(0.0, 2.0 * math.pi))
        entry = {"id": blocker, "uid": blocker, "x": round(tx + d * math.cos(a), 1),
                 "y": round(ty + d * math.sin(a), 1), "yaw": round(float(placer.rng.uniform(-180, 180)), 1),
                 "state": "installed"}
        scene = {"objects": [dict(o, key=o["uid"]) for o in lay["objects"] + [entry]]}
        if not cad.in_zone((entry["x"], entry["y"]), placer.area) or _overlaps(scene, blocker):
            continue
        if blockers(scene, target) == [blocker]:
            placer.taken.append((entry["x"], entry["y"], br))
            return entry
    raise ScenarioError(f"could not crowd {target} with {blocker}")


def _case_setup(sc: Scenario, rng, loose: list[str], bases: dict, lay: dict, labels: dict) -> dict:
    """Case-specific truth labels; returns parts to skip on the table, spares to stock, decoys to add."""
    out = {"skip": set(), "spares": [], "decoys": []}
    want_spare = bool(sc.options.get("spare", True))
    candidates = [p for p in loose if not p.endswith("_01")]  # housings stay: a product needs its first part
    pick = str(rng.choice(candidates))
    if sc.case in ("damaged", "fallen"):
        labels[sc.case] = [pick]
        out["spares"] += [pick] if want_spare else []
    elif sc.case == "missing":
        labels["missing"] = [pick]
        out["skip"].add(pick)
        out["spares"] += [pick] if want_spare else []
    elif sc.case == "lookalike":
        role = str(sc.options.get("decoy_role", "04"))
        target = f"{sc.products[0]}_{role}"
        decoy = str(rng.choice(LOOKALIKES[target]))
        labels["decoy"] = [decoy]
        out["decoys"].append(decoy)
    elif sc.case == "crowded":
        labels["crowded_target"] = pick
        others = [p for p in cad.grasp_frames() if p.startswith("M") and p[:2] not in sc.products]
        labels["blocker_candidates"] = [str(b) for b in rng.permutation(others)]
    elif sc.case == "fix_reverse":
        prod = sc.products[0]
        wrong = LOOKALIKES[f"{prod}_02"][0]
        base = bases[prod]
        lay["objects"] += [{"id": f"{prod}_01", "uid": f"{prod}_01", "state": f"on:{base}"},
                           {"id": wrong, "uid": wrong, "state": f"on:{base}"},
                           {"id": f"{prod}_03", "uid": f"{prod}_03", "state": f"on:{base}"}]
        labels["wrong_part"] = [wrong]
        out["skip"] |= {f"{prod}_01", f"{prod}_03"}
    return out


def expectations(sc: Scenario, bases: dict, labels: dict, case_parts: dict) -> dict:
    """What a correct run ends with, per docs/ROBOT_BRAIN.md section 3 (checked by kpis.final_checks)."""
    built = list(sc.products)
    exp: dict = {"built": {p: bases[p] for p in built}, "in_defect_area": [], "on_test_bed_upright": [],
                 "untouched": [], "not_installed": [], "andon": False, "events": []}
    if sc.case in ("damaged", "fallen"):
        bad = labels[sc.case][0]
        exp["not_installed"].append(bad)
        # damaged -> defect area; fallen -> rejected in place (objects/planner.py refuses to grip a fallen part)
        exp["in_defect_area" if sc.case == "damaged" else "untouched"].append(bad)
        if not case_parts["spares"]:
            exp["andon"] = True
            exp["built"].pop(bad[:2])
    elif sc.case == "missing":
        miss = labels["missing"][0]
        exp["build_order_last"] = miss[:2]          # the product with the missing part is finished last
        if not case_parts["spares"]:
            exp["andon"] = True
            exp["built"].pop(miss[:2])
    elif sc.case == "lookalike":
        exp["untouched"] += labels["decoy"]
        exp["not_installed"] += labels["decoy"]
    elif sc.case == "crowded":
        exp["on_test_bed_upright"].append(labels["blocker"])
        exp["move_before"] = [labels["blocker"], labels["crowded_target"]]
    elif sc.case == "fix_reverse":
        exp["not_installed"] += labels["wrong_part"]
        exp["on_test_bed_upright"] += labels["wrong_part"]   # a good part of another product: kept, not rejected
    elif sc.case == "unsure":
        exp["events"] = ["second_closeup", "andon"]
        exp["andon"] = True
    elif sc.case == "final_check":
        exp["events"] = ["final_closeup_1", "final_closeup_2"]
    elif sc.case == "intrusion":
        exp["events"] = ["freeze", "rescan", "replan"]
    elif sc.case == "foreign":
        exp["in_defect_area"].append("FOREIGN")
    return exp


def all_layouts(path: Path = HERE / "scenarios.yaml", only: list[str] | None = None) -> list[EvalLayout]:
    raw, scs = load(path)
    return [build(raw, sc, k) for sc in scs if not only or sc.id in only or sc.case in only
            for k in range(sc.layouts)]
