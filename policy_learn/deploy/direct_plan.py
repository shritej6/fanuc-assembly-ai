"""Fallback plan: every pick-insert step of one rollout from placement.yaml, without pose perception.
Authored by Shritej Gunda."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULT_PLACEMENT = REPO / "robot" / "cell" / "submission" / "placement.yaml"
NEUTRAL_R = 180.0  # tools/placement/validate.py: the planner's reference roll for the fallback placement
_PRODUCTS_RE = re.compile(r"products?\s+([\d,\sand]+?)\s+on", re.IGNORECASE)


class PlacementError(ValueError):
    """Raised when placement.yaml lacks a rollout, object or prompt the runtime needs."""


@dataclass(frozen=True)
class Rollout:
    """One rollout read from placement.yaml."""
    index: int                                  # 0-based, the order in placement.yaml
    prompt: str
    products: tuple[int, ...]                   # prompt order: the build order
    objects: dict[str, tuple[float, float, float]]   # id -> footprint centre x, y (mm), local +x yaw (deg)


@dataclass(frozen=True)
class Step:
    """One planned pick-insert step."""
    product: int
    object_id: str                              # e.g. M5_01
    base_id: str                                # e.g. F05
    plan_target: tuple[float, ...]              # mimic/instructions.py: part x, y, yaw, roll, base x, y, yaw
    grasp_tcp: tuple[float, ...]                # [x, y, z, w, p, r] fingers closed on the pad
    seat_tcp: tuple[float, ...]                 # [x, y, z, w, p, r] part fully seated (CAD)
    capture_mm: float                           # the insert's xy capture (objects.yaml xy_capture_mm)
    contact_z: float                            # TCP z when the part first engages its mate (seat z + engagement)
    grasp_xy_tol_mm: float                      # the open fingers' slack around the pad: (16.5 - pad width) / 2
    grasp_z_tol_mm: float                       # half the pad's contact height: the fingers still land on the pad

    @property
    def label(self) -> str:
        """Return a short label for this step."""
        return f"{self.object_id} on {self.base_id}"


def _object_id(key: str) -> str:
    return str(key).split()[0]


def _products(prompt: str) -> tuple[int, ...]:
    m = _PRODUCTS_RE.search(prompt)
    if not m:
        raise PlacementError(f"cannot read the product order from the prompt {prompt!r}")
    return tuple(int(d) for d in re.findall(r"\d", m.group(1)))


def load_placement(path: str | Path | None = None) -> list[Rollout]:
    """Load and return the placement file."""
    path = Path(path or DEFAULT_PLACEMENT)
    data = yaml.safe_load(path.read_text()) or {}
    entries = data.get("rollouts") or []
    if not entries:
        raise PlacementError(f"{path} has no rollouts")
    out = []
    for i, entry in enumerate(entries):
        prompt = str(entry.get("prompt") or "")
        objects = {_object_id(k): tuple(float(v) for v in pos) for k, pos in (entry.get("objects") or {}).items()}
        out.append(Rollout(i, prompt, _products(prompt), objects))
    return out


def _centre_to_origin(oid: str, catalog: dict, cx: float, cy: float, yaw: float) -> tuple[float, float]:
    """Return the CAD local origin for a footprint centre."""
    if oid.startswith("F"):
        return cx, cy  # validate.py places bases by their spot centre directly
    b = catalog[oid]["bbox_mm"]
    lx, ly = (b["min"][0] + b["max"][0]) / 2.0, (b["min"][1] + b["max"][1]) / 2.0
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    return cx - (c * lx - s * ly), cy - (s * lx + c * ly)


def _scene(rollout: Rollout, catalog: dict, table_z: float) -> tuple[dict, dict]:
    objs, origins = [], {}
    for oid, (cx, cy, yaw) in rollout.objects.items():
        ox, oy = _centre_to_origin(oid, catalog, cx, cy, yaw)
        origins[oid] = (ox, oy, yaw)
        objs.append({"key": oid, "id": oid, "x": ox, "y": oy, "yaw": yaw, "state": "installed"})
    return {"objects": objs, "spares": [], "table_z_mm": table_z}, origins


def _parts_of(product: int, catalog: dict) -> list[str]:
    return sorted((k for k, e in catalog.items() if e.get("product") == product and e["sheet_order"] > 0),
                  key=lambda k: catalog[k]["sheet_order"])


def _base_of(product: int, rollout: Rollout) -> str:
    from sim.assembly_lab.eval.cad import base_group, base_type

    own = base_type(f"M{product}")
    on_table = [b for b in (own, *base_group(own)) if b in rollout.objects]
    if not on_table:
        raise PlacementError(f"rollout {rollout.index + 1}: no base for product {product} in placement.yaml")
    return on_table[0]


def _seat_in_scene(scene: dict, part: str, base: str) -> None:
    for e in scene["objects"]:
        if e["key"] == part:
            e.clear()
            e.update({"key": part, "id": part, "state": f"on:{base}"})


def plan_rollout(rollout: Rollout) -> list[Step]:
    """Return every pick-insert step of the rollout in build order."""
    from sim.assembly_lab.env.layout import LayoutSolver
    from sim.assembly_lab.eval.cad import matrix_to_pose
    from sim.assembly_lab.objects.planner import OPEN_GAP_MM, load_catalog, plan_pick, plan_place

    catalog, _ = load_catalog()
    solver = LayoutSolver()
    scene, origins = _scene(rollout, catalog, solver.table_z)
    stl = {oid: matrix_to_pose(solver.table_tf(oid, *origins[oid], "installed")) for oid in origins}
    steps = []
    for product in rollout.products:
        base = _base_of(product, rollout)
        for part in _parts_of(product, catalog):
            if part not in rollout.objects:
                raise PlacementError(f"rollout {rollout.index + 1}: {part} is not in placement.yaml")
            g = plan_pick(scene, part, current_r=NEUTRAL_R)
            p = plan_place(scene, part, base, flip=g.flip, current_r=g.r_deg or NEUTRAL_R) if g.feasible else None
            if not (g.feasible and p is not None and p.feasible):
                why = g.reason if not g.feasible else (p.reason if p else "no place plan")
                raise PlacementError(f"rollout {rollout.index + 1}: the planner refuses {part} on {base}: {why}")
            target = (*stl[part][[0, 1, 5]], g.r_deg, *stl[base][[0, 1, 5]])
            pad = catalog[part]["grips"][0]
            steps.append(Step(product, part, base, tuple(float(v) for v in target), tuple(g.tcp), tuple(p.tcp),
                              float(p.capture_mm or 0.0), float(p.contact_z),
                              (OPEN_GAP_MM - float(pad["pad_width_mm"])) / 2.0, float(pad["contact_height_mm"]) / 2.0))
            _seat_in_scene(scene, part, base)
    return steps
