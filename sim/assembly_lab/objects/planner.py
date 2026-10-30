"""Grasp / place / pull-out planner for the 30 printed objects (pure Python + numpy, no Isaac).

    from sim.assembly_lab.objects.planner import plan_pick, plan_place, plan_pull_out, blockers
    g = plan_pick(scene, "M4_02", current_r=180.0)        # GraspPlan; g.feasible, g.path rows [x,y,z,w,p,r,grip]
    p = plan_place(scene, "M4_02", "F04", flip=g.flip)    # PlacePlan onto that base (any base of the right type)

Rules (docs/ROBOT_BRAIN.md, coord/DECISIONS.md 2026-10-29):
- Grip only the CAD pad (objects.yaml `grips`, from cad/grasp_frames.json). Wrist roll r = yaw + roll_offset + 180 k,
  k in {0, 1}; the one nearest the current r whose J6 stays inside the limits and whose fingers clear every
  neighbour. Fingers straight down: w, p = -180, 0.
- Crowded (no k clears): `blockers()` lists what to move first. Never push. Bases never move.
- Fallen parts are not picked for assembly (reject + spare); unknown pieces have no CAD grip.
- Build in sheet order; a part goes on only when every lower step is on that base and nothing above it is.
  Dismantle in reverse order, straight up.

J6: limits +-360 deg (sim/starter/scene_config.py JOINT_LIMITS_DEG, "EXACT (er4ia_physics.usd)"). With fingers straight
down and J4 = 0 the twin's chain (sim/starter/fk.py) gives r = J1 + J6 + 180 (mod 360), so J6 ~ r - J1 - 180,
continued from the current J6. This is the twin's kinematics; the real controller chooses its own turn numbers.
Finger envelope: Team V fingers (cad/fingers/README.md, the STEP-frame STLs in sim/assembly_lab/env/assets/fingers):
rubber gap 16.5 open, NBR pad 0.4 proud. The finger is NOT symmetric about the TCP. In the STEP frame (= TCP frame:
x = v, y = closing, z = along the fingers) the TCP is on the flange axis (X = 0) and the pad / contact centre is at
X = -1.8 mm; the tip 12 mm of the finger spans X -6.0 .. +2.4 (5.35 thick beyond the pad), the spine above it
X -6.0 .. +6.0 (7.8 thick). Team V's validated grasp frames (cad/fingers/product_grasp_frames.json) put the gripper
origin 1.8 mm along +X from the contact centre, and the env mounts the same STLs with TCP x = STEP X
(env/assets.py install_teamv_fingers), so all three agree. Two consequences, both measured in Isaac on 2026-10-30
(agent 3, run3): the commanded TCP is the CAD contact centre + 1.8 mm along v (else the pads land 1.8 mm off the
pad), and the finger body reaches 6 mm on the -v side, so with an edge pad one wrist turn presses the finger onto the
target part itself (M1_01 r - yaw = -90: 8/8 attempts, M4_01 r - yaw = 0: 5/5). finger_hits therefore checks the
target's own body beyond the pad faces as well as every neighbour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .scene import CELL_MM, Instance, Solid, build_instances, solid_of

HERE = Path(__file__).resolve().parent
OBJECTS_YAML = HERE / "objects.yaml"
HEIGHTMAPS = HERE / "heightmaps.npz"

OPEN_GAP_MM = 16.5
PAD_PROUD_MM = 0.4
CONTACT_V_MM = -1.8          # pad (contact) centre along TCP x (v) from the TCP: STEP contact centre X = -1.8
NECK = {"len": 12.0, "thick": 5.35, "v": (-6.0, 2.4)}    # tip 12 mm, v span about the TCP (STEP X)
SPINE = {"thick": 7.8, "v": (-6.0, 6.0)}                 # above the neck
FINGER_LEN_MM = 76.0
XY_MARGIN_MM = 1.0           # pose error allowance around every finger face
Z_MARGIN_MM = 0.5           # the validated M4_02 grasp has its fingertips 0.9 mm above the housing top
J6_LIMIT_DEG = 360.0
J6_MARGIN_DEG = 10.0
APPROACH_MM = 30.0           # finger project approach_height
TRANSIT_CLEAR_MM = 10.0
HOVER_MM = 5.0
SLOW_ZONE_MM = 5.0           # ROBOT_BRAIN: the last 5 mm is slow
GUARD_BELOW_SEAT_MM = 1.0    # ROBOT_BRAIN: no target lower than 1 mm below the seated height
WORKSPACE = {"x": (150.0, 650.0), "y": (-450.0, 450.0), "z": (-334.0, 300.0)}
W, P = -180.0, 0.0


@dataclass
class Plan:
    feasible: bool
    reason: str
    key: str
    oid: str
    flip: int | None = None                    # k: r = yaw + roll_offset + 180 k
    r_deg: float | None = None                 # continuous, nearest the current r
    j6_estimate_deg: float | None = None
    tcp: list[float] | None = None             # grasp (pick/pull-out) or seated (place) [x,y,z,w,p,r]
    path: list[list[float]] = field(default_factory=list)   # [x, y, z, w, p, r, gripper]
    blocked_by: dict[int, list[str]] = field(default_factory=dict)   # per k: colliding instance keys
    notes: list[str] = field(default_factory=list)


@dataclass
class GraspPlan(Plan):
    grip: dict | None = None
    transit_z: float | None = None


@dataclass
class PlacePlan(Plan):
    base: str | None = None
    target_pose: list[float] | None = None     # part origin [x, y, z, yaw] when seated
    hover_z: float | None = None               # TCP z: 5 mm above first contact
    contact_z: float | None = None             # TCP z when the part first engages its mate
    slow_from_z: float | None = None
    guard_floor_z: float | None = None         # never command the TCP below this
    insert_axis: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    capture_mm: float | None = None


@lru_cache(maxsize=1)
def load_catalog(path: str = str(OBJECTS_YAML), maps_path: str = str(HEIGHTMAPS)) -> tuple[dict, dict]:
    catalog = yaml.safe_load(Path(path).read_text())["objects"]
    data = np.load(maps_path)
    maps = {oid: {"origin": tuple(data[f"{oid}/origin"]), "res": float(data[f"{oid}/res"]),
                  "top": data[f"{oid}/top"].astype(float), "bottom": data[f"{oid}/bottom"].astype(float)}
            for oid in catalog}
    return catalog, maps


# --- wrist roll and J6 ---------------------------------------------------------

def j6_estimate(r: float, x: float, y: float, ref: tuple[float, float, float, float, float]) -> float:
    """J6 for TCP roll r at (x, y), continued from the reference (r, x, y, J6) of the current pose."""
    r0, x0, y0, j60 = ref[0], ref[1], ref[2], ref[3]
    j1, j10 = math.degrees(math.atan2(y, x)), math.degrees(math.atan2(y0, x0))
    return j60 + (r - r0) - (j1 - j10)


def roll_candidates(yaw: float, offset: float, current_r: float, flip: int | None = None) -> list[tuple[int, float]]:
    out = []
    for k in ((0, 1) if flip is None else (flip,)):
        r = yaw + offset + 180.0 * k
        out.append((k, r + 360.0 * round((current_r - r) / 360.0)))
    return sorted(out, key=lambda kr: abs(kr[1] - current_r))


def _ref(current_r: float, current_xy: tuple[float, float], current_j6: float | None) -> tuple:
    j6 = current_j6 if current_j6 is not None else current_r - math.degrees(math.atan2(current_xy[1], current_xy[0])) \
        - 180.0
    return (current_r, current_xy[0], current_xy[1], j6, 0.0)


# --- clearance -------------------------------------------------------------------

def v_dir(r: float) -> np.ndarray:
    """TCP x axis (the finger's v direction, STEP X) in the world for fingers straight down at roll r."""
    return np.array([math.cos(math.radians(r)), math.sin(math.radians(r))])


def tcp_for_contact(contact, r: float) -> np.ndarray:
    """Commanded TCP (flange axis, tip plane) that puts the pad centre on the CAD contact point at roll r."""
    out = np.array(contact, dtype=float)
    out[:2] -= CONTACT_V_MM * v_dir(r)
    return out


def finger_hits(solids: list[Solid], tcp: np.ndarray, r: float, pad_width: float, target: Solid | None = None) -> list[str]:
    """Instances that any finger region touches between the open approach and closing on a pad of this width.
    `tcp` is the commanded TCP (flange axis). The target's own body counts beyond its pad faces (cells farther than
    half the pad width from the closing centre line, where the closing fingers travel) at grid resolution: the CAD
    clearances there are 0.1-0.9 mm, so the 1 mm neighbour margin would refuse every validated grasp."""
    c_dir = np.array([math.sin(math.radians(r)), -math.cos(math.radians(r))])
    v = v_dir(r)
    m = XY_MARGIN_MM + CELL_MM * 0.71
    regions = [(pad_width / 2, OPEN_GAP_MM / 2 + PAD_PROUD_MM + NECK["thick"], NECK["v"], 0.0),
               (pad_width / 2, OPEN_GAP_MM / 2 + PAD_PROUD_MM + SPINE["thick"], SPINE["v"], NECK["len"])]
    hits = []
    for s, own in [(s, False) for s in solids] + ([(target, True)] if target is not None else []):
        dx, dy = s.x - tcp[0], s.y - tcp[1]
        dc = np.abs(dx * c_dir[0] + dy * c_dir[1])
        dv = dx * v[0] + dy * v[1]
        below_top = s.bottom < tcp[2] + FINGER_LEN_MM
        for c0, c1, (v0, v1), z0 in regions:
            if own:  # cells clear of the gripped pad faces whose centre lies more than half a cell inside the finger
                mv, mz = -CELL_MM / 2, 0.0   # (validated Team V grasps clear the part by 0.1-0.9 mm: no extra margin)
                inside = (dc >= c0 + CELL_MM) & (dc <= c1) & (dv >= v0 - mv) & (dv <= v1 + mv) & below_top
            else:
                mz = Z_MARGIN_MM
                inside = (dc >= c0 - m) & (dc <= c1 + m) & (dv >= v0 - m) & (dv <= v1 + m) & below_top
            if np.any(inside & (s.top > tcp[2] + z0 - mz)):
                hits.append(s.key)
                break
    return hits


def _grid(held: Solid, margin: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Held part's lowest z per world cell, dilated by `margin` (min over the neighbourhood)."""
    lo = np.array([held.x.min(), held.y.min()]) - margin - CELL_MM
    n = math.ceil(margin / CELL_MM)
    ix = np.floor((held.x - lo[0]) / CELL_MM).astype(int) + 0
    iy = np.floor((held.y - lo[1]) / CELL_MM).astype(int)
    shape = (ix.max() + n + 2, iy.max() + n + 2)
    low = np.full(shape, np.inf)
    np.minimum.at(low, (ix, iy), held.bottom)
    out = low.copy()
    for a in range(-n, n + 1):
        for b in range(-n, n + 1):
            out = np.fmin(out, np.roll(np.roll(low, a, 0), b, 1))
    return out, lo, n


def sweep_hits(held: Solid, others: list[Solid], margin: float = XY_MARGIN_MM) -> list[str]:
    """Instances whose top rises above the held part's underside anywhere under its footprint (straight-down
    sweep from above, or straight-up pull-out with `others` above it)."""
    grid, lo, _ = _grid(held, margin)
    hits = []
    for s in others:
        ix = np.floor((s.x - lo[0]) / CELL_MM).astype(int)
        iy = np.floor((s.y - lo[1]) / CELL_MM).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < grid.shape[0]) & (iy < grid.shape[1])
        if ok.any() and np.any(s.top[ok] > grid[ix[ok], iy[ok]] + Z_MARGIN_MM):
            hits.append(s.key)
    return hits


# --- helpers -------------------------------------------------------------------------

def _row(p, r: float, grip: int) -> list[float]:
    return [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3), W, P, round(r, 3), float(grip)]


def _in_workspace(p) -> bool:
    return all(WORKSPACE[a][0] <= v <= WORKSPACE[a][1] for a, v in zip("xyz", p))


def _stack(instances: dict[str, Instance], catalog: dict, base_key: str) -> dict[int, Instance]:
    return {catalog[i.oid]["sheet_order"]: i for i in instances.values() if i.base == base_key}


def _transit_z(solids: list[Solid], hang_mm: float) -> float:
    top = max((float(s.top.max()) for s in solids if s.top.size), default=-335.0)
    return top + TRANSIT_CLEAR_MM + hang_mm


def _choose(cands, inst_xy, ref, hits_for) -> tuple[int, float, float, dict[int, list[str]]]:
    """First candidate (nearest r first) whose J6 is inside the limits and whose fingers clear."""
    blocked = {}
    for k, r in cands:
        j6 = j6_estimate(r, inst_xy[0], inst_xy[1], ref)
        if abs(j6) > J6_LIMIT_DEG - J6_MARGIN_DEG:
            blocked[k] = ["J6_LIMIT"]
            continue
        hits = hits_for(r)
        if not hits:
            return k, r, j6, blocked
        blocked[k] = hits
    return -1, 0.0, 0.0, blocked


def _context(scene: dict, key: str, catalog, maps):
    if catalog is None:
        catalog, maps = load_catalog()
    instances = build_instances(scene, catalog)
    if key not in instances:
        raise KeyError(f"{key!r} is not in the scene")
    return catalog, maps, instances, instances[key]


def _refusal(cls, inst: Instance, catalog: dict, instances, base_key=None):
    if inst.state == "unknown":
        return cls(False, "unknown piece: no CAD grip pad", inst.key, inst.oid)
    if inst.state == "fallen":
        return cls(False, "fallen: reject and take a spare (the arm cannot reorient)", inst.key, inst.oid)
    if catalog[inst.oid]["sheet_order"] == 0:
        return cls(False, "bases never move", inst.key, inst.oid)
    if inst.base is not None:
        order = catalog[inst.oid]["sheet_order"]
        above = [i.key for o, i in sorted(_stack(instances, catalog, inst.base).items(), reverse=True) if o > order]
        if above:
            return cls(False, "parts above it on the base: remove them first (reverse sheet order)", inst.key,
                       inst.oid, blocked_by={0: above, 1: above})
    return None


# --- public API ---------------------------------------------------------------------

def plan_pick(scene: dict, key: str, current_r: float = 180.0, current_xy=(400.0, 0.0),
              current_j6: float | None = None, flip: int | None = None, catalog=None, maps=None) -> GraspPlan:
    """Grip the CAD pad of instance `key` (standing on the table, or the top part on a base)."""
    catalog, maps, instances, inst = _context(scene, key, catalog, maps)
    refused = _refusal(GraspPlan, inst, catalog, instances)
    if refused:
        return refused
    entry, grip = catalog[inst.oid], catalog[inst.oid]["grips"][0]
    contact = inst.local_to_world(grip["pad_point_local_mm"])
    others = [solid_of(i, catalog, maps) for k, i in instances.items() if k != key]
    own = solid_of(inst, catalog, maps)
    cands = roll_candidates(inst.yaw, grip["roll_offset_deg"], current_r, flip)
    k, r, j6, blocked = _choose(cands, contact, _ref(current_r, current_xy, current_j6),
                                lambda rr: finger_hits(others, tcp_for_contact(contact, rr), rr, grip["pad_width_mm"],
                                                       own))
    if k < 0:
        return GraspPlan(False, "crowded: no wrist roll clears the fingers; move blockers() first", key, inst.oid,
                         blocked_by=blocked, grip=grip)
    tcp = tcp_for_contact(contact, r)
    hang = grip["pad_point_local_mm"][2] - entry["bbox_mm"]["min"][2]
    transit = _transit_z(others + [solid_of(inst, catalog, maps)], hang)
    above = tcp + [0, 0, APPROACH_MM]
    path = [_row([tcp[0], tcp[1], max(transit, above[2])], r, 0), _row(above, r, 0), _row(tcp, r, 0),
            _row(tcp, r, 1), _row(above, r, 1), _row([tcp[0], tcp[1], max(transit, above[2])], r, 1)]
    plan = GraspPlan(True, "ok", key, inst.oid, k, r, j6, _row(tcp, r, 0)[:6], path, blocked, grip=grip,
                     transit_z=max(transit, above[2]))
    if not all(_in_workspace(p[:3]) for p in path):
        plan.feasible, plan.reason = False, "outside the workspace box (PLAN 3.1)"
    return plan


def _place_refusal(inst, base, catalog, instances) -> str | None:
    entry = catalog[inst.oid]
    if base is None or catalog.get(base.oid, {}).get("sheet_order") != 0:
        return "target is not a base"
    if catalog[base.oid]["base_type"] != entry["base_type"]:
        return f"wrong base type: {inst.oid} needs type {entry['base_type']}, {base.key} is {catalog[base.oid]['base_type']}"
    order = entry["sheet_order"]
    stack = {o: i for o, i in _stack(instances, catalog, base.key).items() if i.key != inst.key}
    missing = [o for o in range(1, order) if o not in stack]
    if missing:
        return f"steps {missing} are not on {base.key} yet (build in sheet order)"
    if any(o >= order for o in stack):
        return f"{[stack[o].key for o in sorted(stack) if o >= order]} already on {base.key}: remove first"
    if order >= 2 and catalog[stack[1].oid]["product"] != entry["product"]:
        return f"{base.key} carries {stack[1].oid}, which is not product {entry['product']}"
    return None


def plan_place(scene: dict, key: str, base_key: str, flip: int | None = None, current_r: float = 180.0,
               current_xy=(400.0, 0.0), current_j6: float | None = None, catalog=None, maps=None) -> PlacePlan:
    """Insert the held instance `key` onto base `base_key` (straight down along -Z, the CAD insert axis)."""
    catalog, maps, instances, inst = _context(scene, key, catalog, maps)
    base = instances.get(base_key)
    reason = _place_refusal(inst, base, catalog, instances)
    if reason:
        return PlacePlan(False, reason, key, inst.oid, base=base_key)
    entry, grip = catalog[inst.oid], catalog[inst.oid]["grips"][0]
    origin = base.local_to_world(entry["installed_origin_mm"])
    seated = Instance(key, inst.oid, origin[0], origin[1], origin[2], base.yaw, f"on:{base_key}", base_key)
    contact = seated.local_to_world(grip["pad_point_local_mm"])
    others = [solid_of(i, catalog, maps) for k, i in instances.items() if k != key]
    held = solid_of(seated, catalog, maps)
    foreign = [s for s in others if s.key != base_key and instances[s.key].base != base_key]
    sweep = sweep_hits(held, foreign)
    if sweep:
        return PlacePlan(False, "the part's way down is blocked", key, inst.oid, base=base_key,
                         blocked_by={0: sweep, 1: sweep})
    cands = roll_candidates(base.yaw, grip["roll_offset_deg"], current_r, flip)
    k, r, j6, blocked = _choose(cands, contact, _ref(current_r, current_xy, current_j6),
                                lambda rr: finger_hits(others, tcp_for_contact(contact, rr), rr, grip["pad_width_mm"],
                                                       held))
    if k < 0:
        return PlacePlan(False, "no wrist roll clears the fingers at the seated pose", key, inst.oid, base=base_key,
                         blocked_by=blocked)
    tcp = tcp_for_contact(contact, r)
    ins = entry["insertion"]
    contact, seat_z = tcp[2] + ins["engagement_mm"], tcp[2]
    hang = grip["pad_point_local_mm"][2] - entry["bbox_mm"]["min"][2]
    transit = max(_transit_z(others, hang), contact + HOVER_MM + APPROACH_MM)
    xy = [tcp[0], tcp[1]]
    path = [_row(xy + [transit], r, 1), _row(xy + [contact + HOVER_MM], r, 1), _row(xy + [contact], r, 1),
            _row(xy + [min(contact, seat_z + SLOW_ZONE_MM)], r, 1), _row(tcp, r, 1), _row(tcp, r, 0),
            _row(xy + [seat_z + APPROACH_MM], r, 0), _row(xy + [transit], r, 0)]
    plan = PlacePlan(True, "ok", key, inst.oid, k, r, j6, _row(tcp, r, 1)[:6], path, blocked, base=base_key,
                     target_pose=[round(float(v), 3) for v in origin] + [base.yaw], hover_z=contact + HOVER_MM,
                     contact_z=contact, slow_from_z=min(contact, seat_z + SLOW_ZONE_MM),
                     guard_floor_z=seat_z - GUARD_BELOW_SEAT_MM, capture_mm=ins["xy_capture_mm"])
    if flip is not None and k != flip:
        plan.notes.append("flip differs from the pick: re-grip needed")
    if not all(_in_workspace(p[:3]) for p in path):
        plan.feasible, plan.reason = False, "outside the workspace box (PLAN 3.1)"
    return plan


def plan_pull_out(scene: dict, key: str, current_r: float = 180.0, current_xy=(400.0, 0.0),
                  current_j6: float | None = None, catalog=None, maps=None) -> GraspPlan:
    """Dismantle: grip the top part on a base and pull it straight up, clear of every other installed part."""
    catalog, maps, instances, inst = _context(scene, key, catalog, maps)
    if inst.base is None:
        return GraspPlan(False, "not on a base: use plan_pick", key, inst.oid)
    plan = plan_pick(scene, key, current_r, current_xy, current_j6, catalog=catalog, maps=maps)
    if not plan.feasible:
        return plan
    held = solid_of(inst, catalog, maps)
    lifted = Solid(held.key, held.x, held.y, held.top, held.top)       # underside after a full lift
    above = [s for s in (solid_of(i, catalog, maps) for k, i in instances.items() if k != key)
             if s.bottom.size and float(s.bottom.min()) >= float(held.bottom.min())]
    hits = [h for h in sweep_hits(lifted, above) if instances[h].base == inst.base]
    if hits:
        return GraspPlan(False, "pull-out path blocked", key, inst.oid, blocked_by={0: hits, 1: hits})
    eng = catalog[inst.oid]["insertion"]["engagement_mm"]
    tcp = np.array(plan.tcp[:3])
    clear = tcp + [0, 0, eng + 2.0]
    plan.path = plan.path[:4] + [_row(clear, plan.r_deg, 1)] + plan.path[4:]
    plan.notes.append(f"pull slowly from z {tcp[2]:.1f} to {clear[2]:.1f} with a +-1 deg wiggle (ROBOT_BRAIN 2)")
    return plan


def blockers(scene: dict, key: str, current_r: float = 180.0, catalog=None, maps=None) -> list[str]:
    """Instances to move before `key` can be gripped: [] if a grip is possible now. Takes the roll whose blocker set
    is smallest. A base in the list cannot be moved (Andon)."""
    plan = plan_pick(scene, key, current_r, catalog=catalog, maps=maps)
    if plan.feasible or not plan.blocked_by:
        return []
    if plan.reason.startswith("parts above"):
        return list(plan.blocked_by[0])          # already in reverse sheet order: remove in this order
    sets = [v for v in plan.blocked_by.values() if v != ["J6_LIMIT"]]
    return sorted(min(sets, key=len)) if sets else []
