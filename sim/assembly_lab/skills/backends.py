"""Where the recorder gets its world and its plans.

- `IsaacBackend`: agent 1's `AssemblyCellEnv` + agent 2's planner (`plan_pick`, `plan_place`, `blockers`). The only
  backend whose output is a demo.
- `KinematicBackend`: the dry-run test double (dry_run.py), product 1 base + M1_02 only. Never a demo.
"""

from __future__ import annotations

import math

import numpy as np

from . import checks, dry_run, frames
from .frames import LocalFrameEnv, scene_from_obs
from .geometry import matrix_to_pose, pose_to_matrix
from .plan import FreePlacePlan, InsertPlan, PickPlan, PullOutPlan

WORK_X, WORK_Y = (220.0, 470.0), (-270.0, 270.0)  # sdg/configs/cell_layout.yaml work_area
BASE_GAP_MM, PART_GAP_MM = 25.0, 8.0
FOOTPRINT_MM = {"F": 70.0, "M": 45.0}  # conservative layout spacing (bases ~60 mm, parts <= ~40 mm across)
DEFECT_DROP_MM = 15.0
SHEET = ("01", "02", "03", "04")


def sheet_parts(product: int) -> list[str]:
    return [f"M{product}_{s}" for s in SHEET]


def base_of(product: int) -> str:
    return f"F0{product}"


def sample_layout(rng: np.random.Generator, ids: list[str], installed: dict[str, str] | None = None,
                  tries: int = 2000) -> dict:
    """Random non-overlapping xy / yaw in the work area for every id not installed on a base."""
    installed = installed or {}
    placed: list[tuple[float, float, float]] = []
    objects = []
    for oid in ids:
        if oid in installed:
            objects.append({"id": oid, "x": 0.0, "y": 0.0, "yaw": 0.0, "state": f"on:{installed[oid]}"})
            continue
        r = FOOTPRINT_MM[oid[0]] / 2
        for _ in range(tries):
            x, y = rng.uniform(*WORK_X), rng.uniform(*WORK_Y)
            if 180.0 <= math.hypot(x, y) <= 490.0 and all(math.hypot(x - a, y - b) >= r + rb + PART_GAP_MM
                                                         for a, b, rb in placed):
                break
        else:
            raise RuntimeError(f"no free spot for {oid} after {tries} tries")
        placed.append((x, y, r))
        objects.append({"id": oid, "x": x, "y": y, "yaw": float(rng.uniform(-180.0, 180.0)), "state": "installed"})
    return {"objects": objects, "spares": [], "defects": []}


def free_spot(rng: np.random.Generator, obs: dict, radius: float = 30.0, tries: int = 2000) -> tuple[float, float]:
    """An empty test-bed spot (remembered by the caller), clear of every object on the table."""
    poses = [p for p in obs["sim"]["object_poses"].values() if WORK_X[0] - 100 < p[0] < WORK_X[1] + 100]
    for _ in range(tries):
        x, y = rng.uniform(*WORK_X), rng.uniform(*WORK_Y)
        if 180.0 <= math.hypot(x, y) <= 490.0 and all(math.hypot(x - p[0], y - p[1]) > 2 * radius for p in poses):
            return x, y
    raise RuntimeError("no empty test-bed spot")


def spot_plan(obs: dict, oid: str, grasp_tcp, rest_pose, kind: str, clear_z: float) -> FreePlacePlan:
    """TCP that leaves `oid` at `rest_pose` (its installed orientation on the table), gripped as planned."""
    t_obj_tcp = np.linalg.inv(pose_to_matrix(checks.object_pose(obs, oid))) @ pose_to_matrix(grasp_tcp)
    tcp = matrix_to_pose(pose_to_matrix(rest_pose) @ t_obj_tcp)
    return FreePlacePlan(oid, tcp, kind, clear_z, DEFECT_DROP_MM if kind == "defect" else 0.0)


def table_rest(oid: str, x: float, y: float, yaw: float) -> np.ndarray:
    """Agent 1's layout solver: the object standing on the table in its installed orientation."""
    from ..env.layout import LayoutSolver, pose6_from_tf, type_id

    return np.asarray(pose6_from_tf(LayoutSolver().table_tf(type_id(oid), x, y, yaw, "installed")), dtype=float)


class NoCase(RuntimeError):
    """The sampled layout does not contain the requested recovery case (resampled, counted separately)."""


def crowd(rng: np.random.Generator, layout: dict, target: str, gap_mm: float = 2.0) -> None:
    """Move one loose neighbour right next to `target` so its grip pad may be blocked (the planner decides)."""
    objs = {o["id"]: o for o in layout["objects"]}
    t = objs[target]
    loose = [o for o in layout["objects"] if o["id"] != target and o["id"][0] == "M" and o["state"] == "installed"]
    if not loose:
        raise NoCase("no loose neighbour to crowd with")
    n = loose[int(rng.integers(len(loose)))]
    a = rng.uniform(0, 2 * math.pi)
    d = FOOTPRINT_MM["M"] / 2 + gap_mm + FOOTPRINT_MM["M"] / 4
    n["x"], n["y"] = t["x"] + d * math.cos(a), t["y"] + d * math.sin(a)


class KinematicBackend:
    name = "kinematic"
    products = (1,)

    def __init__(self):
        self.env = None

    def reset(self, rng: np.random.Generator, product: int, case: str) -> dict:
        part = (rng.uniform(250, 420), rng.uniform(80, 220), rng.uniform(-90, 90))
        base = (rng.uniform(300, 450), rng.uniform(-200, -40), rng.uniform(-90, 90))
        objs, seats = dry_run.demo_world(part, base)
        if case == "fix_above":  # the part starts installed on its base
            objs["M1_02"]["pose"] = list(seats["M1_02"]["pose"])
        self.env = dry_run.KinematicEnv(objs, seats)
        self._plans = dry_run.demo_plans(objs, seats)
        return self.env.reset()

    def parts(self, product: int) -> list[str]:
        return ["M1_02"]

    def pick_insert(self, obs, part, base):
        return self._plans[0], self._plans[1]

    def pull_out(self, obs, part, rng):
        return self._plans[2], self._plans[3]

    def defect(self, obs, part, rng):
        return self._plans[0], self._plans[4]

    def crowded(self, obs, part, rng):
        """Pipeline check only: the dry-run world has no neighbour, so M1_02 itself is moved to a spot."""
        return self._plans[0], self._plans[3]


class IsaacBackend:
    """Agent 1's env (wrapped into CAD local frames) + agent 2's planner on the measured scene."""

    name = "isaac"
    products = (1, 2, 3, 4, 5, 6)

    def __init__(self, device: str = "cuda:0", products=(1, 2, 3, 4, 5, 6), cameras: tuple[str, ...] = ()):
        """Spawns only the listed products' objects. Seed demos need no images by default: Mimic's annotate /
        generate replays them in the env and renders the cameras itself."""
        from ..env.cell_env import AssemblyCellEnv, CellEnvCfg
        from ..objects import planner

        self.planner = planner
        pool = [oid for p in products for oid in (base_of(p), *sheet_parts(p))]
        self.env = LocalFrameEnv(AssemblyCellEnv(CellEnvCfg(device=device, pool=pool, cameras=tuple(cameras))))
        self.layout: dict = {}
        self.installed: dict[str, str] = {}

    def reset(self, rng: np.random.Generator, product: int, case: str, installed=None, target: str | None = None) -> dict:
        ids = [base_of(product), *sheet_parts(product)]
        self.installed = dict(installed or {})
        self.layout = sample_layout(rng, ids, self.installed)
        if case == "crowded" and target is not None:
            crowd(rng, self.layout, target)
        if case == "dropped" and target is not None:
            drop_near_base(rng, self.layout, target, base_of(product))
        return self.env.reset(self.layout)

    def parts(self, product: int) -> list[str]:
        return sheet_parts(product)

    def _ctx(self, obs: dict) -> dict:
        tcp = checks.tcp_pose(obs)
        return {"current_r": float(tcp[5]), "current_xy": (float(tcp[0]), float(tcp[1]))}

    def _scene(self, obs: dict) -> dict:
        return scene_from_obs(obs, self.installed)

    def pick_insert(self, obs, part, base):
        scene = self._scene(obs)
        g = _feasible(self.planner.plan_pick(scene, part, **self._ctx(obs)))
        p = _feasible(self.planner.plan_place(scene, part, base, flip=g.flip, **self._ctx(obs)))
        if p.flip != g.flip:
            raise NoCase(f"{part}: the place needs the other wrist turn (re-grip), not a single pick-insert")
        return to_pick(g), to_insert(p, part, base)

    def pull_out(self, obs, part, rng) -> tuple[PullOutPlan, FreePlacePlan]:
        g = _feasible(self.planner.plan_pull_out(self._scene(obs), part, **self._ctx(obs)))
        eng = float(frames.catalog()[frames.type_id(part)]["insertion"]["engagement_mm"])
        pull = PullOutPlan(part, np.asarray(g.tcp, float), float(g.grip["pad_width_mm"]), eng, float(g.transit_z))
        x, y = free_spot(rng, obs)
        rest = table_rest(part, x, y, float(rng.uniform(-180, 180)))
        return pull, spot_plan(obs, part, pull.grasp_tcp, rest, "test_bed", pull.clear_z_mm)

    def crowded(self, obs, part, rng) -> tuple[PickPlan, FreePlacePlan]:
        """The planner's blocker of `part` goes to an empty remembered test-bed spot (move the blocker first)."""
        scene = self._scene(obs)
        blockers = list(self.planner.blockers(scene, part, current_r=self._ctx(obs)["current_r"]))
        movable = [b for b in blockers if frames.catalog().get(frames.type_id(b), {}).get("sheet_order", 0) > 0]
        if not blockers or len(movable) != len(blockers):
            raise NoCase(f"{part} is not crowded by a movable part in this layout ({blockers})")
        pick = to_pick(_feasible(self.planner.plan_pick(scene, movable[0], **self._ctx(obs))))
        x, y = free_spot(rng, obs)
        rest = table_rest(movable[0], x, y, float(rng.uniform(-180, 180)))
        return pick, spot_plan(obs, movable[0], pick.grasp_tcp, rest, "test_bed", pick.lift_z_mm)

    def defect(self, obs, part, rng) -> tuple[PickPlan, FreePlacePlan]:
        from ..env.layout import LayoutSolver

        pick = to_pick(_feasible(self.planner.plan_pick(self._scene(obs), part, **self._ctx(obs))))
        x, y = LayoutSolver().defect_xy(int(rng.integers(0, 6)))
        rest = table_rest(part, x, y, float(checks.object_pose(obs, part)[5]))
        return pick, spot_plan(obs, part, pick.grasp_tcp, rest, "defect", pick.lift_z_mm)


def _feasible(plan):
    if not plan.feasible:
        raise NoCase(f"planner refused {plan.key}: {plan.reason}")
    return plan


def to_pick(g) -> PickPlan:
    """planner.GraspPlan -> PickPlan (TCP on the CAD pad, approach / lift at the planner's clear transit height)."""
    tcp = np.asarray(g.tcp, dtype=float)
    return PickPlan(g.key, tcp, float(g.grip["pad_width_mm"]), float(g.transit_z), float(g.transit_z),
                    grip_index=0)


def to_insert(p, part: str, base: str) -> InsertPlan:
    """planner.PlacePlan -> InsertPlan, with the per-part insertion numbers from objects.yaml."""
    entry = frames.catalog()[frames.type_id(part)]
    ins, grip = entry["insertion"], entry["grips"][0]
    tcp = np.asarray(p.tcp, dtype=float)
    x, y, z, yaw = p.target_pose
    clearance = float(ins["xy_clearance_mm"])
    return InsertPlan(part, base, tcp, np.array([x, y, z, 0.0, 0.0, yaw]), float(p.contact_z - tcp[2]),
                      float(p.path[0][2]), chamfer_mm=max(0.0, float(ins["xy_capture_mm"]) - clearance),
                      clearance_mm=clearance, lead_dir_deg=float(yaw),
                      hang_mm=float(grip["tcp_local_mm"][2] - entry["bbox_mm"]["min"][2]),
                      yaw_tol_deg=ins.get("yaw_tol_deg"))


def drop_near_base(rng: np.random.Generator, layout: dict, target: str, base: str) -> None:
    """Re-pick after a drop: the part lies upright 60-120 mm from its base, where it fell during the carry."""
    objs = {o["id"]: o for o in layout["objects"]}
    b, t = objs[base], objs[target]
    for _ in range(200):
        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(60.0, 120.0)
        x, y = b["x"] + d * math.cos(a), b["y"] + d * math.sin(a)
        clear = all(math.hypot(x - o["x"], y - o["y"]) > FOOTPRINT_MM["M"] for o in layout["objects"]
                    if o is not t and o["state"] == "installed")
        if WORK_X[0] <= x <= WORK_X[1] and WORK_Y[0] <= y <= WORK_Y[1] and clear:
            t["x"], t["y"] = x, y
            return
    raise NoCase("no free spot near the base for the dropped part")
