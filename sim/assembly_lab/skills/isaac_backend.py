"""Agent 1's Isaac Lab env + agent 2's planner: the only backend whose episodes are demos.

The env records through the Isaac Lab recorder (`CellEnvCfg.record_to`), so the HDF5 is exactly what Mimic's
annotate_demos.py / generate_dataset.py read (initial state, actions at the env's 0.1 s step, frame markers). The plan
is made inside the env's reset callback, on the settled layout, and its TCP frames become the Mimic object_ref frames
(grasp_frame / seat_frame / spot_frame) that the env's subtask signals use. Every attempt is one recorder episode;
`finish(ok)` marks it succeeded or failed before the next reset exports it.
"""

from __future__ import annotations

import copy
import functools

import numpy as np

from . import checks, frames
from .backends import (
    FOOTPRINT_MM,
    PART_GAP_MM,
    WORK_X,
    WORK_Y,
    NoCase,
    _feasible,
    base_of,
    drop_near_base,
    free_spot,
    sample_layout,
    sheet_parts,
    spot_plan,
    table_rest,
    to_insert,
    to_pick,
)
from .frames import LocalFrameEnv, scene_from_obs
from .geometry import pose_to_matrix
from .plan import PullOutPlan

SKILL_OF = {"build": "pick_insert", "wont_seat": "pick_insert", "dropped": "pick_insert", "fix_above": "pull_out",
            "crowded": "place_free", "defect": "defect_drop", "align_insert": "pick_insert", "align_grasp": "pick_insert"}
CROWD_LAYOUTS = 20  # random layouts tried per reset for the crowded case
CROWD_DIRECTIONS = 24  # neighbour directions tried around the target
CROWD_GAPS_MM = (1.0, 2.0, 3.0)  # neighbour footprint clear of the target's by this much


class IsaacBackend:
    name = "isaac"

    def __init__(self, scenario: str, product: int, record_to: str, device: str = "cuda:0",
                 cameras: tuple[str, ...] = ()):
        """One env per process: one scenario (the env computes that skill's signals) and one product's objects.
        Seed demos carry no images by default: Mimic re-renders the cameras when it replays them."""
        from ..env.cell_env import AssemblyCellEnv, CellEnvCfg
        from ..objects import planner

        self.planner, self.scenario, self.products = planner, scenario, (product,)
        pool = [base_of(product), *sheet_parts(product)]
        self.raw = AssemblyCellEnv(CellEnvCfg(device=device, pool=pool, cameras=tuple(cameras),
                                              skill=SKILL_OF[scenario], record_to=record_to))
        self.env = LocalFrameEnv(self.raw)
        self.record_path = record_to  # the recorder file now being written; rotate() moves to the next part's
        self.installed: dict[str, str] = {}
        self.plans: tuple | None = None
        self.refusal: str | None = None

    def parts(self, product: int) -> list[str]:
        return sheet_parts(product)

    # ------------------------------------------------------------------ reset + plan
    def reset(self, rng: np.random.Generator, product: int, case: str, installed=None, target: str | None = None) -> dict:
        self.installed = dict(installed or {})
        layout = sample_layout(rng, [base_of(product), *sheet_parts(product)], self.installed)
        if case == "crowded":
            layout = self._crowded_layout(rng, product, target)
        if case == "dropped":
            drop_near_base(rng, layout, target, base_of(product))
        self.plans, self.refusal = None, None
        obs = self.env.reset_with(layout, lambda poses: self._frames(poses, rng, product, target))
        if self.refusal:
            raise NoCase(self.refusal)
        return obs

    def _crowded_layout(self, rng, product: int, target: str) -> dict:
        """A layout where a loose neighbour blocks every wrist turn on `target` (planner-checked on the layout's nominal
        poses before the env spends a reset on it). The neighbour is set `gap` mm clear of the target's footprint
        along a random direction (separating axis: the two never overlap); most directions leave one turn free."""
        last = "no candidate placement"
        for _ in range(CROWD_LAYOUTS):
            layout = sample_layout(rng, [base_of(product), *sheet_parts(product)], self.installed)
            for trial in self._crowd_candidates(rng, layout, target):
                try:
                    self._plan(self._obs_like(self._nominal_body_tf(trial)), rng, product, target)
                    return trial
                except NoCase as e:
                    last = str(e)
        raise NoCase(f"{target}: no crowded layout in {CROWD_LAYOUTS} layouts ({last})")

    def _crowd_candidates(self, rng, layout: dict, target: str):
        """Copies of `layout` with one loose neighbour moved next to `target` (clear of every other object)."""
        from ..objects.scene import build_instances, solid_of

        catalog, maps = self.planner.load_catalog()
        objs = {o["id"]: o for o in layout["objects"]}
        loose = [k for k, o in objs.items() if k != target and k[0] == "M" and o["state"] == "installed"]
        if not loose:
            raise NoCase("no loose neighbour to crowd with")
        inst = build_instances(scene_from_obs(self._obs_like(self._nominal_body_tf(layout)), self.installed), catalog)
        t = inst[target]
        t_pts = _cells(solid_of(t, catalog, maps)) - [t.x, t.y]
        for n in rng.permutation(loose):
            n_pts = _cells(solid_of(inst[n], catalog, maps)) - [inst[n].x, inst[n].y]
            for a in rng.permutation(CROWD_DIRECTIONS) * (2 * np.pi / CROWD_DIRECTIONS):
                u = np.array([np.cos(a), np.sin(a)])
                d = (t_pts @ u).max() - (n_pts @ u).min() + float(rng.choice(CROWD_GAPS_MM))
                x, y = t.x + d * u[0], t.y + d * u[1]
                if not _spot_clear(layout, (x, y), skip=(n, target)):
                    continue
                trial = copy.deepcopy(layout)
                moved = next(o for o in trial["objects"] if o["id"] == n)
                moved["x"], moved["y"] = float(x), float(y)
                yield trial

    @staticmethod
    def _nominal_body_tf(layout: dict) -> dict:
        """Body poses (4x4 UF0 mm) the layout asks for, before physics settles them (installed parts: pose unused)."""
        solver = _layout_solver()
        return {o["id"]: solver.table_tf(frames.type_id(o["id"]), o.get("x", 0.0), o.get("y", 0.0),
                                         o.get("yaw", 0.0), "installed")
                for o in layout["objects"]}

    def _obs_like(self, body_tf: dict) -> dict:
        roles = {u: p.role for u, p in self.raw.placements.items()}
        poses = {u: list(frames.body_to_local(u, _pose6(tf))) for u, tf in body_tf.items()}
        return {"tcp_pose": [400.0, 0.0, 100.0, -180.0, 0.0, 0.0],
                "sim": {"object_poses": poses, "on_table": {u: roles.get(u) != "parked" for u in poses}}}

    def _frames(self, body_tf: dict, rng, product: int, target: str) -> dict:
        """Env reset callback: plan on the settled layout; return the plan's TCP frames (4x4 UF0 mm)."""
        obs = self._obs_like(body_tf)
        try:
            self.plans = self._plan(obs, rng, product, target)
        except NoCase as e:
            self.refusal = str(e)
            any_part = next(u for u in body_tf if u.startswith("M"))
            return {f: body_tf[any_part] for f in ("grasp_frame", "seat_frame", "spot_frame")}
        first, second = self.plans
        grasp = pose_to_matrix(first.grasp_tcp)
        seat = pose_to_matrix(second.seated_tcp) if hasattr(second, "seated_tcp") else grasp
        spot = pose_to_matrix(second.spot_tcp) if hasattr(second, "spot_tcp") else grasp
        return {"grasp_frame": grasp, "seat_frame": seat, "spot_frame": spot}

    def _plan(self, obs: dict, rng, product: int, part: str) -> tuple:
        scene = scene_from_obs(obs, self.installed)
        sc = self.scenario
        if sc in ("build", "wont_seat", "dropped", "align_insert", "align_grasp"):
            g = _feasible(self.planner.plan_pick(scene, part))
            p = _feasible(self.planner.plan_place(scene, part, base_of(product), flip=g.flip))
            if p.flip != g.flip:
                raise NoCase(f"{part}: the place needs the other wrist turn (re-grip)")
            return to_pick(g), to_insert(p, part, base_of(product))
        if sc == "fix_above":
            g = _feasible(self.planner.plan_pull_out(scene, part))
            eng = float(frames.catalog()[frames.type_id(part)]["insertion"]["engagement_mm"])
            pull = PullOutPlan(part, np.asarray(g.tcp, float), float(g.grip["pad_width_mm"]), eng, float(g.transit_z))
            x, y = free_spot(rng, obs)
            rest = table_rest(part, x, y, float(rng.uniform(-180, 180)))
            return pull, spot_plan(obs, part, pull.grasp_tcp, rest, "test_bed", pull.clear_z_mm)
        if sc == "crowded":
            blockers = list(self.planner.blockers(scene, part))
            movable = [b for b in blockers if frames.catalog().get(frames.type_id(b), {}).get("sheet_order", 0) > 0]
            if not blockers or len(movable) != len(blockers):
                raise NoCase(f"{part} is not crowded by a movable part ({blockers})")
            pick = to_pick(_feasible(self.planner.plan_pick(scene, movable[0])))
            x, y = free_spot(rng, obs)
            rest = table_rest(movable[0], x, y, float(rng.uniform(-180, 180)))
            return pick, spot_plan(obs, movable[0], pick.grasp_tcp, rest, "test_bed", pick.lift_z_mm)
        if sc == "defect":
            from ..env.layout import LayoutSolver

            pick = to_pick(_feasible(self.planner.plan_pick(scene, part)))
            x, y = LayoutSolver().defect_xy(int(rng.integers(0, 6)))
            rest = table_rest(part, x, y, float(checks.object_pose(obs, part)[5]))
            return pick, spot_plan(obs, part, pick.grasp_tcp, rest, "defect", pick.lift_z_mm)
        raise ValueError(f"unknown scenario {sc!r}")

    # ------------------------------------------------------------------ plans for record.py
    def pick_insert(self, obs, part, base):
        return self.plans

    def pull_out(self, obs, part, rng):
        return self.plans

    def crowded(self, obs, part, rng):
        return self.plans

    def defect(self, obs, part, rng):
        return self.plans

    def finish(self, ok: bool) -> None:
        """Mark the recorder episode succeeded / failed; it is exported at the next reset or at close."""
        import torch

        rm = self.raw.env.recorder_manager
        if rm is not None:
            rm.set_success_to_episodes(None, torch.tensor([[bool(ok)]], dtype=torch.bool, device=self.raw.env.device))

    def rotate(self, next_path: str) -> dict | None:
        """Close the current part's file (export its last episode, annotate it) and record on into `next_path`, so
        each part's demos are readable as soon as that part is done (Mimic reads closed files while the run goes on)."""
        rm = self.raw.env.recorder_manager
        if rm is None:
            return None
        rm.export_episodes()
        rm._dataset_file_handler.close()  # Isaac Lab has no public rotate; the handler is the file
        report = self._annotate(self.record_path)
        rm._dataset_file_handler = rm.cfg.dataset_file_handler_class_type()
        rm._dataset_file_handler.create(str(next_path), env_name=getattr(self.raw.env.cfg, "env_name", None))
        self.record_path = str(next_path)
        return report

    @staticmethod
    def _annotate(path) -> dict:
        from .hdf5_io import annotate_recorder_file

        report = annotate_recorder_file(path)
        print(f"[record] annotated {path}: {report}", flush=True)
        return report

    def close(self, annotate_path=None) -> dict | None:
        """Export the last episode and close the recorder file, add agent 3's metadata to it, then shut the app down
        (SimulationApp.close() ends the process, so nothing can run after it)."""
        report = None
        rm = self.raw.env.recorder_manager
        if rm is not None:
            rm.export_episodes()
            rm.close()
        if annotate_path is not None:
            report = self._annotate(annotate_path)
        self.env.close()
        return report


def _cells(solid) -> np.ndarray:
    return np.column_stack([solid.x, solid.y])


def _spot_clear(layout: dict, xy, skip) -> bool:
    """`xy` is in the work area and as far from every other table object as the layout sampler keeps them."""
    x, y = xy
    if not (WORK_X[0] <= x <= WORK_X[1] and WORK_Y[0] <= y <= WORK_Y[1]):
        return False
    r = FOOTPRINT_MM["M"] / 2
    return all(np.hypot(x - o["x"], y - o["y"]) >= r + FOOTPRINT_MM[o["id"][0]] / 2 + PART_GAP_MM
               for o in layout["objects"] if o["id"] not in skip and not o["state"].startswith("on:"))


@functools.lru_cache(maxsize=1)
def _layout_solver():
    from ..env.layout import LayoutSolver

    return LayoutSolver()


def _pose6(tf: np.ndarray) -> list[float]:
    from .geometry import matrix_to_pose

    return list(matrix_to_pose(np.asarray(tf, dtype=float)))
