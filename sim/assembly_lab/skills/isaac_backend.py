"""Agent 1's Isaac Lab env + agent 2's planner: the only backend whose episodes are demos.

The env records through the Isaac Lab recorder (`CellEnvCfg.record_to`), so the HDF5 is exactly what Mimic's
annotate_demos.py / generate_dataset.py read (initial state, actions at the env's 0.1 s step, frame markers). The plan
is made inside the env's reset callback, on the settled layout, and its TCP frames become the Mimic object_ref frames
(grasp_frame / seat_frame / spot_frame) that the env's subtask signals use. Every attempt is one recorder episode;
`finish(ok)` marks it succeeded or failed before the next reset exports it.
"""

from __future__ import annotations

import numpy as np

from . import checks, frames
from .backends import (
    NoCase,
    _feasible,
    base_of,
    crowd,
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
            "crowded": "place_free", "defect": "defect_drop"}


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
            crowd(rng, layout, target)
        if case == "dropped":
            drop_near_base(rng, layout, target, base_of(product))
        self.plans, self.refusal = None, None
        obs = self.env.reset_with(layout, lambda poses: self._frames(poses, rng, product, target))
        if self.refusal:
            raise NoCase(self.refusal)
        return obs

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
        if sc in ("build", "wont_seat", "dropped"):
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

    def close(self, annotate_path=None) -> dict | None:
        """Export the last episode and close the recorder file, add agent 3's metadata to it, then shut the app down
        (SimulationApp.close() ends the process, so nothing can run after it)."""
        report = None
        rm = self.raw.env.recorder_manager
        if rm is not None:
            rm.export_episodes()
            rm.close()
        if annotate_path is not None:
            from .hdf5_io import annotate_recorder_file

            report = annotate_recorder_file(annotate_path)
            print(f"[record] annotated {annotate_path}: {report}", flush=True)
        self.env.close()
        return report


def _pose6(tf: np.ndarray) -> list[float]:
    from .geometry import matrix_to_pose

    return list(matrix_to_pose(np.asarray(tf, dtype=float)))
