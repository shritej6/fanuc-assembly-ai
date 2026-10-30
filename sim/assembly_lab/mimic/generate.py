"""Generate Mimic demos for one (skill, product) group on one GPU, straight over agent 1's AssemblyCellEnv.

    # inside Isaac Lab's python on fa-mimic (one process per GPU slot):
    CUDA_VISIBLE_DEVICES=0 python -m sim.assembly_lab.mimic.generate --skill pick_insert --product 4 \
        --seeds ~/t12_data/demos/pick_insert/product_4.hdf5 --out ~/t12_data/mimic/pick_insert/p4/w0.hdf5 \
        --target 200 --max-attempts 2000 --seed 0

Every attempt is counted. A demo is written only if it reaches every subtask signal AND passes every precision gate
(gates.py, INTERFACES.md). Rejections go to <out>.rejects.jsonl with every gate value; the totals and the measured
wall time go to <out>.stats.json (throughput is computed from those, never estimated).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import gates as G
from .mimicgen import (
    add_noise,
    interpolate,
    load_segments,
    select_source,
    transform_rows,
)
from .spec import load_spec

TAPER_MM = 20.0  # action noise fades to zero within this distance of the subtask frame (precise at contact)
ENV_NAME = "T12-AssemblyCell-Mimic"


def taper_noise(rows: np.ndarray, frame: np.ndarray, pos_mm: float, rot_deg: float, rng) -> np.ndarray:
    d = np.linalg.norm(rows[:, :3] - frame[:3, 3], axis=1)
    scale = np.clip(d / TAPER_MM, 0.0, 1.0)
    out = rows.copy()
    for i in np.flatnonzero(scale > 0):
        out[i] = add_noise(rows[i:i + 1], pos_mm * scale[i], rot_deg * scale[i], rng)[0]
    return out


def trim_after_retreat(rows: np.ndarray) -> np.ndarray:
    """Last subtask: keep rows up to the top of the retreat after the release; the return home is re-planned live
    (a rigidly transformed 'home' would land somewhere else)."""
    opened = np.flatnonzero(rows[:, 6] < 0.5)
    if not opened.size:
        return rows
    k = int(opened[0])
    while k + 1 < len(rows) and rows[k + 1, 2] >= rows[k, 2] - 1e-6 and rows[k + 1, 6] < 0.5:
        k += 1
    return rows[:k + 1]


def split_after_close(segs: list) -> tuple[list, list]:
    """Split before the first row that leaves the grasp pose after the gripper closed (end of the settle hold)."""
    from dataclasses import replace

    last = segs[-1]
    rows = last.rows
    closed = np.flatnonzero(rows[:, 6] > 0.5)
    if not closed.size:
        raise ValueError("grasp subtask never closes the gripper")
    k = int(closed[0])
    while k + 1 < len(rows) and np.allclose(rows[k + 1, :6], rows[k, :6], atol=1e-6):
        k += 1
    return [*segs[:-1], replace(last, rows=rows[:k + 1])], [replace(last, rows=rows[k + 1:])]


def _align_r(rows: np.ndarray, current_r: float) -> np.ndarray:
    rows = rows.copy()
    rows[:, 5] -= 360.0 * np.round((rows[0, 5] - current_r) / 360.0)
    return rows


SCENARIO = {"pick_insert": ("build", "nominal"), "pull_out": ("fix_above", "fix_above"), "defect_drop": ("defect", "defect")}


class Generator:
    """Precision gates: agent 3's skills/gates.py (one definition for scripted and generated demos) plus gates.py's
    stricter non-target check over every step (a part bumped and settled back still fails)."""

    def __init__(self, a):
        from ..skills import backends, episodes, gates, hdf5_io, record
        from ..skills.force_guard import load_limits as skill_limits

        self.spec = load_spec()
        self.skill, self.product, self.args = a.skill, a.product, a
        self.sk = self.spec.skills[a.skill]
        self.backends, self.episodes, self.gates, self.hdf5_io, self.record = backends, episodes, gates, hdf5_io, record
        self.limits_skill = skill_limits(a.limits) if a.limits else skill_limits()
        self.segments = load_segments(str(a.seeds), self.sk.signals, [s.object_ref for s in self.sk.subtasks])
        self.backend = backends.KinematicBackend() if a.backend == "kinematic" else backends.IsaacBackend("cuda:0")
        self.rng = np.random.default_rng(a.seed)
        self.gl = G.GateLimits()

    def parts(self) -> list[str]:
        return [self.args.part] if self.args.part else self.backend.parts(self.product)

    # --- scene + plan -----------------------------------------------------------------------------------------
    def _scene(self, part: str):
        b, rng, prod = self.backends, self.rng, self.product
        scenario, case = SCENARIO[self.skill]
        parts = self.backend.parts(prod)
        upto = parts.index(part) + (1 if self.skill == "pull_out" else 0)
        installed = {p: b.base_of(prod) for p in parts[:upto]} if self.skill != "defect_drop" else {}
        obs = self.record._reset(self.backend, rng, prod, scenario, installed, part)
        if self.skill == "pick_insert":
            return obs, self.backend.pick_insert(obs, part, b.base_of(prod)), case
        if self.skill == "pull_out":
            return obs, self.backend.pull_out(obs, part, rng), case
        return obs, self.backend.defect(obs, part, rng), case

    def _begin(self, rec, plans, case: str) -> None:
        """Same header as agent 3's episodes.<skill>: meta, instruction, Mimic frames, allowed targets."""
        e = self.episodes
        first, second = plans
        oid = first.object_id
        rec.targets.add(oid)
        start = e._part_xy_yaw(rec, oid)
        if self.skill == "pick_insert":
            base = e.checks.object_pose(rec.obs, second.base_id)
            e._begin(rec, self.skill, oid, second.base_id, first.grip_index, case,
                     start + [first.grasp_tcp[5], base[0], base[1], base[5]])
            rec.define_frame("grasp_frame", oid, first.grasp_tcp)
            rec.define_frame("seat_frame", second.base_id, second.seated_tcp)
            return
        yaw = 0.0 if self.skill == "defect_drop" else e._spot_yaw(rec, oid, first.grasp_tcp[5], second)
        e._begin(rec, self.skill, oid, None, first.grip_index, case,
                 start + [first.grasp_tcp[5], second.spot_tcp[0], second.spot_tcp[1], yaw])
        rec.define_frame("grasp_frame", oid, first.grasp_tcp)
        rec.define_frame("spot_frame", None, second.spot_tcp)

    # --- one episode ------------------------------------------------------------------------------------------
    def _subtask_rows(self, rec, i: int, sub, oid: str) -> list:
        from ..skills.motion import Segment

        frame = rec.frame_pose(rec.obs, sub.object_ref)
        src = select_source(self.segments[i], frame, oid, self.spec.raw["selection"]["nn_k"], self.rng)
        rows = _align_r(transform_rows(src.rows, src.frame_pose, frame), rec.pose[5])
        rows = trim_after_retreat(rows) if i == len(self.sk.subtasks) - 1 else rows
        pos_mm = sub.noise_pos_m * 1000.0
        rows = taper_noise(rows, frame, pos_mm, pos_mm * float(self.spec.raw["rot_deg_per_mm"]), self.rng)
        interp = interpolate(np.r_[rec.pose, rec.gripper], rows[0], sub.interp_steps)
        segs = [Segment(self.skill, f"{sub.name}_interp", sub.name, sub.object_ref, interp)] if len(interp) else []
        return [*segs, Segment(self.skill, sub.name, sub.name, sub.object_ref, rows)]

    def _run_subtasks(self, rec, plans) -> None:
        g, checks = self.gates, self.episodes.checks
        first, second = plans
        oid = first.object_id
        planned_grip = rec._planned_grip(oid, first.grasp_tcp)
        for i, sub in enumerate(self.sk.subtasks):
            segs = self._subtask_rows(rec, i, sub, oid)
            if sub.term_signal == "grasped":
                # as agent 3's runner: the grip gate is read once the fingers have closed and settled, on the pad
                head, tail = split_after_close(segs)
                rec.run(head)
                g.grip_gates(rec.gate_log, rec.obs, oid, planned_grip, first.grasp_tcp[5])
                rec.run(tail)
                rec.in_hand[oid] = checks.in_hand(rec.obs, oid)
            else:
                rec.run(segs)
            if sub.term_signal:
                rec.mark(sub.term_signal)
            if sub.term_signal == "dropped":
                rec.guard.exempt.add(oid)
        rec.go_home()
        rec.close()
        final = checks.object_pose(rec.obs, oid)
        if self.skill == "pick_insert":
            g.seat_gates(rec.gate_log, final, second.seated_object_pose, second.yaw_tol_deg)
        elif self.skill == "pull_out":
            g.spot_gates(rec.gate_log, final, checks.expected_object(second.spot_tcp, rec.in_hand[oid]))
        else:
            ok, why = checks.in_defect_area(final, second.spot_tcp)
            rec.gate_log.check("defect_area", 0.0, 0.0, ok=ok)
            rec.gate_log.values["defect_area"]["why"] = why
            rec.gate_log.require(("defect_area",))
        rec.finish_gates()

    def _every_step_others(self, rec, oid: str) -> G.GateReport:
        rep = G.GateReport(ok=False)
        obs_all = [f.obs for f in rec.frames] + [rec.obs]
        uids = obs_all[0]["sim"]["object_poses"].keys()
        traces = {u: np.array([o["sim"]["object_poses"][u] for o in obs_all], dtype=float) for u in uids}
        G.gate_others(rep, traces, oid, self.gl)
        missing = [s for s in self.sk.signals if s not in rec.marks]
        if missing:
            rep.failed.append(f"signals_missing:{missing}")
        return G.finish(rep)

    def attempt(self, part: str):
        from ..skills.force_guard import ForceGuard
        from ..skills.runner import EpisodeFailed, EpisodeRecorder

        obs, plans, case = self._scene(part)
        rec = EpisodeRecorder(self.backend.env, ForceGuard(self.limits_skill), obs,
                              allow_touch_proxy=self.args.allow_touch_proxy)
        failed: list[str] = []
        extra = G.GateReport(ok=False)
        try:
            self._begin(rec, plans, case)
            self._run_subtasks(rec, plans)
            extra = self._every_step_others(rec, plans[0].object_id)
            failed += extra.failed
        except (EpisodeFailed, self.gates.GateFailed, ValueError) as exc:  # env TcpError is a ValueError
            failed.append(f"aborted: {exc}")
        failed += [f"gate:{n}" for n in rec.gate_log.failed if f"gate:{n}" not in failed]
        return rec, failed, {**rec.gate_log.values, **extra.values}


def run(a) -> dict:
    a.seeds = a.seeds.expanduser()
    gen = Generator(a)
    out = a.out.expanduser()
    if a.backend == "kinematic" and out.resolve().is_relative_to((Path.home() / "t12_data").resolve()):
        raise SystemExit("kinematic dry runs never write under ~/t12_data")
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = gen.parts()
    stats = {"skill": a.skill, "product": a.product, "attempts": 0, "accepted": 0, "failed_gates": {}}
    t0 = time.time()
    with open(f"{out}.rejects.jsonl", "a") as rej:
        while stats["accepted"] < a.target and stats["attempts"] < a.max_attempts:
            part = parts[stats["attempts"] % len(parts)]
            rec, failed, values = gen.attempt(part)
            stats["attempts"] += 1
            if not failed:
                attrs = {"gates": json.dumps(values, default=str), "generator": "mimicgen", "backend": a.backend,
                         "touch_gate": rec.gate_log.touch_method, "limits_status": gen.limits_skill.status}
                if a.backend == "kinematic":
                    attrs["KINEMATIC_DRY_RUN"] = True
                gen.hdf5_io.append_demo(out, rec, attrs, ENV_NAME)
                stats["accepted"] += 1
            else:
                for f in failed:
                    key = f.split(":")[1] if f.startswith("gate:") else f.split(":")[0]
                    stats["failed_gates"][key] = stats["failed_gates"].get(key, 0) + 1
                rej.write(json.dumps({"part": part, "failed": failed, "values": values}, default=str) + "\n")
            stats["wall_s"] = round(time.time() - t0, 1)
            stats["acceptance_rate"] = stats["accepted"] / stats["attempts"]
            stats["demos_per_hour"] = round(stats["accepted"] / stats["wall_s"] * 3600, 1) if stats["wall_s"] else None
            Path(f"{out}.stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill", required=True, choices=["pick_insert", "pull_out", "defect_drop"])
    ap.add_argument("--product", type=int, required=True)
    ap.add_argument("--part", default=None, help="only this part (default: every sheet part in turn)")
    ap.add_argument("--seeds", type=Path, required=True, help="agent 3's seed HDF5 for this skill + product")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=("isaac", "kinematic"), default="isaac")
    ap.add_argument("--limits", type=Path, default=None, help="override physics/limits.yaml")
    ap.add_argument("--allow-touch-proxy", action="store_true",
                    help="accept sim.gripper_obj as the touch gate when the env has no sim.robot_touch (stamped)")
    print(json.dumps(run(ap.parse_args()), indent=2))


if __name__ == "__main__":
    main()
