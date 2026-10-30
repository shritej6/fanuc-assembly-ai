"""Generate Mimic demos for one (skill, product) group in one Isaac Sim process, over agent 3's recorder stack.

    # inside the Isaac Lab container (env/isaaclab_run.sh), one process per worker slot:
    python -m sim.assembly_lab.mimic.generate --skill pick_insert --product 4 \
        --seeds ~/t12_data/demos/pick_insert/product_4_*.hdf5 --out ~/t12_data/mimic/pick_insert/p4/w0.hdf5 \
        --target 200 --max-attempts 2000 --seed 0 --cameras table side wrist

Each attempt resets agent 1's env on a fresh layout through agent 3's IsaacBackend (the planner makes the plan and the
Mimic frames on the settled layout), replays MimicGen-transformed source segments (mimicgen.py) through agent 3's
EpisodeRecorder, and keeps the episode only if it passes exactly the gates a scripted source demo must pass: grip
gates, grasp/slip check, seat check, force guard on every step (per-object limits), seat gates, tracking <= 0.1 mm,
others unmoved, robot touch only on the target, plus mimic/gates.py's every-step non-target check. The Isaac Lab
recorder writes every attempt to --out (kept ones with success=True, annotated with agent 3's metadata at close), so
generated files have the same layout as the source demos. The recorder file is closed and annotated every
--rotate-every attempts (<out stem>_c<k>.hdf5), so kept demos can be exported while the worker runs. Totals and the measured wall time go to <out>.stats.json;
every rejection with its reason goes to <out>.rejects.jsonl.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from . import gates as G
from .mimicgen import MAX_CARRY_DRIFT_MM, load_sources, select_source, transform_held, transform_rows
from .mimicgen import add_noise as _add_noise
from .spec import load_spec

TAPER_MM = 20.0  # action noise fades to zero within this distance of the subtask's contact row (precise at contact)
SCENARIO = {"pick_insert": "build", "pull_out": "fix_above"}  # skill -> the scenario agent 3's backend plans for
SOURCE_CASES = {"pick_insert": ("nominal",), "pull_out": ("fix_above",)}  # staged recovery cases are never sources
CORRIDOR_MM = 15.0
RETREAT_XY_MM = 1.0


def taper_noise(rows: np.ndarray, ref_xyz: np.ndarray, pos_mm: float, rot_deg: float, rng) -> np.ndarray:
    d = np.linalg.norm(rows[:, :3] - ref_xyz, axis=1)
    scale = np.clip(d / TAPER_MM, 0.0, 1.0)
    out = rows.copy()
    for i in np.flatnonzero(scale > 0):
        out[i] = _add_noise(rows[i:i + 1], pos_mm * scale[i], rot_deg * scale[i], rng)[0]
    return out


def contact_row(rows: np.ndarray) -> int:
    """The row the subtask is precise about: the first gripper change, else the last row."""
    change = np.flatnonzero(np.abs(np.diff(rows[:, 6])) > 0.5)
    return int(change[0]) + 1 if change.size else len(rows) - 1


def trim_to_corridor(rows: np.ndarray) -> np.ndarray:
    """Drop the leading rows that travel from where the SOURCE robot was (home, or the source part's lift spot):
    after the transform they mean nothing. Keep from the first row after which every row up to the subtask's contact
    row stays within CORRIDOR_MM (xy) of it: the vertical corridor above the grasp pad / seat. The connector rises in
    place, crosses at the higher z and descends to that row instead."""
    c = contact_row(rows)
    far = np.flatnonzero(np.linalg.norm(rows[:c + 1, :2] - rows[c, :2], axis=1) > CORRIDOR_MM)
    return rows[int(far[-1]) + 1:] if far.size else rows


def trim_after_retreat(rows: np.ndarray) -> np.ndarray:
    """Last subtask: keep the release and the straight-up retreat (xy within RETREAT_XY_MM); the return home is
    re-planned live by the recorder's go_home."""
    opened = np.flatnonzero(rows[:, 6] < 0.5)
    if not opened.size:
        return rows
    k = int(opened[0])
    xy = rows[k, :2]
    while (k + 1 < len(rows) and rows[k + 1, 2] >= rows[k, 2] - 1e-6 and rows[k + 1, 6] < 0.5
           and np.linalg.norm(rows[k + 1, :2] - xy) <= RETREAT_XY_MM):
        k += 1
    return rows[:k + 1]


def split_after_close(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split after the close-and-settle hold (the grip gates are read there, as agent 3's runner does)."""
    closed = np.flatnonzero(rows[:, 6] > 0.5)
    if not closed.size:
        raise ValueError("grasp subtask never closes the gripper")
    k = int(closed[0])
    while k + 1 < len(rows) and np.allclose(rows[k + 1, :6], rows[k, :6], atol=1e-6) and rows[k + 1, 6] > 0.5:
        k += 1
    return rows[:k + 1], rows[k + 1:]


def align_r(rows: np.ndarray, ref_r: float, at: int = 0) -> np.ndarray:
    """Shift r by whole turns so row `at` lands next to ref_r (the planner's roll, chosen inside the J6 range)."""
    rows = rows.copy()
    rows[:, 5] -= 360.0 * np.round((rows[at, 5] - ref_r) / 360.0)
    return rows


def connector(pose: np.ndarray, gripper: float, goal: np.ndarray, skill: str, sub, careful: bool):
    """Rows from the current target to `goal` through agent 3's Motion (workspace box, 6 deg tool tilt, per-row speed
    limits): rise in place, cross at the higher z, descend. `careful`: one straight SLOW move (part near contact)."""
    from ..skills.motion import FREE, SLOW, Motion, above

    m = Motion(np.asarray(pose, float), gripper, skill, sub.name, sub.object_ref)
    if careful:
        m.move("mimic_connect", goal[:6], SLOW)
        return m.segments
    z = max(pose[2], goal[2])
    m.move("mimic_rise", above(pose, z), FREE)
    m.move("mimic_cross", above(goal[:6], z), FREE)
    m.move("mimic_descend", goal[:6], FREE)
    return m.segments


def free_space_starts(subtasks) -> list[bool]:
    """Subtasks that start in free space (the first, or one on a new reference frame): trimmed to their corridor and
    reached through a rise-cross-descend connector. The others continue in contact from where the last one ended."""
    return [i == 0 or s.object_ref != subtasks[i - 1].object_ref for i, s in enumerate(subtasks)]


def plan_roll(plans, object_ref: str) -> float:
    """The planner's roll (chosen inside the J6 range) for a subtask frame."""
    first, second = plans
    return {"grasp_frame": first.grasp_tcp[5], "seat_frame": getattr(second, "seated_tcp", [0.0] * 6)[5],
            "spot_frame": getattr(second, "spot_tcp", [0.0] * 6)[5]}[object_ref]


def check_rows(rows: np.ndarray, skill: str) -> None:
    """Transformed rows obey the same motion rules as scripted ones (box, tool tilt); raises MotionError."""
    from ..skills.motion import Motion

    m = Motion(rows[0, :6], rows[0, 6], skill, "check", "world")
    for r in rows:
        m._check(r)


class Generator:
    def __init__(self, a, backend=None):
        from ..skills import record
        from ..skills.force_guard import load_limits

        self.spec = load_spec()
        self.skill, self.product, self.args = a.skill, a.product, a
        self.scenario = SCENARIO[a.skill]
        self.sk = self.spec.skills[a.skill]
        self.record = record
        self.limits = load_limits(a.limits) if a.limits else load_limits()
        self.per_object = record.per_object_limits()
        refs = [s.object_ref for s in self.sk.subtasks]
        held = [i > 0 and s.object_ref == "seat_frame" for i, s in enumerate(self.sk.subtasks)]
        self.held = held
        self.trim = free_space_starts(self.sk.subtasks)
        self.segments, self.source_counts = load_sources(a.seeds, self.sk.signals, refs, held,
                                                         cases=SOURCE_CASES[a.skill], max_seat_tries=1)
        if not self.segments[0]:
            raise SystemExit(f"no usable source demo in {len(a.seeds)} seed files: {self.source_counts}")
        self.backend = backend or self._isaac_backend(a)
        self.rng = np.random.default_rng(a.seed)
        self.gl = G.GateLimits()
        self.run_cfg = record.Run(self.limits, self.per_object, a.max_attempts, a.allow_touch_proxy,
                                  path=chunk_path(a.out, 0))

    def _isaac_backend(self, a):
        from ..skills.isaac_backend import IsaacBackend

        return IsaacBackend(self.scenario, a.product, str(chunk_path(a.out, 0)), a.device, tuple(a.cameras))

    def rotate(self) -> None:
        """Close and annotate the current recorder file and record on into the next chunk (agent 3's
        IsaacBackend.rotate), so kept demos are exportable while the worker keeps generating."""
        k = int(self.run_cfg.path.stem.rsplit("_c", 1)[1]) + 1
        nxt = chunk_path(self.args.out, k)
        self.backend.rotate(str(nxt))
        self.run_cfg.path, self.run_cfg.metas = nxt, []

    def parts(self) -> list[str]:
        """Sheet parts that have a source segment for every subtask (a part's path never transfers to another
        part's geometry)."""
        have = set.intersection(*({s.object_id for s in segs} for segs in self.segments))
        parts = [self.args.part] if self.args.part else self.backend.parts(self.product)
        usable = [p for p in parts if p in have]
        if not usable:
            raise SystemExit(f"no source demos for {parts}; sources cover {sorted(have)}")
        return usable

    # --- one subtask ----------------------------------------------------------------------------------------
    def _new_frame(self, rec, i: int, sub, plans) -> np.ndarray:
        from .mimicgen import seat_object_frame

        if self.held[i]:
            return seat_object_frame(rec.frame_pose(rec.obs, "seat_frame"), self.ep["grasp_frame"], self.ep["part0"])
        return rec.frame_pose(rec.obs, sub.object_ref)

    def subtask_rows(self, rec, i: int, sub, plans, src=None) -> tuple[list, object]:
        """Connector + transformed rows of subtask i (source chosen by nearest frame unless `src` is given)."""
        from ..skills.motion import Segment

        frame = self._new_frame(rec, i, sub, plans)
        if src is None:
            src = select_source(self.segments[i], frame, plans[0].object_id, self.spec.raw["selection"]["nn_k"],
                                self.rng)
        rows = self._rows(rec, i, sub, plans, src, frame)
        lead = connector(rec.pose, rec.gripper, rows[0], self.skill, sub, careful=not self.trim[i])
        return [*lead, Segment(self.skill, sub.name, sub.name, sub.object_ref, rows)], src

    def _rows(self, rec, i: int, sub, plans, src, frame) -> np.ndarray:
        if self.held[i]:
            rows = transform_held(src.rows, src.frame_pose, frame, src.in_hand, rec.in_hand[plans[0].object_id])
        else:
            rows = transform_rows(src.rows, src.frame_pose, frame)
        if self.trim[i]:
            rows = trim_to_corridor(rows)
            rows = align_r(rows, plan_roll(plans, sub.object_ref), contact_row(rows))
        else:
            rows = trim_after_retreat(rows) if i == len(self.sk.subtasks) - 1 else rows
            rows = align_r(rows, rec.pose[5])
        pos_mm = sub.noise_pos_m * 1000.0
        rows = taper_noise(rows, rows[contact_row(rows), :3], pos_mm, pos_mm * float(self.spec.raw["rot_deg_per_mm"]),
                           self.rng)
        check_rows(rows, self.skill)
        return rows

    def _insert(self, rec, i: int, sub, plans) -> object:
        """Held insert: carry to the top of the seat corridor, re-measure the part in the fingers (as the scripted
        insert does before entry, so carry slip is absorbed), re-transform the corridor rows, then enter."""
        from ..skills import checks
        from ..skills.motion import Segment
        from ..skills.runner import EpisodeFailed

        oid = plans[0].object_id
        segs, src = self.subtask_rows(rec, i, sub, plans)
        rec.run(segs[:-1])
        rec.in_hand[oid] = checks.in_hand(rec.obs, oid)
        rows = self._rows(rec, i, sub, plans, src, self._new_frame(rec, i, sub, plans))
        lead = connector(rec.pose, rec.gripper, rows[0], self.skill, sub, careful=True)
        rec.run([*lead, Segment(self.skill, sub.name, sub.name, sub.object_ref, rows)], soft_stop=True)
        ok, err = rec._seated(plans[1])
        if not ok:
            raise EpisodeFailed(f"not seated: { {k: round(float(v), 3) for k, v in err.items()} }")
        return src

    # --- one episode ----------------------------------------------------------------------------------------
    def _begin(self, rec, plans) -> None:
        """Same header as agent 3's episodes.<skill>: meta, instruction, Mimic frames, allowed targets."""
        from ..skills import checks, episodes
        from ..skills.geometry import pose_to_matrix

        first, second = plans
        oid = first.object_id
        start = episodes._part_xy_yaw(rec, oid)
        if self.skill == "pick_insert":
            base = checks.object_pose(rec.obs, second.base_id)
            episodes._begin(rec, self.skill, oid, second.base_id, first.grip_index, "nominal",
                            start + [first.grasp_tcp[5], base[0], base[1], base[5]])
            rec.define_frame("seat_frame", second.base_id, second.seated_tcp)
        else:  # pull_out: set down on a remembered test-bed spot
            yaw = episodes._spot_yaw(rec, oid, first.grasp_tcp[5], second)
            episodes._begin(rec, self.skill, oid, None, first.grip_index, SCENARIO[self.skill],
                            start + [first.grasp_tcp[5], second.spot_tcp[0], second.spot_tcp[1], yaw])
            rec.define_frame("spot_frame", None, second.spot_tcp)
        rec.define_frame("grasp_frame", oid, first.grasp_tcp)
        rec.targets.add(oid)
        self.ep = {"grasp_frame": pose_to_matrix(first.grasp_tcp),
                   "part0": pose_to_matrix(checks.object_pose(rec.obs, oid)), "sources": [],
                   "planned_grip": rec._planned_grip(oid, first.grasp_tcp)}

    def _grasp(self, rec, segs, plans) -> None:
        """approach_grasp: grip gates once the fingers closed and settled, then agent 3's grasp + slip check."""
        from dataclasses import replace

        from ..skills import checks, gates
        from ..skills.runner import EpisodeFailed

        pick = plans[0]
        oid = pick.object_id
        planned = self.ep["planned_grip"]
        head, tail = split_after_close(segs[-1].rows)
        rec.run([*segs[:-1], replace(segs[-1], rows=head)])
        gates.grip_gates(rec.gate_log, rec.obs, oid, planned, pick.grasp_tcp[5])
        before = rec.obs
        rec.run([replace(segs[-1], rows=tail)])
        ok, why = checks.grasp_ok(before, rec.obs, oid)
        if not ok:
            raise EpisodeFailed(f"grasp check: {why}")
        rec.in_hand[oid] = checks.in_hand(rec.obs, oid)
        rec.holding = oid

    def _run_subtasks(self, rec, plans) -> None:
        from ..skills.runner import EpisodeFailed

        for i, sub in enumerate(self.sk.subtasks):
            src = self._subtask(rec, i, sub, plans)
            self.ep["sources"].append(src.demo)
            if sub.term_signal:
                rec.mark(sub.term_signal)
        self._end_gates(rec, plans)
        rec.go_home()
        rec.close()
        rec.finish_gates()
        drift = rec.in_hand_drift_mm["carry"]
        rec.gate_log.check("carry_drift_mm", drift, MAX_CARRY_DRIFT_MM)
        rec.gate_log.require(("carry_drift_mm",))  # the seed filter, applied to generated demos too
        missing = [s for s in self.sk.signals if s not in rec.marks]
        if missing:
            raise EpisodeFailed(f"episode ended without the subtask signals {missing}")

    def _subtask(self, rec, i: int, sub, plans):
        """Run subtask i with the closed-loop checks agent 3's runner makes at the same signal."""
        from ..skills import checks, gates
        from ..skills.runner import EpisodeFailed

        oid = plans[0].object_id
        if sub.term_signal == "seated":
            return self._insert(rec, i, sub, plans)
        segs, src = self.subtask_rows(rec, i, sub, plans)
        if sub.term_signal == "grasped" and self.skill == "pick_insert":
            self._grasp(rec, segs, plans)
        elif sub.term_signal == "grasped":  # pull_out: grip read on the installed part once closed, before the pull
            from dataclasses import replace

            head, tail = split_after_close(segs[-1].rows)
            rec.run([*segs[:-1], replace(segs[-1], rows=head)])
            gates.grip_gates(rec.gate_log, rec.obs, oid, self.ep["planned_grip"], plans[0].grasp_tcp[5])
            self.ep["before_pull"] = rec.obs
            if len(tail):
                rec.run([replace(segs[-1], rows=tail)])
        elif sub.term_signal == "pulled":
            rec.run(segs)
            ok, why = checks.grasp_ok(self.ep["before_pull"], rec.obs, oid)
            if not ok:
                raise EpisodeFailed(f"pull-out: {why}")
            rec.in_hand[oid] = checks.in_hand(rec.obs, oid)
            rec.holding = oid
        elif sub.term_signal == "placed":
            self.ep["expected"] = checks.expected_object(plans[1].spot_tcp, checks.in_hand(rec.obs, oid))
            rec.run(segs, soft_stop=True)  # as the scripted set-down: release where it touched
        else:
            rec.run(segs)
        return src

    def _end_gates(self, rec, plans) -> None:
        """The skill's final placement gates, read after the release (as agent 3's episodes do)."""
        from ..skills import checks, frames, gates

        first, second = plans
        actual = checks.object_pose(rec.obs, first.object_id)
        if self.skill == "pick_insert":
            gates.seat_gates(rec.gate_log, actual, second.seated_object_pose, second.yaw_tol_deg,
                             frames.installed_up(first.object_id))
        else:
            gates.spot_gates(rec.gate_log, actual, self.ep["expected"])

    def every_step_others(self, rec, oid: str) -> G.GateReport:
        """mimic/gates.py: no non-target part moved > 1 mm / 2 deg at ANY step (bumped and settled back fails too)."""
        rep = G.GateReport(ok=False)
        obs_all = [f.obs for f in rec.frames] + [rec.obs]
        uids = obs_all[0]["sim"]["object_poses"].keys()
        traces = {u: np.array([o["sim"]["object_poses"][u] for o in obs_all], dtype=float) for u in uids}
        G.gate_others(rep, traces, oid, self.gl)
        return G.finish(rep)

    def attempt(self, part: str) -> dict:
        """One attempt. Returns {"status": kept|failed|reset_invalid|no_case, "reason", "values", "sources"}."""
        from ..skills.backends import NoCase
        from ..skills.force_guard import ForceGuard
        from ..skills.gates import GateFailed
        from ..skills.motion import MotionError
        from ..skills.runner import EpisodeFailed, EpisodeRecorder, ResetInvalid

        rec, R = None, self.record
        self.ep = {}
        installed = R._installed_before(self.backend, self.product, part)
        if self.scenario == "fix_above":  # the part to remove is installed, with everything below it
            from ..skills.backends import base_of

            installed[part] = base_of(self.product)
        try:
            obs = R._reset(self.backend, self.rng, self.product, self.scenario, installed, part)
            rec = EpisodeRecorder(self.backend.env, ForceGuard(self.limits, per_object=self.per_object), obs,
                                  allow_touch_proxy=self.args.allow_touch_proxy)
            self._begin(rec, self.backend.plans)
            self._run_subtasks(rec, self.backend.plans)
            extra = self.every_step_others(rec, part)
            if not extra.ok:
                raise EpisodeFailed(f"every-step others: {extra.failed} {extra.values}")
        except NoCase as e:
            return {"status": "no_case", "reason": str(e)}
        except ResetInvalid as e:
            self._close_episode(rec, False, f"reset_invalid: {e}")
            return {"status": "reset_invalid", "reason": str(e)}
        except (EpisodeFailed, GateFailed, MotionError, ValueError) as e:  # env TcpError is a ValueError
            self._close_episode(rec, False, str(e))
            return {"status": "failed", "reason": f"{type(e).__name__}: {e}",
                    "values": rec.gate_log.values if rec else {},
                    "sources": self.ep.get("sources", [])}
        self._close_episode(rec, True, "ok")
        return {"status": "kept", "reason": "ok", "values": rec.gate_log.values, "sources": self.ep["sources"]}

    def _close_episode(self, rec, ok: bool, reason: str) -> None:
        if rec is None:
            self.backend.finish(False)
            return
        rec.meta = {**rec.meta, "generator": "mimicgen", "mimic_sources": json.dumps(self.ep.get("sources", []))}
        self.record._finish(self.backend, self.run_cfg, rec, ok, self.scenario, reason)


def chunk_path(out: Path, k: int) -> Path:
    """Recorder file k of a worker: <stem>_c<k>.hdf5 next to <out> (stats and rejects stay at <out>)."""
    return out.with_name(f"{out.stem}_c{k:03d}{out.suffix}")


def reason_key(reason: str) -> str:
    """Stats bucket: the reason without numbers, vectors and step indices (the full text is in rejects.jsonl)."""
    key = re.sub(r"\[.*", "", reason.split(" at step")[0])
    key = re.sub(r"[-+]?\d+(\.\d+)?", "#", key)
    return re.sub(r"\s+", " ", key)[:90].strip()


def run(a) -> dict:
    a.out = a.out.expanduser()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    gen = Generator(a)
    parts = gen.parts()
    stats = {"skill": a.skill, "product": a.product, "attempts": 0, "accepted": 0, "reset_invalid": 0,
             "no_case": 0, "failed_gates": {}, "sources": gen.source_counts, "out": str(a.out)}
    t0, k = time.time(), 0
    try:
        with open(f"{a.out}.rejects.jsonl", "a") as rej:
            while stats["accepted"] < a.target and stats["attempts"] < a.max_attempts:
                part = parts[k % len(parts)]
                k += 1
                t1 = time.time()
                res = gen.attempt(part)
                if res["status"] in ("no_case", "reset_invalid"):
                    stats[res["status"]] += 1
                    if stats[res["status"]] > 5 * a.max_attempts:
                        stats["aborted"] = f"too many {res['status']}: {res['reason']}"
                        break
                    continue
                stats["attempts"] += 1
                if res["status"] == "kept":
                    stats["accepted"] += 1
                else:
                    key = reason_key(res["reason"])
                    stats["failed_gates"][key] = stats["failed_gates"].get(key, 0) + 1
                rej.write(json.dumps({"part": part, "s": round(time.time() - t1, 1), **res}, default=str) + "\n")
                rej.flush()
                _write_stats(a.out, stats, t0)
                if a.rotate_every and stats["attempts"] % a.rotate_every == 0:
                    gen.rotate()
    finally:
        _write_stats(a.out, stats, t0)
        gen.backend.close(gen.run_cfg.path)
    return stats


def _write_stats(out: Path, stats: dict, t0: float) -> None:
    stats["wall_s"] = round(time.time() - t0, 1)
    stats["acceptance_rate"] = round(stats["accepted"] / stats["attempts"], 4) if stats["attempts"] else None
    stats["demos_per_hour"] = round(stats["accepted"] / stats["wall_s"] * 3600, 2) if stats["wall_s"] else None
    Path(f"{out}.stats.json").write_text(json.dumps(stats, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skill", required=True, choices=sorted(SCENARIO))
    ap.add_argument("--product", type=int, required=True)
    ap.add_argument("--part", default=None, help="only this part (default: every sheet part in turn)")
    ap.add_argument("--seeds", type=Path, nargs="+", required=True, help="agent 3's kept recorder HDF5 files")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target", type=int, default=200)
    ap.add_argument("--max-attempts", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rotate-every", type=int, default=1, help="close the recorder file every N attempts (0: never)")
    ap.add_argument("--cameras", nargs="*", default=["table", "side", "wrist"], choices=("table", "side", "wrist"))
    ap.add_argument("--device", default="cuda:0", help="physics device: cuda:0, or cpu (one env: often faster)")
    ap.add_argument("--limits", type=Path, default=None, help="override physics/limits.yaml")
    ap.add_argument("--allow-touch-proxy", action="store_true",
                    help="accept sim.gripper_obj as the touch gate when the env has no sim.robot_touch (stamped)")
    print(json.dumps(run(ap.parse_args()), indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
