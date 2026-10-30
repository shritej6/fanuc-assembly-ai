"""Re-render agent 3's kept source demos with the 3 cameras on: replay each demo's recorded actions 1:1 from its
recorded initial state in N batched cells, and keep it only if every recorded gate still holds on the replay.

    GPU=0 CACHE_TAG=_ne REPO=~/t12_data/repo_ne1r sim/assembly_lab/env/isaaclab_run.sh \
        sim/assembly_lab/env/replay_cams.py --product 1 --skill pick_insert --num_envs 16 \
        --files ~/t12_data/demos/pick_insert/product_1_build_*.hdf5 --out_dir ~/t12_data/r1_replay

All files of one process must share one product (the scene pool) and skill. Each lane resets with reset_to(initial
state, is_relative=True) and steps its recorded actions; when a lane finishes it is reset to the next demo while the
others keep stepping (no physics step happens during a reset). Observations are taken before each step, like the
Isaac Lab recorder, from the env's policy group (joints, tcp_pose, eef, frames, sim_contact_peak, 3 cameras);
plan_ids / plan_target / train_mask are copied from the source.

Keep rule (no gate loosened). The replay is compared with the recording at every step: dobj = largest object position
deviation (mm) / rotation deviation (deg), deef = largest TCP position deviation (mm). Each recorded gate value v with
limit l must still pass after adding the most the deviation can change it:
  - seat_* and position gates: v + dobj (or deg) <= l
  - tracking_mm: v + deef <= l
  - grip_pad_mm: v + dobj + deef <= l
  - other_moved_*: v + 2 dobj <= l
  - in_hand_drift: the replay's own largest target move in the TCP frame, grasp to release, <= 0.5 mm
  - any gate not listed: dobj <= 0.1 mm and 0.1 deg
Forces are re-checked on the replayed contact trace (force_guard.check_trace, the exporter's check), and the robot
touch on every non-target object is re-measured (<= the recorded touch limit). One HDF5 per kept demo:
<out_dir>/<skill>/<source stem>_<demo key>.hdf5 with data/demo_0, written under <out_dir>/_writing and moved in when
closed (listed in <out_dir>/closed.txt); every decision goes to <out_dir>/report_<tag>.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--product", type=int, required=True)
ap.add_argument("--skill", required=True)
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--num_envs", type=int, default=16)
ap.add_argument("--max_demos", type=int, default=0)
ap.add_argument("--out_dir", type=Path, required=True)
ap.add_argument("--tag", default="")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import h5py
import layout as L
from cell_env import AssemblyCellEnv, CellEnvCfg

from sim.assembly_lab.mimic.force_guard import check_trace, load_limits

FRAME_NAMES = ("grasp_frame", "seat_frame", "spot_frame")
COPY_OBS = ("plan_ids", "plan_target", "train_mask")
CAMS = ("table", "side", "wrist")
DRIFT_LIMIT_MM = 0.5
DRIFT_MARGIN_MM = 0.01   # the recomputed drift matched the recorded one to 0.003 mm (offline check): stay stricter
TIGHT = (0.1, 0.1)


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes")


def read(g) -> dict:
    return {k: (read(v) if isinstance(v, h5py.Group) else v[()]) for k, v in g.items()}


def load_jobs(files: list[str]) -> list[dict]:
    jobs = []
    for path in files:
        with h5py.File(path, "r") as f:
            for key, demo in f["data"].items():
                if not _truthy(demo.attrs.get("kept", False)):
                    continue
                if (args.out_dir / args.skill / f"{Path(path).stem}_{key}.hdf5").exists():
                    continue                     # kept by an earlier run
                jobs.append({"file": path, "key": key, "actions": demo["actions"][()],
                             "init": read(demo["initial_state"]), "states": read(demo["states"]),
                             "obs_src": {k: demo["obs"][k][()] for k in demo["obs"]},
                             "attrs": {k: demo.attrs[k] for k in demo.attrs}})
    return jobs


def quat_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(np.sum(a * b, axis=-1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(d))


class Lane:
    def __init__(self, job: dict, out: Path, obs0: dict, k: int):
        self.job, self.t, self.n = job, 0, len(job["actions"])
        self.name = f"{Path(job['file']).stem}_{job['key']}.hdf5"
        self.path = out / "_writing" / self.name
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = h5py.File(self.path, "w")
        self.g = self.f.create_group("data").create_group("demo_0")
        self.obs = self.g.create_group("obs")
        for name, t in obs0.items():
            shape = (self.n, *t.shape[1:])
            chunk = (1, *t.shape[1:]) if name.endswith("_cam") else None
            self.obs.create_dataset(name, shape=shape, dtype=t.dtype, chunks=chunk,
                                    compression="lzf" if name.endswith("_cam") else None)
        self.dev = {"obj_mm": 0.0, "obj_deg": 0.0, "target_mm": 0.0, "eef_mm": 0.0}
        self.tdev = np.zeros((self.n, 2))      # target position (mm) / rotation (deg) deviation from the recording
        self.odev = np.zeros((self.n, 2))      # same, worst non-target object
        self.p0 = self.pf = None               # UF0 mm poses after reset / at the end (exact seat_xy)
        self.in_hand = np.zeros((self.n, 3))   # target position in the TCP frame (mm) after each step
        self.touch_other = 0.0
        self.contact = np.zeros((self.n, 4), dtype=np.float32)

    def record_obs(self, obs: dict, k: int) -> None:
        for name, t in obs.items():
            self.obs[name][self.t] = t[k]
        eef = self.job["obs_src"].get("eef_pos")
        if eef is not None:
            self.dev["eef_mm"] = max(self.dev["eef_mm"], float(np.linalg.norm(obs["eef_pos"][k] - eef[self.t])) * 1e3)


def bound_gates(lane: Lane, limits, solver) -> tuple[bool, list[str]]:
    """Every recorded gate re-evaluated on the replay (see the module docstring)."""
    a, d = lane.job["attrs"], lane.dev
    marks = json.loads(a.get("marks", "{}"))
    lo = int(marks.get("grasped", 0))
    hi = int(marks.get("seated", marks.get("placed", marks.get("dropped", lane.n))))   # in hand until release
    win = lane.tdev[lo:max(lo + 1, hi)]
    t_win_mm, t_fin_mm, t_fin_deg = float(win[:, 0].max()), float(lane.tdev[-1, 0]), float(lane.tdev[-1, 1])
    fails, exact = [], {}
    oid, base = str(a.get("object_id", "")), str(a.get("base_id", ""))
    gates = json.loads(a.get("gates", "{}"))
    if "seat_xy_mm" in gates and base:
        seat = L.pose6_from_tf(solver.seat_tf_for(oid, base, lane.p0))
        fin = L.pose6_from_tf(lane.pf[oid])
        exact["seat_xy_mm"] = float(np.hypot(fin[0] - seat[0], fin[1] - seat[1]))
    for name, g in gates.items():
        v, lim = abs(float(g.get("value", 0.0))), g.get("limit")
        if lim is None:
            continue
        lim = float(lim)
        if name in exact:
            worst = exact[name]
        elif name in ("seat_z_mm", "spot_xy_mm"):
            worst = v + t_fin_mm
        elif name in ("seat_yaw_deg", "seat_tilt_deg", "spot_yaw_deg", "spot_tilt_deg"):
            worst = v + t_fin_deg
        elif name == "grip_roll_deg":
            worst = v + float(lane.tdev[lo, 1])
        elif name == "grip_pad_mm":
            worst = v + float(lane.tdev[lo, 0]) + d["eef_mm"]
        elif name == "grip_holds_target":
            worst, lim = t_win_mm, 1.0          # held: the target followed its recorded in-hand path (grip_pad limit)
        elif name == "tracking_mm":
            worst = v + d["eef_mm"]
        elif name == "other_moved_mm":
            worst = v + 2 * float(lane.odev[:, 0].max())
        elif name == "other_moved_deg":
            worst = v + 2 * float(lane.odev[:, 1].max())
        elif name == "touch_non_target_N":
            worst = lane.touch_other
        else:
            worst, lim = t_fin_mm, TIGHT[0]     # a gate this replay cannot recompute: the outcome must match
        if worst > lim + 1e-9:
            fails.append(f"{name}: {worst:.4f} > {lim}")
    if json.loads(a.get("in_hand_drift_mm", "{}")):
        # the replay's own in-hand drift: largest move of the target in the TCP frame from the grasp to the release,
        # a superset of the recorded carry / contact drift windows, against the same limit
        rel = lane.in_hand[lo:max(lo + 1, hi)]
        drift = float(np.linalg.norm(rel - rel[0], axis=1).max())
        exact["in_hand_drift_mm"] = drift
        if drift + DRIFT_MARGIN_MM > DRIFT_LIMIT_MM:
            fails.append(f"drift grasp..release: {drift:.4f} > {DRIFT_LIMIT_MM}")
    guard = check_trace(lane.contact[:, :3], limits)
    if not guard.ok:
        fails.append(f"force guard: {guard}")
    lane.exact = {k: round(v, 4) for k, v in exact.items()}
    lane.window = {"grasped": lo, "seated": hi, "target_win_mm": round(t_win_mm, 4), "target_final_mm":
                   round(t_fin_mm, 4), "target_final_deg": round(t_fin_deg, 4)}
    return not fails, fails


def finish(lane: Lane, limits, report, solver) -> bool:
    ok, fails = bound_gates(lane, limits, solver)
    j = lane.job
    if ok:
        g = lane.g
        g.create_dataset("actions", data=j["actions"])
        g.create_dataset("processed_actions", data=j["actions"])
        for name in COPY_OBS:
            if name in j["obs_src"]:
                lane.obs.create_dataset(name, data=j["obs_src"][name])
        ini = g.create_group("initial_state")

        def put(grp, d):
            for k, v in d.items():
                put(grp.create_group(k), v) if isinstance(v, dict) else grp.create_dataset(k, data=v)
        put(ini, j["init"])
        for k, v in j["attrs"].items():
            g.attrs[k] = v
        g.attrs["kept"] = "True"
        g.attrs["success"] = True
        g.attrs["num_samples"] = lane.n
        g.attrs["replay_of"] = f"{j['file']}#{j['key']}"
        g.attrs["replay_set"] = "r1_replay"
        g.attrs["env_manifest_md5"] = (L.REPO / "ENV_MANIFEST_MD5").read_text().strip()
        g.attrs["generator"] = "r1_replay"
        g.attrs["steps"] = lane.n
        g.attrs["replay_dev"] = json.dumps({**{k: round(v, 4) for k, v in lane.dev.items()}, **lane.window,
                                            **{f"replay_{k}": v for k, v in lane.exact.items()}})
        lane.f["data"].attrs["env_args"] = json.dumps({"env_name": "", "type": 2, "sim_args": {
            "dt": 0.002, "decimation": 50, "render_interval": 50, "num_envs": 1}})
        lane.f["data"].attrs["total"] = lane.n
        lane.f.close()
        final = args.out_dir / args.skill / lane.name                # only closed, kept files enter the tree
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(lane.path, final)
        with (args.out_dir / "closed.txt").open("a") as c:
            c.write(f"{final}\n")
    else:
        lane.f.close()
        lane.path.unlink()
    rec = {"file": j["file"], "key": j["key"], "steps": lane.n, "kept": ok, "fails": fails,
           "dev": {k: round(v, 4) for k, v in lane.dev.items()}, "touch_other_N": round(lane.touch_other, 4),
           "peak": lane.contact.max(axis=0).round(4).tolist(), "window": lane.window, "exact": lane.exact}
    report.write(json.dumps(rec) + "\n")
    report.flush()
    print(f"[replay] {Path(j['file']).name}#{j['key']} kept={ok} dev={rec['dev']} {fails[:2]}", flush=True)
    return ok


def stack_state(states: list[dict]) -> dict:
    import torch

    out = {}
    for k in states[0]:
        if isinstance(states[0][k], dict):
            out[k] = stack_state([s[k] for s in states])
        else:
            out[k] = torch.tensor(np.concatenate([s[k] for s in states]), dtype=torch.float32, device=args.device)
    return out


def obs_np(env) -> dict:
    pol = env.env.obs_buf["policy"]
    return {k: v.cpu().numpy() for k, v in pol.items()}


def main() -> None:
    import torch

    jobs = load_jobs(args.files)
    if args.max_demos:
        jobs = jobs[:args.max_demos]
    pool = sorted(k for k in jobs[0]["init"]["rigid_object"] if k not in FRAME_NAMES)
    jobs = [j for j in jobs if sorted(k for k in j["init"]["rigid_object"] if k not in FRAME_NAMES) == pool]
    print(f"[replay] {len(jobs)} kept demos, pool {pool}", flush=True)
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=pool, cameras=CAMS, skill=args.skill,
                                     num_envs=args.num_envs))
    e = env.env
    e.cfg.rerender_on_reset = True
    limits = load_limits(L.REPO / "sim" / "assembly_lab" / "physics" / "limits.yaml")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = (args.out_dir / f"report_{args.tag or f'p{args.product}_{args.skill}'}.jsonl").open("a")
    by_name = dict(zip(e.names, e.pool))
    lanes: dict[int, Lane] = {}
    queue, kept, t0, steps = list(jobs), 0, time.time(), 0

    def start(ids: list[int]) -> None:
        nonlocal queue
        take = [(k, queue.pop(0)) for k in ids if queue]
        if not take:
            return
        env_ids = torch.tensor([k for k, _ in take], device=e.device)
        e.reset_to(stack_state([j["init"] for _, j in take]), env_ids, is_relative=True)
        e.sim.render()
        e.scene.update(dt=e.physics_dt)
        e.obs_buf = e.observation_manager.compute()
        o = obs_np(env)
        for k, j in take:
            lanes[k] = Lane(j, args.out_dir, o, k)
            lanes[k].p0 = e.object_poses_mm(k)

    start(list(range(args.num_envs)))
    while lanes:
        o = obs_np(env)
        act = np.concatenate([env.term.tcp_cmd, env.term.grip_cmd[:, None]], axis=1).astype(np.float32)
        for k, lane in lanes.items():
            lane.record_obs(o, k)
            act[k] = lane.job["actions"][lane.t]
        e._metrics.reset_chunk(sorted(lanes))
        e.obs_buf, *_ = e.step(torch.tensor(act, device=e.device))
        steps += 1
        state = e.scene.get_state(is_relative=True)["rigid_object"]
        done = []
        for k, lane in lanes.items():
            j, t = lane.job, lane.t
            tgt = j["attrs"].get("object_id", "")
            for name in pool:
                rp = state[name]["root_pose"][k].cpu().numpy()
                sp = j["states"]["rigid_object"][name]["root_pose"][t]
                dmm = float(np.linalg.norm(rp[:3] - sp[:3])) * 1e3
                ddeg = float(quat_deg(rp[3:], sp[3:]))
                lane.dev["obj_mm"], lane.dev["obj_deg"] = max(lane.dev["obj_mm"], dmm), max(lane.dev["obj_deg"], ddeg)
                if by_name.get(name, name) == tgt:
                    lane.dev["target_mm"] = max(lane.dev["target_mm"], dmm)
                    lane.tdev[t] = (dmm, ddeg)
                else:
                    lane.odev[t] = np.maximum(lane.odev[t], (dmm, ddeg))
            lane.contact[t] = e.obs_buf["policy"]["sim_contact_peak"][k].cpu().numpy()
            if tgt in pool:
                tcp6 = e.obs_buf["policy"]["tcp_pose"][k].cpu().numpy().astype(float)
                tcp_tf = L.translation(*tcp6[:3])
                tcp_tf[:3, :3] = L.fanuc_rotation(*tcp6[3:6])
                rp = state[tgt]["root_pose"][k].cpu().numpy().astype(float)
                part = np.eye(4)
                part[:3, :3], part[:3, 3] = L.rotation_from_quat_xyzw([*rp[4:7], rp[3]]), rp[:3]
                lane.in_hand[t] = (np.linalg.inv(tcp_tf) @ L.world_m_to_uf0(part, e._top()))[:3, 3]
            touch = e._metrics.chunk(k)["robot_touch"]
            other = [v for n, v in touch.items() if by_name.get(n, n) != tgt]
            lane.touch_other = max(lane.touch_other, max(other, default=0.0))
            lane.t += 1
            if lane.t >= lane.n:
                lane.pf = e.object_poses_mm(k)
                done.append(k)
        for k in done:
            kept += finish(lanes.pop(k), limits, report, env.solver)
        if done:
            start(done)
        if steps % 50 == 0:
            el = time.time() - t0
            print(f"[replay] {steps} steps, {el:.0f} s, {len(queue)} queued, {len(lanes)} live, kept {kept}", flush=True)
    print(f"[replay] DONE kept {kept}/{len(jobs)} in {time.time() - t0:.0f} s, {steps} process steps", flush=True)
    report.close()
    env.close()


if __name__ == "__main__":
    main()
