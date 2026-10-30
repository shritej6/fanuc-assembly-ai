"""Run the GR00T policy on every eval scenario in the Isaac Lab cell; write KPIs JSON and short videos.

    # on the sim box, GR00T server reachable on :5555 (agent 5's serve_policy.sh, or an ssh tunnel)
    CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p sim/assembly_lab/eval/run_eval.py --run groot_e1 \
        --policy groot://127.0.0.1:5555 [--only nominal_M4 crowded] [--layouts 2]
    python -m sim.assembly_lab.eval.run_eval --run check --dry      # host: layouts only, checked by the env's solver

Output: ~/t12_data/eval/<run>/results.json (per episode, per scenario, per product, per case) and videos/*.mp4.
Every chunk goes through policy_learn/deploy/guards.py exactly as on the real arm. A scenario whose requirement the
env or director lacks is SKIPPED with the reason; skips never count as passes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from policy_learn.deploy.groot_policy import GrootPolicy
from sim.assembly_lab.eval import cad, kpis, layouts
from sim.assembly_lab.eval.director import Director

OUT_ROOT = Path.home() / "t12_data" / "eval"
# The real cell's home (robot/cell/mock_cell.py HOME, "same API, limits and timing as the real cell"): TCP about
# (300, 0, -90) with the fingers straight down. The sim default (scene_config HOME_POSE_DEG) puts the TCP at about
# (87, 0, -216) tilted 50 deg, outside the workspace box, so every episode would start from a pose the real arm never has.
CELL_HOME_DEG = {"J1": -0.0065, "J2": 3.2909, "J3": -11.8967, "J4": -0.0007, "J5": -74.8218, "J6": -0.0286}


def policy_obs(obs: dict) -> dict:
    """What the policy may see: never obs["sim"]."""
    return {k: obs[k] for k in ("images", "joints", "tcp_pose", "instruction", "depth") if k in obs}


def video_backend() -> str | None:
    """cv2 if importable (mp4), else None: videos are then skipped and the report says so (KPIs are unaffected)."""
    try:
        import cv2  # noqa: F401
    except ImportError:
        return None
    return "cv2"


class Video:
    def __init__(self, path: Path, fps: float):
        self.path, self.fps, self.writer = path, fps, None

    def add(self, obs: dict) -> None:
        import cv2

        imgs = obs["images"]
        frame = np.hstack([imgs[c] for c in ("table", "wrist") if c in imgs])
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()


PROGRESS: list[Path] = []


def progress(line: str) -> None:
    """Kit buffers stdout until exit, so progress also goes to <out>/progress.log, flushed per line."""
    print(line, flush=True)
    for path in PROGRESS:
        with path.open("a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {line}\n")


def missing_requirement(ev_reqs: list[str], env, custom_director: bool) -> str | None:
    for req in ev_reqs:
        kind, _, name = req.partition(":")
        if kind == "env" and not hasattr(env, name):
            return f"env has no {name}() (agent 1)"
        if kind == "director" and not custom_director:
            return "planner-only case: needs the T09 state machine as --director (the oracle would pass it trivially)"
    return None


def seat_gap(obs, task) -> dict | None:
    """Progress only: the part's error against its seated pose (pick_insert), rounded."""
    poses = obs["sim"]["object_poses"]
    if task.skill != "pick_insert" or task.uid not in poses or task.base not in poses:
        return None
    seated = cad.seated_pose(cad.type_id(task.uid), task.base, np.asarray(poses[task.base], float))
    return {k: round(float(v), 2) for k, v in cad.seat_check(task.uid, poses[task.uid], seated)[1].items()}


def run_episode(env, policy: GrootPolicy, ev, limit: float, limits: dict, director, video: Video | None) -> dict:
    t0 = time.time()
    obs = env.reset(ev.layout)
    policy.reset()
    director.reset(obs)
    kpi = kpis.EpisodeKPI(limits, {u: list(p) for u, p in obs["sim"]["object_poses"].items()})
    t_s, dt, acts = 0.0, policy.DT, 0
    while t_s < limit:
        n_events = len(director.events)
        task = director.step(obs, t_s)
        for e in director.events[n_events:]:
            progress(f"[{ev.scenario} {ev.seed}] sim {t_s:.1f} s: " + " ".join(
                f"{k}={e[k]}" for k in ("type", "skill", "uid", "attempt", "grip_error", "jam", "reason") if k in e))
        if task is None:
            break
        policy.set_task(task.instruction, director.contact_target())
        rows = policy.act(policy_obs(obs))
        acts += 1
        obs = env.step_tcp(rows, dt)  # one observation per chunk, as on the real cell (video: one frame per chunk)
        t_s += len(rows) * dt
        kpi.on_step(obs, task, t_s)
        if video:
            video.add(obs)
        if acts % 10 == 0 or acts <= 3 or (policy.guard_log and policy.guard_log[-1]["held"] and acts < 30):
            gap = seat_gap(obs, task)
            progress(f"[{ev.scenario} {ev.seed}] act {acts}: sim {t_s:.1f} s, wall {time.time() - t0:.0f} s, "
                     f"step {task.skill} {task.uid} grasped={task.grasped} held={cad.held_uid(obs)} "
                     f"tcp={[round(v, 1) for v in obs['tcp_pose'][:3]]} seat_err={gap}, "
                     f"guard held {sum(e['held'] for e in policy.guard_log)}, rows {len(rows)}, "
                     f"last guard {policy.guard_log[-1]['reasons'][:2] if policy.guard_log else None}")
    if video:
        video.close()
    poses = obs["sim"]["object_poses"]
    gates = kpis.final_checks(ev.expect, poses, kpi.initial, director.events)
    for e in director.events:  # every insert attempt that ended unseated: was the part jammed on its mate?
        if e["type"] in ("step_timeout",) and e["skill"] == "pick_insert" and e.get("jam"):
            kpi.jams.append(e["jam"])
    for prod, base in ev.expect["built"].items():
        for pid in cad.product_parts(prod):
            for uid in (u for u in poses if cad.type_id(u) == pid):
                j = kpis.jam_check(uid, base, poses)
                if j and j not in kpi.jams:
                    kpi.jams.append(j)
    k = kpi.summary()
    measured_ok = all(g["ok"] for g in gates.values())
    timed_out = t_s >= limit and (director.current is not None or bool(director.queue))
    held = [e for e in policy.guard_log if e["rollout"] == policy.rollout]
    success = (measured_ok and not director.failed and not timed_out and k["parts_damaged"] == 0
               and k["collisions"] == 0)
    return {"scenario": ev.scenario, "case": ev.case, "seed": ev.seed, "success": success,
            "success_note": None if k["robot_contacts"] != "UNMEASURED" else
            "arm/finger touch on non-targets UNMEASURED (env gives no sim.robot_touch); disturbance gate applied",
            "sim_time_s": round(t_s, 1), "wall_s": round(time.time() - t0, 1), "acts": acts,
            "failed": director.failed, "timed_out": timed_out, "gates": gates, "kpi": k,
            "guard": {"held": sum(e["held"] for e in held), "clamped": sum(not e["held"] for e in held),
                      "first": held[:3]},
            "events": director.events, "plan_source": director.plan_source, "video": str(video.path) if video else None}


def rate(rows: list[dict]) -> dict:
    ran = [r for r in rows if not r.get("skipped")]
    ok = sum(r["success"] for r in ran)
    return {"pass": ok, "ran": len(ran), "skipped": len(rows) - len(ran),
            "rate": round(ok / len(ran), 3) if ran else None,
            "jams": sum(len(r["kpi"]["jams"]) for r in ran if "kpi" in r),
            "damaged": sum(r["kpi"]["parts_damaged"] for r in ran if "kpi" in r),
            "collisions": sum(r["kpi"]["collisions"] for r in ran if "kpi" in r)}


def aggregate(results: list[dict], products_of: dict) -> dict:
    by_s, by_c, by_p = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in results:
        by_s[r["scenario"]].append(r)
        by_c[r["case"]].append(r)
        for p in products_of[r["scenario"]]:
            by_p[p].append(r)
    return {"per_scenario": {k: rate(v) for k, v in by_s.items()}, "per_case": {k: rate(v) for k, v in by_c.items()},
            "per_product": {k: rate(v) for k, v in sorted(by_p.items())}, "overall": rate(results)}


def make_director(spec: str | None, ev, task_time_s: float):
    if not spec or spec == "oracle":
        return Director.for_layout(ev, task_time_s)
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls).for_layout(ev, task_time_s)


def make_policy(spec: str, holder: dict) -> GrootPolicy:
    """groot://host:port | stub | scripted (SIM-ONLY oracle baseline that reads the director's targets)."""
    if spec == "scripted":
        from sim.assembly_lab.eval.scripted import ScriptedOracleModel

        return GrootPolicy(model=ScriptedOracleModel(lambda: holder["director"]))
    return GrootPolicy(spec)


def dry(evs: list) -> int:
    """Host check: every layout is accepted by the env's own LayoutSolver (no Isaac)."""
    sys.path.insert(0, str(REPO / "sim/assembly_lab/env"))
    import layout as env_layout

    solver = env_layout.LayoutSolver()
    pool = env_layout.default_pool(solver.geometry, 1)
    for ev in evs:
        solver.solve(ev.layout, pool, -400.0)
    print(f"{len(evs)} layouts accepted by env/layout.py LayoutSolver")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--policy", help="groot://host:port, scripted (sim-only oracle baseline) or stub (plumbing)")
    ap.add_argument("--only", nargs="*", help="scenario ids or cases")
    ap.add_argument("--layouts", type=int, help="layouts per scenario (default: scenarios.yaml)")
    ap.add_argument("--director", default="oracle", help="oracle, or module:Class for the T09 state machine")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--time-limit", type=float,
                    help="cap each episode at this many sim seconds (only ever stricter than scenarios.yaml)")
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--dry", action="store_true", help="host: generate layouts and check them, no sim")
    ap.add_argument("--no-cameras", action="store_true",
                    help="render no cameras (much faster); only for --policy scripted / stub, which ignore images")
    args = ap.parse_args()
    raw, scs = layouts.load()
    evs = [layouts.build(raw, sc, k) for sc in scs if not args.only or sc.id in args.only or sc.case in args.only
           for k in range(args.layouts or sc.layouts)]
    out = args.out / args.run
    out.mkdir(parents=True, exist_ok=True)
    PROGRESS.append(out / "progress.log")
    (out / "layouts.json").write_text(json.dumps([ev.__dict__ for ev in evs], indent=1))
    if args.dry:
        return dry(evs)
    if not args.policy:
        ap.error("--policy is required for a sim run")
    from sim.assembly_lab.env.cell_env import AssemblyCellEnv, CellEnvCfg

    if args.no_cameras and args.policy.startswith("groot://"):
        ap.error("GR00T needs the cameras: --no-cameras is only for scripted / stub")
    cams = () if args.no_cameras else tuple(GrootPolicy.CAMERAS)
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, cameras=cams,
                                     image_size=GrootPolicy.IMAGE_SIZE, home_deg=dict(CELL_HOME_DEG)))
    videos_on = video_backend() is not None and not args.no_cameras
    if not videos_on:
        print("videos disabled: no cv2 in this Python (KPIs still measured)", flush=True)
    holder: dict = {}
    policy, limits = make_policy(args.policy, holder), kpis.load_limits()
    req = {sc.id: sc.requires for sc in scs}
    time_limit = {sc.id: min(sc.time_limit_s, args.time_limit or sc.time_limit_s) for sc in scs}
    products_of = {sc.id: sc.products for sc in scs}
    results, seen = [], defaultdict(int)
    for ev in evs:
        why = missing_requirement(req[ev.scenario], env, args.director != "oracle")
        if why:
            results.append({"scenario": ev.scenario, "case": ev.case, "seed": ev.seed, "skipped": why,
                            "success": False})
            continue
        vid = Video(out / "videos" / f"{ev.scenario}_{ev.seed}.mp4", 2.0) \
            if videos_on and seen[ev.scenario] < int(raw["video_layouts"]) else None
        seen[ev.scenario] += 1
        director = holder["director"] = make_director(args.director, ev, float(raw["task_time_s"]))
        res = run_episode(env, policy, ev, time_limit[ev.scenario], limits, director, vid)
        results.append(res)
        progress(f"{ev.scenario} seed {ev.seed}: {'PASS' if res['success'] else 'FAIL'} "
                 f"({res['sim_time_s']} s sim, {res['wall_s']} s wall, {res['kpi']['collisions']} collisions, "
                 f"{res['kpi']['parts_damaged']} damaged, {len(res['kpi']['jams'])} jams, "
                 f"guard held {res['guard']['held']})")
        _write(out, args, policy, limits, results, products_of)
    _write(out, args, policy, limits, results, products_of)
    env.close()
    return 0


def _write(out: Path, args, policy, limits, results, products_of) -> None:
    report = {"run": args.run, "policy": args.policy, "director": args.director, "time_limit_cap_s": args.time_limit,
              "gates": "sim/assembly_lab/INTERFACES.md Precision gates", "limits": limits,
              "summary": aggregate(results, products_of), "episodes": results}
    (out / "results.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    sys.exit(main())
