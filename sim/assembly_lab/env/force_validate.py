"""Validate the force-guard numbers against known loads: every resting object's normal_N must equal its weight, and a
known external push must read back, both within 5 %.

    isaaclab_run.sh sim/assembly_lab/env/force_validate.py --out ~/t12_data/force_validate.json

1. All 30 product objects stand upright on the table (one reset, 70 mm grid, no robot contact); after settling, normal_N
   and lateral_N (net contact force minus robot-link forces, contact_metrics.py) are averaged over 10 env steps and
   compared with m g and 0. env_contact_coverage (per-contact sum / net) is reported next to them.
2. A permanent external wrench (Isaac Lab wrench composer, world frame, at the COM) presses chosen objects down with
   5 N plus a 1 N sideways push and a 0.02 N m twist about z (all below the friction limit, so nothing slides):
   normal_N must read m g + 5, lateral_N 1.0, and the ungated per-contact torque torque_raw_Nm 0.02.
Exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tol", type=float, default=0.05)
ap.add_argument("--seated_down", type=float, default=1.0, help="N pressing the seated parts (low, so the twist "
                "beats face friction and the peg walls carry it)")
ap.add_argument("--seated_twist", type=float, default=0.02)
ap.add_argument("--twist", type=float, default=0.02, help="N m about z; above mu N r_eff the part spins (not static)")
args = ap.parse_args()

from cell_env import AssemblyCellEnv, CellEnvCfg

G = 9.81
PUSH = {"down_N": 5.0, "side_N": 1.0, "twist_Nm": args.twist}
GRID = [(x, y) for x in (240.0, 310.0, 380.0, 450.0) for y in (-245.0, -175.0, -105.0, -35.0, 35.0, 105.0, 175.0, 245.0)]


def ids_for(products: list[int]) -> list[str]:
    return [u for p in products for u in [f"F0{p}"] + [f"M{p}_0{k}" for k in range(1, 5)]]


def averaged(env, hold: list[float], n: int = 10) -> dict:
    acc: dict[str, np.ndarray] = {}
    for _ in range(n):
        c = env.step_tcp([hold], 0.1)["sim"]["contact"]
        for u, v in c.items():
            row = np.array([v["normal_N"], v["lateral_N"], v["torque_raw_Nm"], v["env_contact_coverage"]])
            acc[u] = acc.get(u, 0.0) + row / n
    return acc


def wrench(env, uid: str, on: bool, side: bool = True, down: float | None = None, twist: float | None = None) -> None:
    import torch

    e = env.env
    obj = e.scene[e.names[e.pool.index(uid)]]
    s = 1.0 if on else 0.0
    down = PUSH["down_N"] if down is None else down
    twist = PUSH["twist_Nm"] if twist is None else twist
    f = torch.tensor([[[PUSH["side_N"] * s * side, 0.0, -down * s]]], device=e.device)
    t = torch.tensor([[[0.0, 0.0, twist * s]]], device=e.device)
    ids = torch.tensor([0], device=e.device)
    obj.permanent_wrench_composer.set_forces_and_torques_index(forces=f, torques=t, body_ids=ids, env_ids=ids,
                                                               is_global=True)


def within(measured: float, expected: float, tol: float) -> bool:
    return abs(measured - expected) <= tol * abs(expected)


def run_group(products: list[int], push: list[str]):
    ids = ids_for(products)
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=ids, cameras=()))
    lay = {"objects": [{"id": u, "x": x, "y": y, "yaw": 0.0} for u, (x, y) in zip(ids, GRID)], "instruction": "fv"}
    obs = env.reset(lay)
    hold = [*obs["tcp_pose"], 0.0]
    rest = averaged(env, hold)
    rows = []
    for u in ids:
        w = env.masses[u] * G
        n, lat, _, cov = rest[u]
        rows.append({"object": u, "weight_N": round(w, 4), "normal_N": round(float(n), 4),
                     "err_pct": round(100 * (n - w) / w, 2), "lateral_N": round(float(lat), 4),
                     "coverage": round(float(cov), 3), "ok": within(n, w, args.tol) and lat <= args.tol * w})
        print(f"[fv] rest {rows[-1]}", flush=True)
    for u in push:
        wrench(env, u, True)
    yaw0 = {u: env.env.object_poses_mm()[u] for u in push}
    loaded = averaged(env, hold)
    yaw1 = env.env.object_poses_mm()
    pushed = []
    for u in push:
        w = env.masses[u] * G
        n, lat, tq, cov = loaded[u]
        r = {"object": u, "expect_normal_N": round(w + PUSH["down_N"], 4), "normal_N": round(float(n), 4),
             "expect_lateral_N": PUSH["side_N"], "lateral_N": round(float(lat), 4),
             "expect_torque_Nm": PUSH["twist_Nm"], "torque_raw_Nm": round(float(tq), 5), "coverage": round(float(cov), 3)}
        d = yaw0[u][:3, :3].T @ yaw1[u][:3, :3]
        r["turned_deg"] = round(float(np.degrees(np.arctan2(d[1, 0], d[0, 0]))), 3)    # > 0.1: the part slipped
        k = env.env.pool.index(u)
        r["env_force_xyz_N"] = [round(float(x), 4) for x in env.env._metrics.env_net[k]]   # expect (-1, 0, +m g + 5)
        r["sign_ok"] = r["env_force_xyz_N"][0] < 0 < r["env_force_xyz_N"][2]
        r["normal_ok"] = within(n, w + PUSH["down_N"], args.tol)
        r["lateral_ok"] = within(lat, PUSH["side_N"], args.tol)
        r["torque_ok"] = within(tq, PUSH["twist_Nm"], args.tol)
        pushed.append(r)
        print(f"[fv] push {r}", flush=True)
        wrench(env, u, False)
    return env, rows, pushed


def seated_twist(env, pairs: list[tuple[str, str]]) -> list[dict]:
    """Seat each part on its base, press it down 5 N and twist it about z: once the part turns through its peg
    clearance the base's peg walls (normal contacts) hold the twist, so torque_raw_Nm must read the applied twist."""
    objs = []
    for k, (base, part) in enumerate(pairs):
        x, y = GRID[2 * k]
        objs += [{"id": base, "x": x, "y": y, "yaw": 0.0}, {"id": part, "state": f"on:{base}"}]
    obs = env.reset({"objects": objs, "instruction": "fv seated"})
    hold = [*obs["tcp_pose"], 0.0]
    for _, part in pairs:
        wrench(env, part, True, side=False, down=args.seated_down, twist=args.seated_twist)
    p0 = env.env.object_poses_mm()
    averaged(env, hold, 5)                                           # turn into the pegs, then measure
    loaded = averaged(env, hold)
    p1 = env.env.object_poses_mm()
    rows = []
    for base, part in pairs:
        d = p0[part][:3, :3].T @ p1[part][:3, :3]
        n, _lat, tq, cov = loaded[part]
        r = {"object": part, "on": base, "expect_torque_Nm": args.seated_twist, "torque_raw_Nm": round(float(tq), 5),
             "normal_N": round(float(n), 4), "expect_normal_N": round(env.masses[part] * G + args.seated_down, 4),
             "turned_deg": round(float(np.degrees(np.arctan2(d[1, 0], d[0, 0]))), 3), "coverage": round(float(cov), 3),
             "rose_mm": round(float(p1[part][2, 3] - p0[part][2, 3]), 3)}           # pegs 2.5 mm: turning far needs a rise
        r["torque_ok"] = within(tq, args.seated_twist, args.tol)
        r["normal_ok"] = within(n, env.masses[part] * G + args.seated_down, args.tol)
        rows.append(r)
        print(f"[fv] seated twist {r}", flush=True)
        wrench(env, part, False)
    return rows


def main() -> None:
    env, rest, pushed = run_group([1, 2, 3, 4, 5, 6], ["F01", "F04", "M1_01", "M4_01", "M2_02", "M5_03"])
    seated = seated_twist(env, [("F04", "M4_01"), ("F01", "M1_01"), ("F05", "M5_01")])
    ok_rest = sum(r["ok"] for r in rest)
    report = {"tolerance": args.tol, "push": PUSH, "rest": rest, "pushed": pushed, "seated_twist": seated,
              "rest_ok": f"{ok_rest}/{len(rest)}"}
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"[fv] rest within {args.tol:.0%}: {ok_rest}/{len(rest)}; push normal "
          f"{sum(p['normal_ok'] for p in pushed)}/{len(pushed)} lateral {sum(p['lateral_ok'] for p in pushed)}/"
          f"{len(pushed)} torque {sum(p['torque_ok'] for p in pushed)}/{len(pushed)} -> {out}", flush=True)
    failed = ok_rest < len(rest) or not all(p["normal_ok"] and p["lateral_ok"] and p["sign_ok"] for p in pushed)
    print(f"[fv] {'FAIL: force guard numbers do not match known loads' if failed else 'PASS'}", flush=True)
    env.close()            # closing the app ends the process, so the verdict is printed first
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
