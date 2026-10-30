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
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--tol", type=float, default=0.05)
ap.add_argument("--seated_down", type=float, default=1.0, help="N pressing the seated parts (low, so the twist "
                "beats face friction and the peg walls carry it)")
ap.add_argument("--seated_twists", type=lambda s: [float(x) for x in s.split(",")], default=[0.02, 0.05, 0.08])
ap.add_argument("--drop_mm", type=float, default=4.0, help="offset-insert start height above the seat")
ap.add_argument("--twist", type=float, default=0.02, help="N m about z; above mu N r_eff the part spins (not static)")
args = ap.parse_args()

import layout as L
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


BASE_HOLD_N = 30.0      # presses each base onto the table so the base itself cannot turn (table friction ~0.2 N m)


def _yaw_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))


def _seat(env, pairs: list[tuple[str, str]], label: str) -> list[float]:
    objs = []
    for k, (base, part) in enumerate(pairs):
        x, y = GRID[2 * k]
        objs += [{"id": base, "x": x, "y": y, "yaw": 0.0}, {"id": part, "state": f"on:{base}"}]
    obs = env.reset({"objects": objs, "instruction": label})
    for base, _ in pairs:
        wrench(env, base, True, side=False, down=BASE_HOLD_N, twist=0.0)
    return [*obs["tcp_pose"], 0.0]


def peg_clearance_deg(base: str, part: str) -> float:
    """Yaw play of `part` on `base` from CAD (objects.yaml): per-side clearance over half the distance between the
    base's two round locating pegs (the part turns about their midpoint)."""
    objs = yaml.safe_load(L.OBJECTS_YAML.read_text())["objects"]
    pegs = [np.array(p["xy_mm"], float) for p in objs[L.type_id(base)]["pegs"] if p.get("shape") == "round"]
    clear = next(m["clearance_per_side_mm"] for m in objs[L.type_id(part)]["mates_with"])
    return float(np.degrees(np.arctan(clear / (0.5 * np.linalg.norm(pegs[0] - pegs[1])))))


def seated_twist(env, pairs: list[tuple[str, str]], twist: float) -> list[dict]:
    """Seat each part on its (held) base, press it down and twist it about z. The pegs must stop it within the CAD
    clearance (0.5 mm per side over ~15.6 mm: <= 1.8 deg relative to the base) or it must lift >= 2 mm to ride over
    them; once stopped, the base's peg walls carry the twist, so torque_raw_Nm should read it (per-contact check)."""
    hold = _seat(env, pairs, "fv seated twist")
    for _, part in pairs:
        wrench(env, part, True, side=False, down=args.seated_down, twist=twist)
    p0 = env.env.object_poses_mm()
    averaged(env, hold, 5)
    loaded = averaged(env, hold)
    p1 = env.env.object_poses_mm()
    rows = []
    for base, part in pairs:
        rel0, rel1 = np.linalg.inv(p0[base]) @ p0[part], np.linalg.inv(p1[base]) @ p1[part]
        _n, _lat, tq, cov = loaded[part]
        r = {"object": part, "on": base, "twist_Nm": twist, "down_N": args.seated_down,
             "turn_vs_base_deg": round(_yaw_deg(rel0[:3, :3].T @ rel1[:3, :3]), 3),
             "rise_vs_base_mm": round(float(rel1[2, 3] - rel0[2, 3]), 3),
             "base_turn_deg": round(_yaw_deg(p0[base][:3, :3].T @ p1[base][:3, :3]), 3),
             "torque_raw_Nm": round(float(tq), 5), "coverage": round(float(cov), 3)}
        r["cad_clearance_deg"] = round(peg_clearance_deg(base, part), 3)
        r["held"] = abs(r["turn_vs_base_deg"]) <= r["cad_clearance_deg"] + 0.1 or r["rise_vs_base_mm"] >= 2.0
        r["torque_ok"] = within(tq, twist, args.tol)
        rows.append(r)
        print(f"[fv] seated twist {r}", flush=True)
    for base, part in pairs:
        wrench(env, part, False)
        wrench(env, base, False)
    return rows


def offset_insert(env, pairs: list[tuple[str, str]], offset_mm: float) -> list[dict]:
    """Drop each FREE part from 4 mm above its seat, offset `offset_mm` sideways (+x), and press it down 3 N for 2 s.
    CAD capture of a free part: the 0.5 mm-chamfered peg tip (radius 1.5 mm) must land inside the 0.5 mm-chamfered
    socket mouth (radius 3.0 mm), i.e. offsets up to 1.5 mm slide in and must seat (< 0.3 mm high); 2 mm must jam
    (>= 2 mm high, on the peg tops). A part held stiffly by the gripper captures less (agent 2: 0.5 mm; demo_check
    --offset_mm tests that)."""
    hold = _seat(env, pairs, "fv offset insert")
    e = env.env
    seats = {part: e.solver.seat_tf_for(part, base, e.object_poses_mm()) for base, part in pairs}
    names = [e.names[e.pool.index(part)] for _, part in pairs]
    e._write_poses(names, [L.translation(offset_mm, 0.0, args.drop_mm) @ seats[part] for _, part in pairs])
    e.scene.write_data_to_sim()
    for _, part in pairs:
        wrench(env, part, True, side=False, down=3.0, twist=0.0)
    for _ in range(20):
        env.step_tcp([hold], 0.1)
    poses = e.object_poses_mm()
    rows = []
    for base, part in pairs:
        seat = e.solver.seat_tf_for(part, base, poses)
        d = np.linalg.inv(seat) @ poses[part]
        r = {"object": part, "on": base, "offset_mm": offset_mm, "drop_mm": args.drop_mm,
             "height_above_seat_mm": round(float(d[2, 3]), 3),
             "tilt_deg": round(float(np.degrees(np.arccos(np.clip(d[2, 2], -1.0, 1.0)))), 3),
             "xy_from_seat_mm": [round(float(d[0, 3]), 3), round(float(d[1, 3]), 3)]}
        r["ok"] = d[2, 3] < 0.3 if offset_mm <= 1.5 else d[2, 3] >= 2.0
        rows.append(r)
        print(f"[fv] offset insert {r}", flush=True)
    for base, part in pairs:
        wrench(env, part, False)
        wrench(env, base, False)
    return rows


def main() -> None:
    env, rest, pushed = run_group([1, 2, 3, 4, 5, 6], ["F01", "F04", "M1_01", "M4_01", "M2_02", "M5_03"])
    pairs = [("F04", "M4_01"), ("F01", "M1_01"), ("F05", "M5_01")]
    seated = [r for t in args.seated_twists for r in seated_twist(env, pairs, t)]
    inserts = [r for off in (0.0, 1.0, 2.0) for r in offset_insert(env, pairs, off)]
    ok_rest = sum(r["ok"] for r in rest)
    report = {"tolerance": args.tol, "push": PUSH, "rest": rest, "pushed": pushed, "seated_twist": seated,
              "offset_insert": inserts,
              "rest_ok": f"{ok_rest}/{len(rest)}"}
    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    print(f"[fv] rest within {args.tol:.0%}: {ok_rest}/{len(rest)}; push normal "
          f"{sum(p['normal_ok'] for p in pushed)}/{len(pushed)} lateral {sum(p['lateral_ok'] for p in pushed)}/"
          f"{len(pushed)} torque {sum(p['torque_ok'] for p in pushed)}/{len(pushed)} -> {out}", flush=True)
    failed = ok_rest < len(rest) or not all(p["normal_ok"] and p["lateral_ok"] and p["sign_ok"] for p in pushed) \
        or not all(r["held"] for r in seated) or not all(r["ok"] for r in inserts)
    print(f"[fv] {'FAIL: force guard numbers do not match known loads' if failed else 'PASS'}", flush=True)
    env.close()            # closing the app ends the process, so the verdict is printed first
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
