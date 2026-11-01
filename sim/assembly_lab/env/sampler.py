"""Default episode sampler for Mimic generation: a random layout of one product and its TCP frames.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import math

import layout as L
import numpy as np

GRASPS = json.loads((L.REPO / "cad/fingers/product_grasp_frames.json").read_text())
TIP_Z_MM = 86.1
WORK_X, WORK_Y = (220.0, 470.0), (-270.0, 270.0)
GAP = {"F": 40.0, "M": 26.0}      # footprint radius used for spacing, mm


def cad_grasp(solver: L.LayoutSolver, uid: str, part_tf: np.ndarray) -> np.ndarray:
    tid = L.type_id(uid)
    g = GRASPS[next(k for k in GRASPS if k.startswith(tid + "_"))]
    rot = np.array(g["rotation_gripper_to_world"], float)
    tcp_asm = np.eye(4)
    tcp_asm[:3, :3] = rot
    tcp_asm[:3, 3] = rot @ (0.0, 0.0, TIP_Z_MM) + np.array(g["translation_mm"])
    return part_tf @ np.linalg.inv(np.array(solver.geometry[tid]["installed_in_base_frame"])) @ tcp_asm


def random_layout(rng: np.random.Generator, ids: list[str], installed: dict[str, str]) -> dict:
    placed, objects = [], []
    for oid in ids:
        if oid in installed:
            objects.append({"id": oid, "state": f"on:{installed[oid]}"})
            continue
        rad = GAP[oid[0]]
        for _ in range(4000):
            x, y = rng.uniform(*WORK_X), rng.uniform(*WORK_Y)
            if 180.0 <= math.hypot(x, y) <= 490.0 and all(math.hypot(x - a, y - b) >= rad + rb for a, b, rb in placed):
                break
        else:
            raise RuntimeError(f"no free spot for {oid}")
        placed.append((x, y, rad))
        objects.append({"id": oid, "x": float(x), "y": float(y), "yaw": float(rng.uniform(-180, 180))})
    return {"objects": objects, "free": placed}


def free_spot(rng: np.random.Generator, placed: list) -> tuple[float, float]:
    for _ in range(4000):
        x, y = rng.uniform(*WORK_X), rng.uniform(*WORK_Y)
        if 180.0 <= math.hypot(x, y) <= 490.0 and all(math.hypot(x - a, y - b) >= 60.0 for a, b, _ in placed):
            return float(x), float(y)
    raise RuntimeError("no free test-bed spot")


def sample(api, rng: np.random.Generator) -> dict[str, np.ndarray]:
    solver, skill = api.solver, api.skill
    pool = set(api.pool)
    choices = []                                                 # (product, k): base and parts 1..k all spawned
    for p in sorted({int(u[1:]) for u in pool if u.startswith("F")}):
        for k in range(1, 5):
            if all(f"M{p}_0{j}" in pool for j in range(1, k + 1)):
                choices.append((p, k))
    if not choices:
        raise RuntimeError(f"pool {sorted(pool)} has no base with its first part")
    for _ in range(50):
        p, k = choices[int(rng.integers(len(choices)))]          # target = sheet part k
        ids = [u for u in [f"F0{p}"] + [f"M{p}_0{j}" for j in range(1, 5)] if u in pool]
        below = {f"M{p}_0{j}": f"F0{p}" for j in range(1, k)}
        if skill == "pull_out":
            below[f"M{p}_0{k}"] = f"F0{p}"
        lay = random_layout(rng, ids, below)
        poses = api.apply_layout({"objects": lay["objects"]})
        target = f"M{p}_0{k}"
        grasp = cad_grasp(solver, target, poses[target])
        rel = np.linalg.inv(poses[target]) @ grasp
        state = solver.seat_tf_for(target, f"F0{p}", poses)
        frames = {"grasp_frame": grasp, "seat_frame": state @ rel}
        if skill == "defect_drop":
            x, y = solver.defect_xy(int(rng.integers(0, 6)))
        else:
            x, y = free_spot(rng, lay["free"])
        rest = solver.table_tf(L.type_id(target), x, y, float(rng.uniform(-180, 180)), "installed")
        frames["spot_frame"] = rest @ rel
        needed = [frames["grasp_frame"]] + ([frames["seat_frame"]] if skill == "pick_insert" else [frames["spot_frame"]])
        if all(api.reachable(f) for f in needed):
            return frames
    raise RuntimeError(f"no reachable {skill} episode in 50 layouts")
