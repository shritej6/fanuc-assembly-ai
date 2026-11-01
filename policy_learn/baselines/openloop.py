"""Open-loop check of ANY policy server (ACT, SmolVLA or GR00T) on held-out episodes, through the evaluator's path.

    python -m policy_learn.baselines.openloop --policy groot://127.0.0.1:5561 --dataset ~/baseline_data/val \
        [--episodes 10] [--out report.json]

Every 16th frame of each held-out episode is sent exactly as the sim evaluator sends it: `GrootPolicy(spec)`
(`policy_learn/deploy/groot_policy.py`) -> agent 5's GR00T client -> the server, and the chunk comes back through the
real-arm guard. The guarded rows are compared with the recorded `act_tcp` targets of the same frames:
- xyz error (mm, per row, mean over rows the recording has),
- rotation error (deg, angle of R_pred^T R_rec),
- gripper agreement (fraction of rows with the same open / closed state),
- guard holds (a held query is counted, not scored: its row is the current pose) and clamps.
Because GR00T's own server answers the same protocol, the same command gives GR00T's number on the same frames.
This is an open-loop (teacher-forced) check, not a task success rate; the sim evaluator measures that.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from policy_learn.deploy.cell_groot_policy import GrootPolicy
from policy_learn.groot.prepare_dataset import CAMERAS
from policy_learn.groot.tcp_convert import eef9d_to_tcp, rot6d_to_matrix, wpr_to_matrix

FINGERS_DOWN_WPR = np.array([-180.0, 0.0, 0.0])


def cell_obs(item: dict) -> dict:
    """A LeRobot view frame -> the observation dict the evaluator gives GrootPolicy.act."""
    state = item["observation.state"].numpy().astype(np.float64)   # [eef_9d, gripper, J1..J6]
    images = {cam: (item[f"observation.images.{cam}"].numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
              for cam in CAMERAS}
    tcp = eef9d_to_tcp(state[None, :9], FINGERS_DOWN_WPR)[0]
    return {"images": images, "joints": np.concatenate([state[10:16], state[9:10]]), "tcp_pose": tcp,
            "instruction": item["task"]}


def compare(rows: np.ndarray, recorded: np.ndarray) -> dict:
    """rows (N, 7) guarded act_tcp; recorded (M, 10) [eef_9d, gripper] targets of the same frames."""
    n = min(len(rows), len(recorded))
    rows, recorded = rows[:n], recorded[:n]
    xyz = np.linalg.norm(rows[:, :3] - recorded[:, :3], axis=1)
    rel = np.einsum("nji,njk->nik", wpr_to_matrix(rows[:, 3:6]), rot6d_to_matrix(recorded[:, 3:9]))
    angle = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    grip = (rows[:, 6] >= 0.5) == (recorded[:, 9] >= 0.5)
    return {"rows": n, "xyz_mm": xyz.tolist(), "rot_deg": angle.tolist(), "grip_ok": grip.tolist()}


def run(policy_spec: str, dataset: Path, episodes: int, stride: int) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset("local/fanuc_val", root=dataset, video_backend="pyav",
                        delta_timestamps={"action": [i / 5 for i in range(stride)]})
    policy = GrootPolicy(policy_spec)
    xyz, rot, grip, queries, held_queries = [], [], [], 0, 0
    bounds = ds.episode_data_index
    for ep in range(min(episodes, ds.num_episodes)):
        policy.reset()
        for idx in range(int(bounds["from"][ep]), int(bounds["to"][ep]), stride):
            item = ds[idx]
            n_log = len(policy.guard_log)
            rows = np.asarray(policy.act(cell_obs(item)), float)
            queries += 1
            if any(e["held"] for e in policy.guard_log[n_log:]):
                held_queries += 1   # a HOLD row is the current pose, not the policy's answer: never scored
                continue
            recorded = item["action"].numpy()[~item["action_is_pad"].numpy()]
            res = compare(rows, recorded)
            xyz += res["xyz_mm"]
            rot += res["rot_deg"]
            grip += res["grip_ok"]
    scored = bool(xyz)
    return {"policy": policy_spec, "dataset": str(dataset), "episodes": min(episodes, ds.num_episodes),
            "queries": queries, "queries_held": held_queries, "rows_compared": len(xyz),
            "xyz_err_mm": {"mean": float(np.mean(xyz)), "median": float(np.median(xyz)),
                           "p90": float(np.percentile(xyz, 90))} if scored else None,
            "rot_err_deg": {"mean": float(np.mean(rot)), "median": float(np.median(rot))} if scored else None,
            "gripper_agreement": float(np.mean(grip)) if scored else None,
            "guard_clamped": sum(not e["held"] for e in policy.guard_log),
            "note": "open-loop on held-out recorded frames; errors over the queries the guard passed, held ones "
                    "counted in queries_held"}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="groot://host:port (ACT, SmolVLA or GR00T server)")
    ap.add_argument("--dataset", type=Path, required=True, help="lerobot_view.py output split, e.g. <out>/val")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stride", type=int, default=16, help="frames between queries (= rows compared per query)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    report = run(args.policy, args.dataset.expanduser(), args.episodes, args.stride)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
