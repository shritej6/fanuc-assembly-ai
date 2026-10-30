"""Score FoundationPose estimates where the gripper acts: at the grasp TCP and on the part's up axis (T34).

ADD-S says how far the whole mesh is off; a grasp cares about two things only:
- TCP error: distance between where the estimate and the truth put the commanded grasp TCP
  (`cad/grasp_frames.json` `grips[].tcp_local_mm`, installed-local mm, mapped to the STL frame with
  `inv(T_local_from_stl)`). A part that is symmetric under the swap of the two wrist turns may be estimated turned
  by 180 deg; the TCP of turn k then lands on the truth's other turn, which grips the part just as well, so each
  turn is scored against the nearer truth TCP and the worse turn is kept.
- tilt error: angle between the estimated and true installed-local +Z (the part's up axis). Yaw about +Z is not in
  it, so symmetry about the vertical axis cannot inflate it.

Gate (T34 brief): TCP error <= 1.8 mm and tilt error <= 3 deg. Parts without grips (bases) are not scored.
Poses are 4x4 object-to-camera transforms in metres with the STL as the object frame, as FoundationPose returns.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TCP_GATE_MM = 1.8
TILT_GATE_DEG = 3.0


def load_grasp_points(path: Path) -> dict[str, dict]:
    """Per part: TCP points (STL frame, metres, one row per wrist turn) and the up axis in the STL frame."""
    objects = json.loads(path.read_text())["objects"]
    out = {}
    for part, entry in objects.items():
        if not entry["grips"]:
            continue
        stl_from_local = np.linalg.inv(np.array(entry["T_local_from_stl"], dtype=np.float64))
        tcps = [p for grip in entry["grips"] for p in grip["tcp_local_mm"].values()]
        local = np.hstack([np.array(tcps, dtype=np.float64), np.ones((len(tcps), 1))])
        out[part] = {"tcp_stl_m": (local @ stl_from_local.T)[:, :3] / 1000.0,
                     "up_stl": stl_from_local[:3, :3] @ np.array([0.0, 0.0, 1.0])}
    return out


def grasp_errors(estimate: np.ndarray, truth: np.ndarray, points: dict) -> dict:
    """TCP error (mm, worst turn against the nearer truth TCP) and tilt error (deg) of one estimate."""
    tcp = points["tcp_stl_m"]
    est = tcp @ estimate[:3, :3].T + estimate[:3, 3]
    gt = tcp @ truth[:3, :3].T + truth[:3, 3]
    nearest = np.linalg.norm(est[:, None, :] - gt[None, :, :], axis=2).min(axis=1)
    up_est, up_gt = estimate[:3, :3] @ points["up_stl"], truth[:3, :3] @ points["up_stl"]
    cos = float(np.clip(up_est @ up_gt / (np.linalg.norm(up_est) * np.linalg.norm(up_gt)), -1.0, 1.0))
    return {"tcp_mm": float(nearest.max() * 1000.0), "tilt_deg": float(np.degrees(np.arccos(cos)))}


def upright_hypotheses(rot_grid: np.ndarray, up_stl: np.ndarray, up_cam: np.ndarray, max_deg: float) -> np.ndarray:
    """Mask of start rotations (object-in-camera) whose installed up axis is within max_deg of the table's up.

    A part the detector calls standing (not `__fallen`) stands as installed, so a start pose that turns it upside
    down can only converge to a mirror-like wrong answer; near-symmetric parts did exactly that with all 252 starts.
    """
    ups = rot_grid[:, :3, :3] @ up_stl
    return ups @ up_cam >= np.cos(np.radians(max_deg))


def within_gate(tcp_mm: float, tilt_deg: float) -> bool:
    return tcp_mm <= TCP_GATE_MM and tilt_deg <= TILT_GATE_DEG


def spread(estimates: list[np.ndarray], points: dict) -> dict:
    """Repeatability of repeated estimates of one object: largest TCP and tilt distance from their mean pose.

    The reference is the first estimate's frame averaged over the TCP points, so no truth is needed.
    """
    tcp = points["tcp_stl_m"]
    placed = np.stack([tcp @ e[:3, :3].T + e[:3, 3] for e in estimates])           # runs x turns x 3
    centre = placed.mean(axis=0)
    ups = np.stack([e[:3, :3] @ points["up_stl"] for e in estimates])
    mean_up = ups.mean(axis=0) / np.linalg.norm(ups.mean(axis=0))
    tilt = np.degrees(np.arccos(np.clip(ups @ mean_up, -1.0, 1.0)))
    return {"tcp_spread_mm": float(np.linalg.norm(placed - centre, axis=2).max() * 1000.0),
            "tilt_spread_deg": float(tilt.max())}


def _stats(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    return {"median": float(np.median(arr)), "p95": float(np.percentile(arr, 95)), "max": float(arr.max())}


def part_report(rows: list[dict], points: dict) -> dict:
    """Accuracy over first runs and repeatability over all runs of each object, for one part."""
    first = [r for r in rows if r["run"] == 0]
    scored = [grasp_errors(np.array(r["estimate"]), np.array(r["truth"]), points) for r in first]
    report = {"objects": len(first), "adds_mm": _stats([r["adds_mm"] for r in first]),
              "trans_mm": _stats([r["trans_mm"] for r in first]), "rot_deg": _stats([r["rot_deg"] for r in first]),
              "tcp_mm": _stats([s["tcp_mm"] for s in scored]), "tilt_deg": _stats([s["tilt_deg"] for s in scored]),
              "share_within_gate": float(np.mean([within_gate(s["tcp_mm"], s["tilt_deg"]) for s in scored]))}
    groups: dict[tuple, list[np.ndarray]] = {}
    for r in rows:
        groups.setdefault((r["scene"], r["uid"]), []).append(np.array(r["estimate"]))
    spreads = [spread(g, points) for g in groups.values() if len(g) > 1]
    if spreads:
        report["repeat_runs"] = max(len(g) for g in groups.values())
        report["tcp_spread_mm"] = _stats([s["tcp_spread_mm"] for s in spreads])
        report["tilt_spread_deg"] = _stats([s["tilt_spread_deg"] for s in spreads])
    return report


def report(rows: list[dict], grasp_points: dict) -> dict:
    """Per part report; a part passes when every first-run estimate is inside the gate."""
    out = {}
    for part in sorted({r["object_id"] for r in rows if r["object_id"] in grasp_points}):
        entry = part_report([r for r in rows if r["object_id"] == part], grasp_points[part])
        entry["passes_gate"] = entry["share_within_gate"] == 1.0
        out[part] = entry
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=Path, nargs="+", required=True, help="rows.jsonl from validate_foundationpose")
    ap.add_argument("--grasp-frames", type=Path, required=True, help="cad/grasp_frames.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rest", default="installed", help="score only this rest state (fallen parts are not grasped)")
    args = ap.parse_args()
    rows = [json.loads(line) for path in args.rows for line in path.read_text().splitlines() if line]
    rows = [r for r in rows if r["rest"] == args.rest]
    result = {"tcp_gate_mm": TCP_GATE_MM, "tilt_gate_deg": TILT_GATE_DEG, "rest": args.rest,
              "parts": report(rows, load_grasp_points(args.grasp_frames))}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
