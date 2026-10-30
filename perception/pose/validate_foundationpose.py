"""Measure pretrained FoundationPose (STL + RGB + depth) against the exact poses of the test-bed renders.

Table camera only: its pose is fixed (v0.5 calibration in sim/starter/scene_config.py) plus the scene's jitter, which
is re-applied with pxr.Gf exactly as sdg/isaac/generate.py does. Depth is stored at 640x400 (uint16 mm, the real
cell's size), so RGB and the instance mask are resized to it and the intrinsics are scaled per axis. Only loose,
defect-free parts are scored, standing (installed rest) and fallen ones reported apart. Writes per-object rows + a summary (ADD-S mm, translation mm, rotation deg).

Run inside the fa-foundationpose container with the FoundationPose checkout on PYTHONPATH and usd-core installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

DEPTH_SIZE = (640, 400)
TABLE_TOP_Z = 0.90
UF0_ABOVE_BASE_M = 0.330
USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])   # USD camera looks down -Z with +Y up; OpenCV looks down +Z, +Y down


def load_camera_config(repo: Path, name: str = "TABLE_CAM") -> dict:
    spec = importlib.util.spec_from_file_location("scene_config", repo / "sim/starter/scene_config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(getattr(module, name))


def intrinsics(cfg: dict, size: tuple[int, int]) -> np.ndarray:
    """Pinhole K at `size`: USD square pixels at the render resolution, scaled per axis to the stored size."""
    width, height = cfg["resolution"]
    focal_px = width * cfg["focal_length"] / cfg["horiz_aperture"]
    sx, sy = size[0] / width, size[1] / height
    return np.array([[focal_px * sx, 0.0, width / 2 * sx], [0.0, focal_px * sy, height / 2 * sy], [0.0, 0.0, 1.0]])


def camera_to_world_cv(cfg: dict, jitter: dict) -> np.ndarray:
    """Camera-to-world (metres, OpenCV axes) exactly as the renderer posed the camera prim."""
    from pxr import Gf

    view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*cfg["eye"]), Gf.Vec3d(*cfg["look_at"]), Gf.Vec3d(*cfg["up"]))
    rx, ry, rz = jitter["rotate_deg"]
    delta = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
                                    * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
    delta.SetTranslateOnly(Gf.Vec3d(*(np.array(jitter["translate_mm"]) / 1000.0)))
    row_major = np.array(delta * view.GetInverse())          # USD row-vector convention
    return row_major.T @ USD_TO_CV


def object_to_world(transform_uf0_mm: list[list[float]]) -> np.ndarray:
    m = np.array(transform_uf0_mm, dtype=np.float64)
    m[:3, 3] = m[:3, 3] / 1000.0 + (0.0, 0.0, TABLE_TOP_Z + UF0_ABOVE_BASE_M)
    return m


def project(K: np.ndarray, object_in_camera: np.ndarray) -> tuple[float, float]:
    x, y, z = object_in_camera[:3, 3]
    return K[0, 0] * x / z + K[0, 2], K[1, 1] * y / z + K[1, 2]


def errors(estimate: np.ndarray, truth: np.ndarray, points_m: np.ndarray) -> dict:
    """ADD-S (symmetry-safe mean closest-point distance), translation and rotation error."""
    from scipy.spatial import cKDTree

    est = points_m @ estimate[:3, :3].T + estimate[:3, 3]
    gt = points_m @ truth[:3, :3].T + truth[:3, 3]
    adds = float(cKDTree(est).query(gt)[0].mean())
    cos = (np.trace(estimate[:3, :3].T @ truth[:3, :3]) - 1) / 2
    return {"adds_mm": adds * 1000, "trans_mm": float(np.linalg.norm(estimate[:3, 3] - truth[:3, 3]) * 1000),
            "rot_deg": float(np.degrees(np.arccos(np.clip(cos, -1, 1))))}


def scene_items(chunk: Path, stem: str, cfg: dict) -> list[dict]:
    label = json.loads((chunk / "labels" / f"{stem}.json").read_text())
    spec = label["spec"]
    if "table" not in spec["cameras"]:
        return []
    cam_to_world = camera_to_world_cv(cfg, spec["camera_jitter"]["table"])
    world_to_cam = np.linalg.inv(cam_to_world)
    boxes = {b["uid"]: b for b in label["cameras"]["table"] if b.get("visible_px", 0) > 400}
    items = []
    for o in spec["objects"]:
        loose = o["kind"] == "part" and o["state"] == "on_table" and o.get("variant") == "ok"
        if loose and o["uid"] in boxes:
            items.append({"uid": o["uid"], "object_id": o["object_id"], "stl": o["stl"], "rest": o["rest"],
                          "bbox_xyxy": boxes[o["uid"]]["bbox_xyxy"], "seg_value": int(o["uid"][1:]) + 1,
                          "truth": world_to_cam @ object_to_world(o["transform_uf0"])})
    return items


def check_geometry(args: argparse.Namespace) -> None:
    """No FoundationPose: every ground-truth origin must project inside its labelled box (render resolution)."""
    cfg = load_camera_config(args.repo)
    K = intrinsics(cfg, tuple(cfg["resolution"]))
    inside = total = 0
    for chunk in sorted(c for data in args.data for c in data.glob("chunk_*")):
        for path in sorted((chunk / "labels").glob("*.json"))[: args.scenes]:
            for item in scene_items(chunk, path.stem, cfg):
                u, v = project(K, item["truth"])
                x0, y0, x1, y1 = item["bbox_xyxy"]
                total += 1
                inside += int(x0 - 5 <= u <= x1 + 5 and y0 - 5 <= v <= y1 + 5)
    print(json.dumps({"projected_inside_box": inside, "checked": total}))


def load_views(chunk: Path, stem: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = cv2.cvtColor(cv2.imread(str(chunk / "rgb" / "table" / f"{stem}.jpg")), cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, DEPTH_SIZE, interpolation=cv2.INTER_AREA)
    depth = cv2.imread(str(chunk / "depth" / "table" / f"{stem}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    seg = cv2.imread(str(chunk / "seg" / "table" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    seg = cv2.resize(seg, DEPTH_SIZE, interpolation=cv2.INTER_NEAREST)
    return rgb, depth, seg


def make_estimator(args: argparse.Namespace, item: dict, cache: dict, models: tuple) -> tuple:
    if item["object_id"] not in cache:
        from estimater import FoundationPose

        scorer, refiner, glctx = models
        mesh = trimesh.load(args.repo / item["stl"], force="mesh")
        mesh.apply_scale(0.001)
        estimator = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                                   scorer=scorer, refiner=refiner, debug=0, glctx=glctx,
                                   debug_dir=str(args.out / "debug"))
        if args.upright_prior_deg is not None:
            apply_upright_prior(estimator, args.up_stl[item["object_id"]], args.up_cam, args.upright_prior_deg)
        cache[item["object_id"]] = (mesh, estimator, np.asarray(mesh.sample(2000)))
    return cache[item["object_id"]]


def apply_upright_prior(estimator, up_stl: np.ndarray, up_cam: np.ndarray, max_deg: float) -> None:
    import torch

    from perception.pose.grasp_tolerance import upright_hypotheses

    keep = upright_hypotheses(estimator.rot_grid.cpu().numpy(), up_stl, up_cam, max_deg)
    if not keep.any():
        raise ValueError(f"no start rotation within {max_deg} deg of upright")
    estimator.rot_grid = estimator.rot_grid[torch.as_tensor(keep, device=estimator.rot_grid.device)]


def noisy(depth: np.ndarray, sigma_mm: float, rng: np.random.Generator) -> np.ndarray:
    """Depth with i.i.d. Gaussian noise on valid pixels (sigma in mm); 0 stays 0 (no reading)."""
    if sigma_mm <= 0:
        return depth
    return np.where(depth > 0, depth + rng.normal(0.0, sigma_mm / 1000.0, depth.shape), 0.0).astype(np.float32)


def estimate_rows(args, state: dict, chunk: Path, stem: str, K: np.ndarray, cfg: dict) -> list[dict]:
    """--repeat registrations of each scored object in one scene, until a part has --per-part objects.

    Fresh depth noise is drawn for every registration when --depth-noise-mm is set.
    """
    items = [i for i in scene_items(chunk, stem, cfg) if state["counts"].get(i["object_id"], 0) < args.per_part
             and (args.rest is None or i["rest"] == args.rest)]
    if not items:
        return []
    rgb, depth, seg = load_views(chunk, stem)
    models, cache, rng = state["models"], state["cache"], state["rng"]
    rows = []
    for item in items:
        mask = seg == item["seg_value"]
        if mask.sum() < 30 or state["counts"].get(item["object_id"], 0) >= args.per_part:
            continue
        state["counts"][item["object_id"]] = state["counts"].get(item["object_id"], 0) + 1
        _, estimator, points = make_estimator(args, item, cache, models)
        for run_index in range(args.repeat):
            estimate = np.asarray(estimator.register(K=K, rgb=rgb, depth=noisy(depth, args.depth_noise_mm, rng),
                                                     ob_mask=mask, iteration=args.iterations))
            rows.append({"scene": f"{chunk.name}/{stem}", "uid": item["uid"], "object_id": item["object_id"],
                         "rest": item["rest"], "run": run_index, "depth_noise_mm": args.depth_noise_mm,
                         "estimate": estimate.tolist(), "truth": item["truth"].tolist(),
                         **errors(estimate, item["truth"], points)})
    return rows


def run(args: argparse.Namespace) -> None:
    import nvdiffrast.torch as dr
    from estimater import PoseRefinePredictor, ScorePredictor

    cfg = load_camera_config(args.repo)
    K = intrinsics(cfg, DEPTH_SIZE)
    if args.upright_prior_deg is not None:
        from perception.pose.grasp_tolerance import load_grasp_points

        # the calibrated (unjittered) camera: the cell knows its calibration, not each scene's jitter
        nominal = np.linalg.inv(camera_to_world_cv(cfg, {"translate_mm": [0, 0, 0], "rotate_deg": [0, 0, 0]}))
        args.up_cam = nominal[:3, :3] @ np.array([0.0, 0.0, 1.0])
        args.up_stl = {k: v["up_stl"] for k, v in load_grasp_points(args.grasp_frames).items()}
    state = {"models": (ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()), "cache": {},
             "rng": np.random.default_rng(args.seed), "counts": {}}
    rows: list[dict] = []
    for data in args.data:
        stems = [(c, p.stem) for c in sorted(data.glob("chunk_*")) for p in sorted((c / "labels").glob("*.json"))]
        for chunk, stem in stems[: args.scenes]:
            rows.extend(estimate_rows(args, state, chunk, stem, K, cfg))
    write_summary(rows, args.out)


def write_summary(rows: list[dict], out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    adds = np.array([r["adds_mm"] for r in rows]) if rows else np.zeros(0)
    summary = {"estimates": len(rows), "parts": len({r["object_id"] for r in rows})}
    for rest in ("installed", "fallen"):
        subset = [r["adds_mm"] for r in rows if r["rest"] == rest]
        summary[f"{rest}_estimates"] = len(subset)
        if subset:
            summary[f"{rest}_adds_mm_median"] = float(np.median(subset))
    if len(adds):
        summary.update({"adds_mm_median": float(np.median(adds)),
                        "share_adds_under_2mm": float((adds < 2).mean()), "share_adds_under_5mm": float((adds < 5).mean()),
                        "trans_mm_median": float(np.median([r["trans_mm"] for r in rows])),
                        "per_part_adds_mm_median": {p: float(np.median([r["adds_mm"] for r in rows if r["object_id"] == p]))
                                                    for p in sorted({r["object_id"] for r in rows})}})
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, required=True, help="this repository (STL files, camera config)")
    ap.add_argument("--data", type=Path, nargs="+", required=True,
                    help="test-bed set directories holding chunk_* (e.g. N S)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=60)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=1, help="registrations per object, for repeatability")
    ap.add_argument("--depth-noise-mm", type=float, default=0.0, help="i.i.d. Gaussian depth noise per pixel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-part", type=int, default=10**9, help="stop scoring a part after this many objects")
    ap.add_argument("--rest", choices=("installed", "fallen"), help="score only parts in this rest state")
    ap.add_argument("--upright-prior-deg", type=float,
                    help="start only from rotations whose installed up is within this angle of the table's up")
    ap.add_argument("--grasp-frames", type=Path, help="cad/grasp_frames.json (installed up axis per part)")
    ap.add_argument("--check-geometry", action="store_true", help="only verify the camera/object poses against boxes")
    args = ap.parse_args()
    check_geometry(args) if args.check_geometry else run(args)


if __name__ == "__main__":
    main()
