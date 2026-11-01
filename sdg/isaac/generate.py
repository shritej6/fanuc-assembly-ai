"""Render one chunk of scene specs with Isaac Sim Replicator.
Authored by Shritej Gunda.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STARTER = REPO / "sim" / "starter"

ap = argparse.ArgumentParser()
ap.add_argument("--specs", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--limit", type=int, default=0, help="render only the first N scenes (pilot)")
ap.add_argument("--rt-subframes", type=int, default=4)
args = ap.parse_args()
args.out = args.out.resolve()     # Kit changes the working directory

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})

import cv2
import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from pxr import Gf, UsdGeom

sys.path[:0] = [str(STARTER), str(REPO)]
import cell
import ik
import scene_config as C

from sdg.chunk_marks import chunk_done, scene_complete, specs_digest, write_json_atomic
from sdg.isaac.stage import SceneStage
from sdg.postprocess import apply_post, box_labels, depth_mm_u16, instance_mask, mask_box_labels, resize_nearest
from sdg.wrist_ik import solve_tcp

CAM_PATHS = {"table": cell.TABLE_CAM_PATH, "side": cell.SIDE_CAM_PATH, "wrist": cell.WRIST_CAM_PATH}
CAM_CFG = {"table": C.TABLE_CAM, "side": C.SIDE_CAM, "wrist": C.WRIST_CAM}
DEPTH_SIZE = {"table": (640, 400), "wrist": (848, 480)}     # what the real cell sends; side has no depth


class Arm:
    def __init__(self):
        self.art = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
        self.art.initialize()
        names = list(self.art.dof_names)
        self.idx = [names.index(f"J{i}") for i in range(1, 7)]
        self.jaw = {names.index(j): v for j, v in cell.jaw_targets(C.JAW_OPEN).items()}
        self.limits = np.deg2rad([C.JOINT_LIMITS_DEG[f"J{i}"] for i in range(1, 7)])
        ik.set_tool_axis(C.TOOL_AXIS)
        ik.set_spread_axis(C.FINGER_SPREAD_AXIS)
        self.tool = tuple(np.array(C.TOOL_AXIS) * cell.tip_offset())
        self.park = np.deg2rad([C.CAPTURE_POSE_DEG[f"J{i}"] for i in range(1, 7)])
        self.home = np.deg2rad([C.HOME_POSE_DEG[f"J{i}"] for i in range(1, 7)])

    def pose(self, arm: dict) -> None:
        q = self.park if arm["mode"] == "parked" else self._solve(arm["tcp_uf0"])
        full = self.art.get_joint_positions().copy()
        full[self.idx] = q
        for i, v in self.jaw.items():
            full[i] = v
        self.art.set_joint_positions(full)

    def _solve(self, tcp: list[float]) -> np.ndarray:
        q = solve_tcp(ik, tcp, self.home, self.park, self.limits, self.tool)
        if q is None:
            raise RuntimeError(f"IK failed for TCP {tcp} from all seeds")
        return q


class Cameras:
    def __init__(self, stage):
        self.stage = stage
        self.base = {}
        self.annot = {}
        for name, path in CAM_PATHS.items():
            op = UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetOrderedXformOps()[0]
            self.base[name] = (op, op.Get())
            rp = rep.create.render_product(path, tuple(CAM_CFG[name]["resolution"]))
            self.annot[name] = {k: rep.AnnotatorRegistry.get_annotator(k, init_params=p) for k, p in (
                ("rgb", None), ("bounding_box_2d_tight", None), ("distance_to_image_plane", None),
                ("instance_id_segmentation", {"colorize": False}))}
            for a in self.annot[name].values():
                a.attach([rp])

    def jitter(self, jitter: dict) -> None:
        for name, (op, base) in self.base.items():
            j = jitter.get(name, {"translate_mm": [0, 0, 0], "rotate_deg": [0, 0, 0]})
            rx, ry, rz = j["rotate_deg"]
            delta = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), rx) * Gf.Rotation(Gf.Vec3d(0, 1, 0), ry)
                                            * Gf.Rotation(Gf.Vec3d(0, 0, 1), rz))
            delta.SetTranslateOnly(Gf.Vec3d(*(np.array(j["translate_mm"]) / 1000.0)))
            op.Set(delta * base)


def write_scene(spec: dict, cams: Cameras, paths: dict, out: Path) -> int:
    objects = {o["uid"]: o for o in spec["objects"]}
    idx = f"{spec['index']:06d}"
    labels = {"spec": spec, "cameras": {}}
    for k, cam in enumerate(spec["cameras"]):
        a = cams.annot[cam]
        rgb = apply_post(a["rgb"].get_data()[:, :, :3], spec["post"], spec["seed"] * 7 + k)
        _save(out / "rgb" / cam / f"{idx}.jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
              [cv2.IMWRITE_JPEG_QUALITY, spec["post"]["jpeg_quality"]])
        seg = a["instance_id_segmentation"].get_data()   # per-prim ids keyed by USD path; needs no semantic labels
        if os.environ.get("SDG_DEBUG"):
            info = seg["info"]
            print(f"[sdg-debug] {cam} ids={np.unique(seg['data'])[:20].tolist()} dtype={seg['data'].dtype} "
                  f"info_keys={list(info)} idToLabels={dict(list(info.get('idToLabels', {}).items())[:8])} "
                  f"paths_sample={list(paths)[:3]}", flush=True)
        mask = instance_mask(seg["data"], seg["info"]["idToLabels"], paths)
        _save(out / "seg" / cam / f"{idx}.png", mask)
        if cam in DEPTH_SIZE and "depth" in spec["outputs"]:
            depth = depth_mm_u16(a["distance_to_image_plane"].get_data())
            _save(out / "depth" / cam / f"{idx}.png", resize_nearest(depth, DEPTH_SIZE[cam]))
        boxes = mask_box_labels(mask, objects)
        bb = a["bounding_box_2d_tight"].get_data()   # adds occlusion where Replicator reports the object
        occ = {b["uid"]: b["occlusion"] for b in box_labels(bb["data"], bb["info"]["primPaths"], paths, objects)}
        for b in boxes:
            b["occlusion"] = occ.get(b["uid"])
        labels["cameras"][cam] = boxes
    write_json_atomic(out / "labels" / f"{idx}.json", labels)   # last: the label is what marks the scene done
    return len(spec["cameras"])


def _save(path: Path, img, params=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img, params or []):
        raise OSError(f"could not write {path}")


def _deployed_commit() -> str:
    """Commit the deploy step wrote next to the code (git archive copies have no .git)."""
    path = REPO / "DEPLOYED_COMMIT"
    return path.read_text().strip() if path.exists() else "unknown"


def main() -> None:
    if chunk_done(args.out, args.specs):
        print(f"[sdg] {args.out} already done from these specs")
        return
    (args.out / "_FAILED.txt").unlink(missing_ok=True)   # an earlier attempt's failure is not this run's
    specs = [json.loads(line) for line in args.specs.read_text().splitlines() if line.strip()]
    specs = specs[: args.limit] if args.limit else specs
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    scene = SceneStage(stage, REPO, C.TABLE_TOP_Z, table_z_uf0=specs[0]["table_z_uf0"])
    world.reset()
    arm, cams = Arm(), Cameras(stage)
    for _ in range(60):          # let MDL / shaders compile before the first capture
        app.update()
    t0, images = time.time(), 0
    for spec in specs:
        if scene_complete(args.out, spec, set(DEPTH_SIZE)):
            continue
        paths = scene.load(spec)
        arm.pose(spec["arm"])
        cams.jitter(spec["camera_jitter"])
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        images += write_scene(spec, cams, paths, args.out)
    wall = time.time() - t0
    done = {"scenes": len(specs), "images_this_run": images, "wall_s": round(wall, 1),
            "s_per_image": round(wall / images, 3) if images else None, "rt_subframes": args.rt_subframes,
            "container": os.environ.get("ISAAC_SIF", "unknown"), "limit": args.limit,
            "code_commit": _deployed_commit(), "specs_sha256": specs_digest(args.specs),
            "depth_cams": sorted(DEPTH_SIZE)}
    if not args.limit:
        write_json_atomic(args.out / "_DONE.json", done)
    print(f"[sdg] {json.dumps(done)}")


# SimulationApp.close() ends the process, which would swallow an exception raised in main(). Print it and leave
# _FAILED.txt first, so an unattended run never fails silently.
try:
    main()
except Exception:  # noqa: BLE001 - process boundary: record every failure, then exit
    trace = traceback.format_exc()
    print(f"[sdg] FAILED\n{trace}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "_FAILED.txt").write_text(trace)
finally:
    sys.stdout.flush()
    app.close()
