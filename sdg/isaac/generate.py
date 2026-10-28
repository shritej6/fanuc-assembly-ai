"""Render one chunk of scene specs with Isaac Sim Replicator. Runs under Isaac's python (Rosie: inside the container).

UNTESTED until the Rosie pilot (T08). Written against the Isaac Sim 6.x API; the pilot checks every output type.

    $ISAAC_PY sdg/isaac/generate.py --specs specs/A/chunk_0000.jsonl --out $DATA_ROOT/synthetic/A/chunk_0000

Output per chunk:  rgb/<cam>/<index>.jpg   depth/<cam>/<index>.png (uint16 mm)   seg/<cam>/<index>.png (uint16 uid+1)
                   labels/<index>.json (written last: a scene with labels is complete)   _DONE.json (timing)
A re-run skips scenes that already have labels and skips the chunk if _DONE.json exists.
"""

import argparse
import json
import math
import os
import sys
import time
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

from sdg.isaac.stage import UF0_ABOVE_BASE_M, SceneStage
from sdg.postprocess import apply_post, box_labels, depth_mm_u16, instance_mask, mask_box_labels, resize_nearest

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

    def pose(self, arm: dict) -> None:
        q = self.park if arm["mode"] == "parked" else self._solve(arm["tcp_uf0"])
        full = self.art.get_joint_positions().copy()
        full[self.idx] = q
        for i, v in self.jaw.items():
            full[i] = v
        self.art.set_joint_positions(full)

    def _solve(self, tcp: list[float]) -> np.ndarray:
        """UF0 [x y z w p r] (mm/deg, FANUC fixed-XYZ) -> joints. Base frame = UF0 shifted down 330 mm, metres."""
        x, y, z, w, p, r = tcp
        rot = _rz(r) @ _ry(p) @ _rx(w)
        target = np.array([x, y, z]) / 1000.0 + (0.0, 0.0, UF0_ABOVE_BASE_M)
        q, ok = ik.solve(self.park, target, rot @ (0, 0, 1), self.limits, tool_offset=self.tool,
                         spread_dir=rot @ (0, 1, 0))
        if not ok:
            raise RuntimeError(f"IK failed for TCP {tcp}")
        return q


def _rx(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(d):
    c, s = math.cos(math.radians(d)), math.sin(math.radians(d))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


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
                ("instance_segmentation", {"colorize": False}))}
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
        seg = a["instance_segmentation"].get_data()
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
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "labels" / f"{idx}.json").write_text(json.dumps(labels))
    return len(spec["cameras"])


def _save(path: Path, img, params=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img, params or []):
        raise OSError(f"could not write {path}")


def main() -> None:
    if (args.out / "_DONE.json").exists():
        print(f"[sdg] {args.out} already done")
        return
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
        if (args.out / "labels" / f"{spec['index']:06d}.json").exists():
            continue
        paths = scene.load(spec)
        arm.pose(spec["arm"])
        cams.jitter(spec["camera_jitter"])
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        images += write_scene(spec, cams, paths, args.out)
    wall = time.time() - t0
    done = {"scenes": len(specs), "images_this_run": images, "wall_s": round(wall, 1),
            "s_per_image": round(wall / images, 3) if images else None, "rt_subframes": args.rt_subframes,
            "container": os.environ.get("ISAAC_SIF", "unknown"), "limit": args.limit}
    if not args.limit:
        (args.out / "_DONE.json").write_text(json.dumps(done, indent=1))
    print(f"[sdg] {json.dumps(done)}")


try:
    main()
finally:
    app.close()
