"""Replay a `plan_assembly.py` plan in Isaac Sim and write captioned video frames. Runs under Isaac's python.

The scene is rebuilt from the same label spec with the SDG stage code, so parts, base, lights and cameras match the
training renders. Joints are set per frame (kinematic replay, no contact physics); the held part follows the plan.
Each frame: table camera (large), side camera and wrist camera, with the plan step as a caption.

    $ISAAC_PY sim/demo/render_assembly.py --plan plan.json --out frames/
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STARTER = REPO / "sim" / "starter"

ap = argparse.ArgumentParser()
ap.add_argument("--plan", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--rt-subframes", type=int, default=4)
ap.add_argument("--every", type=int, default=1, help="render every n-th frame (preview)")
args = ap.parse_args()
args.plan, args.out = args.plan.resolve(), args.out.resolve()     # Kit changes the working directory

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
import scene_config as C

from sdg.isaac.stage import SCENE_ROOT, SceneStage

VIEWS = {"table": (cell.TABLE_CAM_PATH, C.TABLE_CAM), "wrist": (cell.WRIST_CAM_PATH, C.WRIST_CAM)}
CLOSE_CAM_PATH = "/World/DemoCloseCam"
CLOSE_CAM_OFFSET_M = (0.25, -0.155, 0.16)     # from the product base: the far side, so the gripper comes toward us
CLOSE_RES = (1920, 1080)
CAPTION_H = 64


def base_to_world(m) -> Gf.Matrix4d:
    w = np.array(m, dtype=np.float64)
    w[2, 3] += C.TABLE_TOP_Z
    return Gf.Matrix4d(w.T.tolist())        # USD is row-vector: transpose


def add_close_camera(stage, base_m) -> str:
    """A video-only camera looking at the product being built (not one of the cell's three cameras)."""
    target = Gf.Vec3d(base_m[0], base_m[1], base_m[2] + C.TABLE_TOP_Z + 0.035)   # mid-height of a tall part
    eye = target + Gf.Vec3d(*CLOSE_CAM_OFFSET_M)
    cam = UsdGeom.Camera.Define(stage, CLOSE_CAM_PATH)
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955 * CLOSE_RES[1] / CLOSE_RES[0])
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
    cam.AddTransformOp().Set(view.GetInverse())
    return CLOSE_CAM_PATH


def _label(frame, text, x, y, scale=0.7):
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(frame, (x - 8, y - h - 8), (x + w + 8, y + 8), (20, 20, 20), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def compose(images: dict, caption: str) -> np.ndarray:
    big = cv2.resize(images["close"], (1280, 720), interpolation=cv2.INTER_AREA)
    table = cv2.resize(images["table"], (640, 360), interpolation=cv2.INTER_AREA)
    wrist = cv2.resize(images["wrist"], (640, 360), interpolation=cv2.INTER_AREA)
    frame = np.zeros((720 + CAPTION_H, 1920, 3), np.uint8)
    frame[CAPTION_H:, :1280], frame[CAPTION_H:CAPTION_H + 360, 1280:], frame[CAPTION_H + 360:, 1280:] = big, table, wrist
    cv2.putText(frame, caption, (24, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "Isaac Sim - motion scripted from scene ground truth", (1290, 42), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (190, 190, 190), 1, cv2.LINE_AA)
    _label(frame, "close-up", 24, CAPTION_H + 40)
    _label(frame, "table camera", 1300, CAPTION_H + 36)
    _label(frame, "wrist camera", 1300, CAPTION_H + 396)
    return frame


def main() -> None:
    plan = json.loads(args.plan.read_text())
    spec = plan["spec"]
    world = World(stage_units_in_meters=1.0)
    stage = omni.usd.get_context().get_stage()
    cell.build(stage, with_objects=False)
    scene = SceneStage(stage, REPO, C.TABLE_TOP_Z, table_z_uf0=spec["table_z_uf0"])
    uids = {uid: path for path, uid in scene.load(spec).items()}
    world.reset()
    arm = SingleArticulation(prim_path=C.ROBOT_PRIM, name="er4ia")
    arm.initialize()
    names = list(arm.dof_names)
    arm_idx = [names.index(f"J{i}") for i in range(1, 7)]
    annot = {}
    views = {**{v: (path, tuple(cfg["resolution"])) for v, (path, cfg) in VIEWS.items()},
             "close": (add_close_camera(stage, plan["base_m"]), CLOSE_RES)}
    for view, (path, resolution) in views.items():
        rp = rep.create.render_product(path, resolution)
        annot[view] = rep.AnnotatorRegistry.get_annotator("rgb")
        annot[view].attach([rp])
    for _ in range(60):          # let MDL / shaders compile before the first frame
        app.update()
    args.out.mkdir(parents=True, exist_ok=True)
    ops = {uid: UsdGeom.Xformable(stage.GetPrimAtPath(p)).GetOrderedXformOps()[0] for uid, p in uids.items()}
    for n, frame in enumerate(plan["frames"]):
        for uid, pose in frame["parts"].items():
            ops[uid].Set(base_to_world(pose))
        if n % args.every:
            continue
        full = arm.get_joint_positions().copy()
        full[arm_idx] = frame["q"]
        for joint, value in cell.jaw_targets(frame["jaw"]).items():
            full[names.index(joint)] = value
        arm.set_joint_positions(full)
        rep.orchestrator.step(rt_subframes=args.rt_subframes)
        images = {v: cv2.cvtColor(a.get_data()[:, :, :3], cv2.COLOR_RGB2BGR) for v, a in annot.items()}
        if not cv2.imwrite(str(args.out / f"f{n:05d}.jpg"), compose(images, frame["caption"]),
                           [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f"could not write frame {n}")
    print(f"[demo] wrote {len(list(args.out.glob('f*.jpg')))} frames to {args.out} (scene root {SCENE_ROOT})")


try:
    main()
except Exception:  # noqa: BLE001 - process boundary: SimulationApp.close() would otherwise swallow it
    print(f"[demo] FAILED\n{traceback.format_exc()}", flush=True)
    (args.out / "_FAILED.txt").parent.mkdir(parents=True, exist_ok=True)
    (args.out / "_FAILED.txt").write_text(traceback.format_exc())
finally:
    sys.stdout.flush()
    app.close()
