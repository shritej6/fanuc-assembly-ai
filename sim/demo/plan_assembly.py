"""Plan a scripted assembly of one product from a rendered test-bed scene: every frame's joints, jaw and part poses.

numpy only (runs anywhere); `render_assembly.py` replays the plan in Isaac Sim. Poses come from the scene's ground
truth, not from perception, so the video shows the motion plan and the insertion rule, not model accuracy.

Rules followed (DECISIONS 2026-10-28): parts stay in their installed orientation and the tool only points straight
down, so the arm changes nothing but yaw; the part is yawed over the base before it goes straight down onto it;
grasp width comes from the CAD and the measured fingers. Each part is picked by its top grip tab: the thin side of
the tab's cross-section 0.5-3.5 mm below the top, the jaws closing across it. A part whose tab does not fit the
measured fingertip gaps, or any frame the IK cannot reach to 1 mm with the jaws aligned, stops the plan.

    python sim/demo/plan_assembly.py --label N/chunk_0000/labels/000003.json --product 1 --out plan.json
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np

from sdg.wrist_ik import UF0_ABOVE_BASE_M, _starter_robot

REPO = Path(__file__).resolve().parents[2]
FPS = 30
TIP_GAP_MM = (4.0, 15.0)        # v0.5 measured fingertip gap closed / open (scene_config: 4 +/- 1, 15 +/- 2 mm)
TIP_GAP_TOL_MM = (1.0, 2.0)
SQUEEZE_MM = 1.0                # the closed gap must be at least this far inside the tab to hold it
TAB_BAND_MM = (1.0, 3.0)        # slices below the part top where the jaws hold the tab
GRIP_DEPTH_MM = 3.0             # fingertips this far below the part top
HOVER_M = 0.10                  # approach / retreat height above pick and place
SEAT_HOVER_M = 0.012            # slow insertion starts this far above the seat
READY_TIP_M = np.array([0.30, 0.0, 0.24])   # tool down above the work area (the starter sort demo's home)


class PlanError(RuntimeError):
    """The scene cannot be assembled within the gripper or IK limits; nothing is rendered."""


def read_stl_mm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 80)
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n,
                        offset=84)
    return rec["v"].reshape(-1, 3).astype(np.float64)


def uf0_to_base(tf_mm) -> np.ndarray:
    """Scene transform (UF0, mm) -> robot base frame (m), the frame the starter's IK and FK use."""
    m = np.array(tf_mm, dtype=np.float64)
    m[:3, 3] = m[:3, 3] / 1000.0 + (0.0, 0.0, UF0_ABOVE_BASE_M)
    return m


def cross_section(world_m: np.ndarray, z: float) -> np.ndarray:
    """Points where the mesh's triangle edges cross the plane at height z (the outline of the slice)."""
    tri = world_m.reshape(-1, 3, 3)
    points = []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = tri[:, i], tri[:, j]
        crosses = (a[:, 2] - z) * (b[:, 2] - z) < 0
        t = (z - a[crosses, 2]) / (b[crosses, 2] - a[crosses, 2])
        points.append(a[crosses] + (b[crosses] - a[crosses]) * t[:, None])
    out = np.vstack(points)
    if len(out) < 3:
        raise PlanError(f"no grip tab: the CAD slice {z * 1000:.1f} mm is empty")
    return out


def grip_tab(verts_mm: np.ndarray, part_in_base: np.ndarray) -> dict:
    """Tab grasp in the PART frame: fingertip point and the jaw-closing direction, from the CAD cross-section."""
    world = verts_mm / 1000.0 @ part_in_base[:3, :3].T + part_in_base[:3, 3]
    top = world[:, 2].max()
    band = np.vstack([cross_section(world, top - depth / 1000) for depth in TAB_BAND_MM])
    best = None
    for deg in np.arange(0.0, 180.0, 1.0):
        d = np.array([math.cos(math.radians(deg)), math.sin(math.radians(deg))])
        proj = band[:, :2] @ d
        width = proj.max() - proj.min()
        if best is None or width < best[0]:
            centre_along = (proj.max() + proj.min()) / 2
            perp = band[:, :2] @ np.array([-d[1], d[0]])
            centre = d * centre_along + np.array([-d[1], d[0]]) * (perp.max() + perp.min()) / 2
            best = (width, d, centre)
    width_mm, d, centre = best[0] * 1000, best[1], best[2]
    closed_max, open_min = TIP_GAP_MM[0] + TIP_GAP_TOL_MM[0], TIP_GAP_MM[1] - TIP_GAP_TOL_MM[1]
    if not closed_max + SQUEEZE_MM <= width_mm <= open_min:
        raise PlanError(f"grip tab {width_mm:.1f} mm outside the fingers' {closed_max + SQUEEZE_MM:.0f}-"
                        f"{open_min:.0f} mm")
    inv = np.linalg.inv(part_in_base)
    tip_world = np.array([centre[0], centre[1], top - GRIP_DEPTH_MM / 1000, 1.0])
    return {"tip_part": (inv @ tip_world)[:3], "spread_part": inv[:3, :3] @ np.array([d[0], d[1], 0.0]),
            "width_mm": width_mm}


def jaw_for_width(width_mm: float) -> float:
    """Per-side jaw opening (m) that brings the simulated straight finger faces (0-12 mm gap) onto the tab."""
    return width_mm / 2000.0


class Planner:
    def __init__(self):
        self.ik, robot = _starter_robot()
        import cell
        import fk
        import scene_config as C

        self.fk, self.C = fk, C
        self.limits, self.tool = robot["limits"], robot["tool_offset"]
        self.spread_axis = np.array(C.FINGER_SPREAD_AXIS, float)
        self.q = robot["home"].copy()
        self.home_tip = READY_TIP_M
        self.solve(READY_TIP_M, np.array([0.0, 1.0, 0.0]), "the ready pose")
        self.frames: list[dict] = []
        self.jaw_open = C.JAW_OPEN
        del cell

    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.fk.tool_pose(self.q, self.tool)

    def solve(self, tip: np.ndarray, spread: np.ndarray, where: str, free_sign: bool = True) -> None:
        """free_sign: empty parallel jaws close the same either way round; a held part fixes the sign."""
        down = np.array([0.0, 0.0, -1.0])
        r, _ = self.tool_pose()
        if free_sign and np.dot(r @ self.spread_axis, spread) < 0:
            spread = -spread
        for iters, damping in ((200, 0.05), (600, 0.02)):
            q, ok = self.ik.solve(self.q, tip, down, self.limits, tool_offset=self.tool, spread_dir=spread,
                                  iters=iters, damping=damping)
            r, actual = self.fk.tool_pose(q, self.tool)
            if ok and np.linalg.norm(actual - tip) < 0.001 and abs(np.dot(r @ self.spread_axis, spread)) > 0.999:
                self.q = q
                return
        raise PlanError(f"IK cannot reach {where} at {np.round(tip, 4).tolist()} m with the jaws aligned")

    def move(self, tip1, spread1, seconds, jaw0, jaw1, caption, carried=None):
        r0, tip0 = self.tool_pose()
        s0 = r0 @ self.spread_axis
        if carried is None and np.dot(s0, spread1) < 0:
            s0 = -s0
        a0, a1 = math.atan2(s0[1], s0[0]), math.atan2(spread1[1], spread1[0])
        turn = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        n = max(1, round(seconds * FPS))
        for i in range(n):
            t = (i + 1) / n
            e = t * t * (3 - 2 * t)
            a = a0 + turn * e
            self.solve(tip0 + (np.asarray(tip1) - tip0) * e, np.array([math.cos(a), math.sin(a), 0.0]), caption,
                       free_sign=carried is None)
            frame = {"q": self.q.tolist(), "jaw": jaw0 + (jaw1 - jaw0) * e, "caption": caption, "parts": {}}
            if carried:
                uid, tool_to_part = carried
                r, tip = self.tool_pose()
                tool = np.eye(4)
                tool[:3, :3], tool[:3, 3] = r, tip
                frame["parts"][uid] = (tool @ tool_to_part).tolist()
            self.frames.append(frame)

    def hold(self, seconds, caption, carried=None):
        r, tip = self.tool_pose()
        self.move(tip, r @ self.spread_axis, seconds, self.frames[-1]["jaw"] if self.frames else self.jaw_open,
                  self.frames[-1]["jaw"] if self.frames else self.jaw_open, caption, carried)

    def assemble_part(self, obj: dict, target_in_base: np.ndarray, verts_mm: np.ndarray, name: str) -> dict:
        start = uf0_to_base(obj["transform_uf0"])
        tab = grip_tab(verts_mm, start)
        pick_tip = (start @ np.append(tab["tip_part"], 1))[:3]
        pick_spread = start[:3, :3] @ tab["spread_part"]
        place_tip = (target_in_base @ np.append(tab["tip_part"], 1))[:3]
        place_spread = target_in_base[:3, :3] @ tab["spread_part"]
        up = np.array([0, 0, HOVER_M])
        seat = np.array([0, 0, SEAT_HOVER_M])
        grip = jaw_for_width(tab["width_mm"])
        o = self.jaw_open
        self.move(pick_tip + up, pick_spread, 1.6, o, o, f"{name}: move over the part")
        self.move(pick_tip, pick_spread, 1.0, o, o, f"{name}: descend to the grip tab ({tab['width_mm']:.1f} mm)")
        self.move(pick_tip, pick_spread, 0.6, o, grip, f"{name}: close the fingers")
        r, tip = self.tool_pose()
        tool = np.eye(4)
        tool[:3, :3], tool[:3, 3] = r, tip
        carried = (obj["uid"], np.linalg.inv(tool) @ start)
        place_spread = place_spread * np.sign(np.dot(r @ self.spread_axis, pick_spread))   # jaws keep their side
        self.move(pick_tip + up, pick_spread, 0.9, grip, grip, f"{name}: lift", carried)
        self.move(place_tip + up, place_spread, 1.8, grip, grip, f"{name}: carry and yaw the holes over the pegs",
                  carried)
        self.move(place_tip + seat, place_spread, 1.0, grip, grip, f"{name}: lower above the seat", carried)
        self.move(place_tip, place_spread, 1.2, grip, grip, f"{name}: insert straight down", carried)
        seated = np.array(self.frames[-1]["parts"][obj["uid"]])
        seat_error_mm = float(np.linalg.norm(seated[:3, 3] - target_in_base[:3, 3]) * 1000)
        seat_yaw_deg = float(np.degrees(np.arccos(np.clip((np.trace(seated[:3, :3].T @ target_in_base[:3, :3]) - 1)
                                                          / 2, -1, 1))))
        self.move(place_tip, place_spread, 0.5, grip, o, f"{name}: release")
        for f in self.frames[-round(0.5 * FPS):]:
            f["parts"][obj["uid"]] = target_in_base.tolist()
        self.move(place_tip + up, place_spread, 0.9, o, o, f"{name}: retreat")
        return {"uid": obj["uid"], "object_id": obj["object_id"], "tab_width_mm": round(tab["width_mm"], 2),
                "yaw_change_deg": round(math.degrees(math.atan2(*_cross_dot(pick_spread, place_spread))), 1),
                "seat_error_mm": round(seat_error_mm, 3), "seat_rotation_error_deg": round(seat_yaw_deg, 3)}


def _cross_dot(a, b):
    return a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]


def plan(label_path: Path, product: int, names: dict) -> dict:
    spec = json.loads(label_path.read_text())["spec"]
    geometry = json.loads((REPO / "sdg/data/object_geometry.json").read_text())["objects"]
    objs = spec["objects"]
    base = next(o for o in objs if o["kind"] == "base" and o["product"] == product)
    loose = sorted((o for o in objs if o["kind"] == "part" and o["product"] == product and o["state"] == "on_table"
                    and o.get("rest") == "installed" and o["variant"] == "ok"),
                   key=lambda o: geometry[o["object_id"]]["install_step"])
    if not loose:
        raise PlanError(f"product {product} has no loose parts standing as installed in {label_path}")
    p = Planner()
    base_in_base = uf0_to_base(base["transform_uf0"])
    steps = []
    p.hold(1.0, f"Product {product}: {len(loose)} parts to install")
    for obj in loose:
        g = geometry[obj["object_id"]]
        target = base_in_base @ uf0_to_base_offset(g["installed_in_base_frame"])
        verts = read_stl_mm(REPO / obj["stl"])
        steps.append(p.assemble_part(obj, target, verts, f"{obj['object_id']} {names.get(obj['object_id'], '')}"
                                     .strip()))
    p.move(p.home_tip, p.tool_pose()[0] @ p.spread_axis, 1.6, p.jaw_open, p.jaw_open, "Return home")
    p.hold(1.0, f"Product {product} assembled")
    return {"label": str(label_path), "spec": spec, "product": product, "base_m": base_in_base[:3, 3].tolist(), "fps": FPS, "steps": steps, "frames": p.frames,
            "source": "scene ground truth (no perception in the loop)"}


def uf0_to_base_offset(tf_mm) -> np.ndarray:
    m = np.array(tf_mm, dtype=np.float64)
    m[:3, 3] /= 1000.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", type=Path, required=True, help="a rendered scene's labels/<index>.json")
    ap.add_argument("--product", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    taxonomy = json.loads((REPO / "contracts/taxonomy.json").read_text())["objects"]
    names = {o["object_id"]: o.get("name", "") for o in taxonomy}
    result = plan(a.label, a.product, names)
    a.out.write_text(json.dumps(result))
    print(json.dumps({"frames": len(result["frames"]), "steps": result["steps"]}, indent=1))


if __name__ == "__main__":
    main()
