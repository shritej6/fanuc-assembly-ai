"""Layout dict -> UF0 poses for every object in the sim cell. numpy only (runs and is tested off Isaac).

Frames (INTERFACES.md): API = UF0 mm / deg. The sim world is metres with the robot base mount at (0, 0, TABLE_TOP_Z)
and UF0's origin 330 mm above the base mount, the same mapping `sdg/isaac/stage.py` renders with:
    world_m = (x / 1000, y / 1000, TABLE_TOP_Z + 0.330 + z / 1000)

Object poses reuse `sdg/data/object_geometry.json` (the geometry the synthetic data was rendered with):
- `installed`: the part stands in its assembled orientation (`installed_rest`), any yaw.
- `fallen`: the most likely stable pose whose up axis differs from the installed one.
- `on:<base uid>`: seated on that base (`installed_in_base_frame`), i.e. already assembled.
- a base is always placed upright (`bounds_mm` lifted to the table).
x, y in a layout entry is where the STL origin of the posed part lands, exactly as `sdg/scene_sampler.py` does.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
UF0_ABOVE_BASE_M = 0.330
PLA_DENSITY_G_PER_MM3 = 1.24e-3     # PLA 1.24 g/cm3 (INTERFACES.md objects.yaml rule)
PARK_ORIGIN_MM = (1600.0, 1200.0)   # floor-level park grid for unused pool objects: behind the table camera (+x) and the
                                    # side camera (+y), so no camera sees it
PARK_PITCH_MM = 90.0


OBJECTS_YAML = REPO / "sim/assembly_lab/objects/objects.yaml"


def load_local_origins() -> dict:
    """{part type: installed_origin_mm in its base frame} from agent 2's objects.yaml (layout x, y = that origin)."""
    if not OBJECTS_YAML.exists():
        return {}
    objs = (yaml.safe_load(OBJECTS_YAML.read_text()) or {}).get("objects", {})
    return {t: [float(v) for v in e["installed_origin_mm"]] for t, e in objs.items() if "installed_origin_mm" in e}


class LayoutError(ValueError):
    """A layout entry names an unknown object, an unknown base, or more copies of a part than the pool holds."""


def load_geometry(repo: Path = REPO) -> dict:
    return json.loads((repo / "sdg/data/object_geometry.json").read_text())["objects"]


def load_cell_layout(repo: Path = REPO) -> dict:
    return yaml.safe_load((repo / "sdg/configs/cell_layout.yaml").read_text())


def type_id(object_id: str) -> str:
    """'M4_02_Sensor_Insert' / 'M4_02#2' / 'M4_02' -> 'M4_02' (the key of object_geometry.json)."""
    head = object_id.split("#")[0]
    return "_".join(head.split("_")[:2]) if head.startswith("M") else head.split("_")[0]


# ---------------------------------------------------------------- transforms
def translation(x: float, y: float, z: float) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = (x, y, z)
    return t


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


def fanuc_rotation(w: float, p: float, r: float) -> np.ndarray:
    """FANUC fixed-XYZ angles (deg): W about X, then P about Y, then R about Z."""
    cw, sw = math.cos(math.radians(w)), math.sin(math.radians(w))
    cp, sp = math.cos(math.radians(p)), math.sin(math.radians(p))
    cr, sr = math.cos(math.radians(r)), math.sin(math.radians(r))
    rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


def wpr_from_rotation(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of fanuc_rotation, degrees. At p = +-90 (gimbal lock) w is set to 0."""
    p = -math.asin(max(-1.0, min(1.0, rot[2, 0])))
    if abs(math.cos(p)) < 1e-9:
        return 0.0, math.degrees(p), math.degrees(math.atan2(-rot[0, 1], rot[1, 1]))
    w = math.atan2(rot[2, 1], rot[2, 2])
    r = math.atan2(rot[1, 0], rot[0, 0])
    return math.degrees(w), math.degrees(p), math.degrees(r)


def pose6_from_tf(tf_mm: np.ndarray) -> list[float]:
    """4x4 UF0 mm -> [x, y, z, w, p, r] (mm, deg)."""
    return [*map(float, tf_mm[:3, 3]), *wpr_from_rotation(tf_mm[:3, :3])]


def uf0_to_world_m(tf_mm: np.ndarray, table_top_z: float) -> np.ndarray:
    m = np.array(tf_mm, dtype=np.float64)
    m[:3, 3] = m[:3, 3] / 1000.0 + (0.0, 0.0, table_top_z + UF0_ABOVE_BASE_M)
    return m


def world_m_to_uf0(tf_m: np.ndarray, table_top_z: float) -> np.ndarray:
    m = np.array(tf_m, dtype=np.float64)
    m[:3, 3] = (m[:3, 3] - (0.0, 0.0, table_top_z + UF0_ABOVE_BASE_M)) * 1000.0
    return m


def quat_xyzw(rot: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (x, y, z, w), the Isaac Lab 3.x order."""
    m = rot
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def rotation_from_quat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


# ---------------------------------------------------------------- mesh mass
def read_binary_stl(path: Path) -> np.ndarray:
    """Triangles (N, 3, 3) in the STL's units (mm for cad/stl)."""
    data = path.read_bytes()
    (n,) = struct.unpack_from("<I", data, 80)
    if len(data) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL")
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
    return rec["v"].astype(np.float64)


def mass_properties(tris_mm: np.ndarray) -> tuple[float, np.ndarray]:
    """Closed-mesh volume (mm3) and centre of mass (mm) by signed tetrahedra about the origin."""
    a, b, c = tris_mm[:, 0], tris_mm[:, 1], tris_mm[:, 2]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    total = float(vol.sum())
    if total <= 0:
        raise ValueError("mesh volume is not positive (open mesh or inverted normals)")
    com = ((a + b + c) / 4.0 * vol[:, None]).sum(axis=0) / total
    return total, com


def pla_mass_kg(stl: Path) -> tuple[float, np.ndarray]:
    vol, com = mass_properties(read_binary_stl(stl))
    return vol * PLA_DENSITY_G_PER_MM3 / 1000.0, com


# ---------------------------------------------------------------- layout
@dataclass
class Placement:
    uid: str                    # pool instance name, e.g. "M4_02" or "M4_02#2"
    type_id: str                # object_geometry.json key
    tf_uf0: np.ndarray          # 4x4, mm
    state: str                  # installed | fallen | on:<uid> | spare:<slot> | defect | parked
    role: str = "object"        # object | spare | defect | parked


@dataclass
class LayoutSolver:
    geometry: dict = field(default_factory=load_geometry)
    cell: dict = field(default_factory=load_cell_layout)
    origins: dict = field(default_factory=load_local_origins)

    @property
    def table_z(self) -> float:
        return float(self.cell["table_z"])

    def rest_tf(self, tid: str, state: str) -> np.ndarray:
        """Local transform that sets the object on z = 0 in the requested orientation, with the object's CAD local
        origin (objects/objects.yaml, agent 2's convention) on the layout x, y."""
        geo = self.geometry[tid]
        if tid.startswith("F"):
            return translation(0, 0, -geo["bounds_mm"][0][2])
        tf = self._part_rest(tid, state)
        o = tf @ self.origin_in_stl(tid)
        return translation(-o[0], -o[1], 0.0) @ tf

    def origin_in_stl(self, tid: str) -> np.ndarray:
        """Homogeneous STL-frame point of the part's CAD local origin (0 0 0 1 when objects.yaml has none)."""
        if tid not in self.origins:
            return np.array([0.0, 0.0, 0.0, 1.0])
        seat = np.array(self.geometry[tid]["installed_in_base_frame"])
        return np.linalg.inv(seat) @ np.array([*self.origins[tid], 1.0])

    def _part_rest(self, tid: str, state: str) -> np.ndarray:
        geo = self.geometry[tid]
        installed = np.array(geo["installed_rest"]["transform"])
        if state != "fallen":
            return installed
        up = installed[2, :3]
        fallen = [p for p in geo["stable_poses"] if not np.allclose(np.array(p["transform"])[2, :3], up, atol=1e-2)]
        if not fallen:
            raise LayoutError(f"{tid} has no resting pose other than its installed one")
        return np.array(max(fallen, key=lambda p: p["probability"])["transform"])

    def table_tf(self, tid: str, x: float, y: float, yaw: float, state: str) -> np.ndarray:
        return translation(x, y, self.table_z) @ rot_z(yaw) @ self.rest_tf(tid, state)

    def seat_tf_for(self, part: str, base: str, poses: dict) -> np.ndarray:
        """CAD installed pose (4x4 UF0 mm) of `part` on the base instance `base`, given the current body poses.

        `installed_in_base_frame` is in the base's STL frame, which is the base body frame (sdg/scene_sampler.py and
        sdg/tools/build_clearance.py apply it the same way; F0x STL z = 0 is the seating face). An earlier version
        also multiplied by inv(rest_tf(base)), which put every seated part 8 mm low, inside the base (measured: the
        spawned M4_01 touched the mat through F04 with 3.9 mm SDF penetration and never settled)."""
        seat = np.array(self.geometry[type_id(part)]["installed_in_base_frame"])
        return poses[base] @ seat

    def spare_slot(self, slot: str) -> tuple[float, float]:
        for s in self.cell["spare_store"]["slots"]:
            if s["id"] == slot:
                return float(s["x"]), float(s["y"])
        raise LayoutError(f"unknown spare slot {slot!r}")

    def defect_xy(self, k: int) -> tuple[float, float]:
        """k-th default spot in the reject zone (a 3 x 2 grid) when a defect entry gives no x, y."""
        z = self.cell["reject_zone"]
        cols, rows = 3, 2
        x = z["x"][0] + (k % cols + 0.5) * (z["x"][1] - z["x"][0]) / cols
        y = z["y"][0] + (k // cols % rows + 0.5) * (z["y"][1] - z["y"][0]) / rows
        return x, y

    def park_tf(self, k: int, tid: str, floor_z_uf0: float) -> np.ndarray:
        x = PARK_ORIGIN_MM[0] + (k % 8) * PARK_PITCH_MM
        y = PARK_ORIGIN_MM[1] + (k // 8) * PARK_PITCH_MM
        return translation(x, y, floor_z_uf0) @ self.rest_tf(tid, "installed")

    def solve(self, layout: dict, pool: list[str], floor_z_uf0: float) -> list[Placement]:
        """Assign every layout entry to a pool instance; everything not named is parked off-table."""
        free = {}
        for uid in pool:
            free.setdefault(type_id(uid), []).append(uid)
        out: dict[str, Placement] = {}

        def take(entry: dict) -> tuple[str, str]:
            tid = type_id(entry["id"])
            if tid not in self.geometry:
                raise LayoutError(f"unknown object id {entry['id']!r}")
            want = entry.get("uid") or (entry["id"] if "#" in entry["id"] else None)
            options = free.get(tid, [])
            if want is not None and want in options:
                uid = want
            elif want is None and options:
                uid = options[0]
            else:
                raise LayoutError(f"no free pool instance for {entry['id']!r} (pool: {free.get(tid, [])})")
            options.remove(uid)
            return uid, tid

        objects = list(layout.get("objects", []))
        seated = [o for o in objects if str(o.get("state", "installed")).startswith("on:")]
        for o in [o for o in objects if o not in seated]:
            uid, tid = take(o)
            state = o.get("state", "installed")
            out[uid] = Placement(uid, tid, self.table_tf(tid, o["x"], o["y"], o.get("yaw", 0.0), state), state)
        for o in seated:
            uid, tid = take(o)
            base = o["state"].split(":", 1)[1]
            if base not in out or not out[base].type_id.startswith("F"):
                raise LayoutError(f"{o['id']} is on {base!r}, which is not a base placed in this layout")
            out[uid] = Placement(uid, tid, self.seat_tf_for(uid, base, {base: out[base].tf_uf0}), o["state"])
        for s in layout.get("spares", []):
            uid, tid = take(s)
            x, y = (s["x"], s["y"]) if "x" in s else self.spare_slot(s["slot"])
            state = s.get("state", "installed")
            out[uid] = Placement(uid, tid, self.table_tf(tid, x, y, s.get("yaw", 0.0), state),
                                 f"spare:{s.get('slot', '')}", role="spare")
        for k, d in enumerate(layout.get("defects", [])):
            uid, tid = take(d)
            x, y = (d["x"], d["y"]) if "x" in d else self.defect_xy(k)
            out[uid] = Placement(uid, tid, self.table_tf(tid, x, y, d.get("yaw", 0.0), d.get("state", "installed")),
                                 "defect", role="defect")
        k = 0
        for uid in pool:
            if uid not in out:
                out[uid] = Placement(uid, type_id(uid), self.park_tf(k, type_id(uid), floor_z_uf0), "parked", "parked")
                k += 1
        return [out[uid] for uid in pool]


def default_pool(geometry: dict, spare_copies: int = 1) -> list[str]:
    """All 30 objects once, plus `spare_copies` extra instances of each of the 24 parts (never a base) for the spare
    store and the defect area (user 2026-10-28: spares are any part, never a base)."""
    pool = sorted(geometry)
    for k in range(2, 2 + spare_copies):
        pool += [f"{tid}#{k}" for tid in sorted(geometry) if tid.startswith("M")]
    return pool
