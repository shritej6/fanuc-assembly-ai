"""CAD facts for the evaluator: products, seated poses, grip TCPs and seat tolerances.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from ..skills import gates as G

REPO = Path(__file__).resolve().parents[3]
PRODUCTS = ("M1", "M2", "M3", "M4", "M5", "M6")
ROLES = ("01", "02", "03", "04")  # assembly sheet order
# Precision gates: one source of numbers (skills/gates.py), so demos and eval use the same rule.
# Yaw is compared modulo the part's CAD symmetry here.
SEAT_XY_MM, SEAT_YAW_DEG, SEAT_Z_MM, SEAT_TILT_DEG = G.SEAT_XY_MM, G.SEAT_YAW_DEG, G.SEAT_Z_MM, G.SEAT_TILT_DEG
DISTURB_MM, DISTURB_DEG = G.OTHER_MOVE_MM, G.OTHER_MOVE_DEG
SPOT_XY_MM, SPOT_YAW_DEG, SPOT_TILT_DEG = G.SPOT_XY_MM, G.SPOT_YAW_DEG, G.SPOT_TILT_DEG
TOUCH_N = G.FINGER_TOUCH_N
# Grip on the pad: the stricter of skills/gates.py (1.0 mm / 1 deg) and mimic/gates.py (0.5 mm /
# 1 deg), so a policy that passes eval also passes both demo filters.
GRIP_XYZ_MM, GRIP_ROLL_DEG = min(G.GRIP_PAD_MM, 0.5), G.GRIP_ROLL_DEG


@lru_cache(maxsize=1)
def geometry() -> dict:
    """Return the object geometry table."""
    return json.loads((REPO / "sdg/data/object_geometry.json").read_text())["objects"]


@lru_cache(maxsize=1)
def grasp_frames() -> dict:
    """Return the CAD grasp frames per object."""
    return json.loads((REPO / "cad/grasp_frames.json").read_text())["objects"]


@lru_cache(maxsize=1)
def tolerances() -> dict:
    """Return the insertion tolerances per part."""
    return json.loads((REPO / "sdg/data/insertion_tolerance.json").read_text())["parts"]


@lru_cache(maxsize=1)
def objects_yaml() -> dict:
    """Return the objects from objects.yaml."""
    return yaml.safe_load((REPO / "sim/assembly_lab/objects/objects.yaml").read_text())["objects"]


@lru_cache(maxsize=1)
def cell_layout() -> dict:
    """Return the cell layout config."""
    return yaml.safe_load((REPO / "sdg/configs/cell_layout.yaml").read_text())


def type_id(uid: str) -> str:
    """'M4_02#2' -> 'M4_02'; 'F04' -> 'F04'."""
    return uid.split("#")[0]


def product_parts(product: str) -> list[str]:
    """Return the part ids of a product."""
    return [f"{product}_{r}" for r in ROLES]


def base_type(product: str) -> str:
    """The base a product is built on (cad/grasp_frames.json 'base' of its first part), e.g. M4 -> F04."""
    return grasp_frames()[f"{product}_01"]["base"]


def base_group(base_id: str) -> tuple[str, ...]:
    """Identical bases are interchangeable (PLAN section 3.2): A = F01/F02, B = F03, C = F04/F05/F06."""
    for group in (("F01", "F02"), ("F03",), ("F04", "F05", "F06")):
        if type_id(base_id) in group:
            return group
    raise KeyError(base_id)


# ------------------------------------------------------------------ poses
def pose_to_matrix(pose) -> np.ndarray:
    """Return the 4x4 matrix of a W/P/R pose in mm."""
    x, y, z, w, p, r = (float(v) for v in pose)
    cw, sw, cp, sp = math.cos(math.radians(w)), math.sin(math.radians(w)), math.cos(math.radians(p)), math.sin(
        math.radians(p))
    cr, sr = math.cos(math.radians(r)), math.sin(math.radians(r))
    rot = (np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
           @ np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]]))
    t = np.eye(4)
    t[:3, :3], t[:3, 3] = rot, (x, y, z)
    return t


def matrix_to_pose(t: np.ndarray) -> np.ndarray:
    """Return the W/P/R pose of a 4x4 matrix."""
    rot = t[:3, :3]
    p = -math.asin(max(-1.0, min(1.0, rot[2, 0])))
    w, r = math.atan2(rot[2, 1], rot[2, 2]), math.atan2(rot[1, 0], rot[0, 0])
    return np.array([*t[:3, 3], math.degrees(w), math.degrees(p), math.degrees(r)])


def wrap180(a: float) -> float:
    """Wrap an angle in deg to [-180, 180)."""
    return (a + 180.0) % 360.0 - 180.0


def seated_pose(part_id: str, base_id: str, base_pose) -> np.ndarray:
    """Part STL-frame pose when fully seated on base_id at base_pose (the env/layout.py `on:<base>` transform)."""
    seat = np.array(geometry()[type_id(part_id)]["installed_in_base_frame"])
    return matrix_to_pose(pose_to_matrix(base_pose) @ seat)  # base STL z = 0 is the seating face


def grip_tcp(part_id: str, object_pose, grip: int = 0, current_r: float = 0.0) -> np.ndarray:
    """Return the TCP that puts the pads on the CAD grip pad for an object pose."""
    g = grasp_frames()[type_id(part_id)]
    pad = g["grips"][grip]
    local_tcp = pose_to_matrix([*pad["pad_point_local_mm"], 180.0, 0.0, pad["roll_offset_deg"]])
    t = pose_to_matrix(object_pose) @ np.linalg.inv(np.array(g["T_local_from_stl"])) @ local_tcp
    pose = matrix_to_pose(t)
    if abs(abs(pose[3]) - 180.0) < 1e-6:
        pose[3] = -180.0  # w = +-180 are the same pose; the API convention is -180
    options = [pose[5] + k for k in (-360.0, -180.0, 0.0, 180.0, 360.0) if abs(pose[5] + k) <= 360.0]
    pose[5] = min(options, key=lambda r: abs(r - current_r))
    # pad_point_local_mm is the finger pad point; the commanded TCP (flange axis) sits 1.8 mm off it, per roll
    # (objects/planner.py tcp_for_contact)
    from ..objects.planner import tcp_for_contact

    pose[:3] = tcp_for_contact(pose[:3], float(pose[5]))
    return pose


def carry_tcp(start_obj, grasp_tcp, goal_obj) -> np.ndarray:
    """Return the TCP that puts the held object at goal_obj with its in-hand transform kept."""
    t_obj_tcp = np.linalg.inv(pose_to_matrix(start_obj)) @ pose_to_matrix(grasp_tcp)
    pose = matrix_to_pose(pose_to_matrix(goal_obj) @ t_obj_tcp)
    if abs(abs(pose[3]) - 180.0) < 1e-6:
        pose[3] = -180.0
    options = [pose[5] + k for k in (-720.0, -360.0, 0.0, 360.0, 720.0) if abs(pose[5] + k) <= 360.0]
    pose[5] = min(options, key=lambda r: abs(r - grasp_tcp[5]))
    return pose


def world_delta(a, b) -> tuple[float, float]:
    """Return the tilt and yaw in deg of pose a relative to pose b, in the world frame."""
    d = pose_to_matrix(a)[:3, :3] @ pose_to_matrix(b)[:3, :3].T
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(d[2, 2])))))
    return tilt, math.degrees(math.atan2(d[1, 0], d[0, 0]))


def _turned_about_part_axis(part_id: str, seated, angle_deg: float) -> np.ndarray:
    """Return the seated pose turned by angle about the part's vertical axis."""
    ts = pose_to_matrix(seated)
    p = np.linalg.inv(np.array(grasp_frames()[type_id(part_id)]["T_local_from_stl"]))[:3, 3]
    axis = ts[:3, :3].T @ np.array([0.0, 0.0, 1.0])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    th = math.radians(angle_deg)
    rot = np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * k @ k
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = rot, p - rot @ p
    return matrix_to_pose(ts @ m)


def _seat_error(actual, seated) -> dict:
    tilt, yaw = world_delta(actual, seated)
    return {"dz": float(actual[2] - seated[2]), "xy": float(np.hypot(*(actual[:2] - seated[:2]))), "tilt": tilt,
            "yaw": abs(yaw)}


def seat_check(part_id: str, actual, seated) -> tuple[bool, dict]:
    """Precision gates against the seated pose and its symmetric twins (sdg/data/insertion_tolerance.json)."""
    a, s = np.asarray(actual, float), np.asarray(seated, float)
    tol = tolerances().get(type_id(part_id))
    sym = tol["symmetry"] if tol else "every 360 deg"
    if sym == "any":
        angles = [world_delta(a, s)[1]]
    else:
        step = {"every 360 deg": 360.0, "every 180 deg": 180.0}[sym]
        angles = [k * step for k in range(round(360.0 / step))]
    errs = [_seat_error(a, _turned_about_part_axis(part_id, s, th)) for th in angles]
    err = min(errs, key=lambda e: (e["xy"], e["yaw"]))
    ok = abs(err["dz"]) <= SEAT_Z_MM and err["xy"] <= SEAT_XY_MM and err["tilt"] <= SEAT_TILT_DEG \
        and err["yaw"] <= SEAT_YAW_DEG
    return ok, err


def in_zone(pose, zone: dict, margin_mm: float = 0.0) -> bool:
    """Return True if a pose lies inside the zone."""
    return (zone["x"][0] - margin_mm <= pose[0] <= zone["x"][1] + margin_mm
            and zone["y"][0] - margin_mm <= pose[1] <= zone["y"][1] + margin_mm)


def tilt_between(a, b) -> float:
    """How far pose a is tipped relative to pose b (deg), yaw ignored."""
    return world_delta(a, b)[0]


@dataclass(frozen=True)
class Delta:
    """Distance in mm and angle in deg between two poses."""
    mm: float
    deg: float


def pose_delta(a, b) -> Delta:
    """Return the distance and angle between two poses."""
    ta, tb = pose_to_matrix(a), pose_to_matrix(b)
    rel = ta[:3, :3].T @ tb[:3, :3]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0))))
    return Delta(float(np.linalg.norm(ta[:3, 3] - tb[:3, 3])), ang)


def table_pose(part_id: str, x: float, y: float, yaw: float) -> np.ndarray:
    """STL-frame pose of a part standing in its installed orientation on the table (env/layout.py table_tf)."""
    rest = np.array(geometry()[type_id(part_id)]["installed_rest"]["transform"])
    t = pose_to_matrix([x, y, float(cell_layout()["table_z"]), 0.0, 0.0, yaw]) @ rest
    return matrix_to_pose(t)


GRASP_N = 0.5  # sim/assembly_lab/env/task.py GRASP_N: both jaws push the object with at least this, gripper closed


def held_uid(obs: dict) -> str | None:
    """The object in the fingers: sim.gripper_obj if the env gives it, else the env's own grasp rule on sim.jaw_n."""
    sim = obs["sim"]
    if "gripper_obj" in sim:
        return sim["gripper_obj"]
    if float(obs["joints"][6]) < 0.5:
        return None
    held = {u: min(f) for u, f in (sim.get("jaw_n") or {}).items() if min(f) >= GRASP_N}
    return max(held, key=held.get) if held else None
