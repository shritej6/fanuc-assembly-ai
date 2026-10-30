"""Grasp and insert frames for all 30 printed objects, from the product CAD that was printed.

    uv run python cad/grasp_frames.py          # writes cad/grasp_frames.json (same CAD -> same file)

Source: each model's `build()` (cad/products), the same code that wrote the printed STLs (cad/stl is byte-identical
to the printed FANUC_PRINT_FIRST/output/stl, checked 2026-10-29). Per object:
- `installed_origin_mm`: the part's local origin (centre of its lowest seating plane) in its product's assembly frame,
  whose origin is the base's local origin. A part is installed when its origin sits at the base pose + this offset,
  at the base's yaw (parts install without extra rotation in this frame).
- `T_local_from_stl`: STL mesh frame -> the part's installed local frame. Identity except the rear guards, whose STL
  is laid flat for printing (print_rotation, then re-centred in x, y and seated on z = 0). FoundationPose estimates
  the STL frame's pose, so a measured pose must be multiplied by the inverse of this before using the offsets.
- `grips`: the CAD grip pads, in the installed local frame: closing axis, pad width, span, contact height, the pad
  point and the TCP to command. Both come from the Team V printed fingers' validated grasp of that part
  (cad/fingers/product_grasp_frames.json, 24/24 PASS in that project's CAD check):
  - `pad_point_local_mm`: the fingers' contact centre moved down by the 1.85 mm from contact centre to fingertip.
    It must lie on the CAD pad, or nothing is written. It is NOT the point to command.
  - `tcp_local_mm`: {wrist turn: TCP}. The TCP (UTOOL, on the flange axis at the tip plane) is 1.8 mm from the pad
    point along the gripper STEP X axis, which is (cos r, sin r) in UF0 at w, p = -180, 0, so the two turns r and
    r + 180 put the TCP on opposite sides of the pad. Key k is the turn relative to the part: r = measured yaw + k.
    Every Team V validated grasp is checked to satisfy this, or nothing is written.
- `roll_offset_deg`: the turn that spreads the jaws along the closing axis (the other one is roll_offset + 180).
  The twin's IK spreads the jaws along rot(w, p, r) @ +y (sim/starter/ik.py, scene_config.FINGER_SPREAD_AXIS = +y):
  at w = -180, p = 0 that is (sin r, -cos r, 0) in UF0, so jaws closing along the part's local X need r = yaw + 90
  and along local Y need r = yaw.
Every pad must fit the Team V fingers on the SMC MHZL2-16D2 (rubber fingertip gap 4.5 mm closed, 16.5 mm open,
cad/fingers/README.md): wider than the closed gap and narrower than the open gap, or nothing is written.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "cad" / "grasp_frames.json"
sys.path.insert(0, str(ROOT / "cad" / "products"))
MODULES = ("model_01_rotary", "model_02_indicator", "model_03_pushbutton",
           "model_04_sensor_connector", "model_05_io_connector", "model_06_power_connector")
FINGERS = ROOT / "cad" / "fingers" / "product_grasp_frames.json"
CLOSED_GAP_MAX_MM = 4.5     # Team V fingers, rubber pads closed (cad/fingers/README.md)
OPEN_GAP_MIN_MM = 16.5      # Team V fingers, rubber pads open (same source)
TIP_BELOW_CONTACT_MM = 86.10 - 84.25   # tip plane minus contact centre, gripper STEP frame (same source)
CONTACT_ON_PAD_MM = 0.5     # the fingers' contact centre may sit this far from the CAD pad centre across the pad
TCP_FROM_CONTACT_MM = 1.8   # flange axis minus contact centre along gripper STEP X (contact at STEP X = -1.8)
TCP_CHECK_TOL_MM = 0.01     # every validated Team V grasp must match that offset this closely
ROLL_OFFSET_DEG = {"X": 90.0, "Y": 0.0}


class GraspFrameError(ValueError):
    """A part's grip pad does not fit the gripper or the CAD is not what this script expects."""


def object_id(name: str) -> str:
    """'M1_01_Housing' -> 'M1_01'; 'F01_Rotary_Fixture' -> 'F01'."""
    head = name.split("_")
    return "_".join(head[:2]) if name.startswith("M") else head[0]


def _axis_rotation(axis: tuple[float, float, float], deg: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    rot = np.eye(4)
    rot[:3, :3] = c * np.eye(3) + s * k + (1 - c) * np.outer(a, a)
    return rot


def local_from_stl(part) -> np.ndarray:
    """Inverse of build_all.stl_shape: rotate by print_rotation, then re-centre x, y and seat on z = 0."""
    if part.print_rotation is None:
        return np.eye(4)
    axis, deg = part.print_rotation
    bb = part.local().rotate((0, 0, 0), axis, deg).val().BoundingBox()
    shift = np.eye(4)
    shift[:3, 3] = (-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin)
    return np.linalg.inv(shift @ _axis_rotation(axis, deg))


def _contact(fingers: dict, name: str) -> np.ndarray:
    if name not in fingers:
        raise GraspFrameError(f"{name}: no validated grasp in {FINGERS.name}")
    contact = np.array([float(v) for v in fingers[name]["contact_centre_world"].strip("()").split(",")])
    frame = fingers[name]
    step_x = np.asarray(frame["rotation_gripper_to_world"], dtype=float)[:, 0]
    gap = np.asarray(frame["translation_mm"], dtype=float)[:2] - (contact[:2] + TCP_FROM_CONTACT_MM * step_x[:2])
    if np.abs(gap).max() > TCP_CHECK_TOL_MM:
        raise GraspFrameError(f"{name}: validated gripper origin is not {TCP_FROM_CONTACT_MM} mm from the contact "
                              f"along STEP X (off by {gap.round(4).tolist()} mm)")
    return contact


def tcp_by_turn(pad_point: np.ndarray, roll_offset: float) -> dict[str, list[float]]:
    """TCP in the part's local frame for each wrist turn k (r = yaw + k): the pad point + 1.8 mm * (cos k, sin k)."""
    out = {}
    for turn in (roll_offset, roll_offset + 180.0):
        k = math.remainder(turn, 360.0) + 0.0     # (-180, 180], with +0.0 instead of -0.0
        k = 180.0 if k == -180.0 else k
        shift = TCP_FROM_CONTACT_MM * np.array([math.cos(math.radians(k)), math.sin(math.radians(k)), 0.0])
        out[f"{k:g}"] = [round(float(v), 4) + 0.0 for v in pad_point + shift]
    return out


def grip_entry(name: str, grip, origin: tuple[float, float, float], contact: np.ndarray) -> dict:
    """One CAD pad with the fingers' validated contact on it; contact is in the assembly frame."""
    width = 2 * grip.half_width
    if not CLOSED_GAP_MAX_MM < width < OPEN_GAP_MIN_MM:
        raise GraspFrameError(f"{name}: grip pad {width} mm does not fit the {CLOSED_GAP_MAX_MM}-{OPEN_GAP_MIN_MM} mm gap")
    across, along = (contact[0], contact[1]) if grip.axis == "X" else (contact[1], contact[0])
    on_pad = (abs(across - grip.center) <= CONTACT_ON_PAD_MM and grip.span[0] <= along <= grip.span[1]
              and grip.z_top - grip.height <= contact[2] <= grip.z_top)
    if not on_pad:
        raise GraspFrameError(f"{name}: the fingers' contact {contact.tolist()} is not on the CAD grip pad")
    pad_point = contact - np.asarray(origin) - (0.0, 0.0, TIP_BELOW_CONTACT_MM)
    return {"closing_axis": grip.axis, "pad_width_mm": round(width, 4),
            "pad_top_z_mm": round(grip.z_top - origin[2], 4), "contact_height_mm": round(grip.height, 4),
            "pad_point_local_mm": [round(float(v), 4) for v in pad_point],
            "tcp_local_mm": tcp_by_turn(pad_point, ROLL_OFFSET_DEG[grip.axis]),
            "roll_offset_deg": ROLL_OFFSET_DEG[grip.axis]}


def part_entry(part, base: str, fingers: dict) -> dict:
    grips = [grip_entry(part.name, g, part.origin, _contact(fingers, part.name)) for g in part.grips]
    return {"name": part.name, "base": base, "installed_origin_mm": [round(v, 4) for v in part.origin],
            "T_local_from_stl": np.round(local_from_stl(part), 6).tolist(), "grips": grips}


def frames() -> dict:
    fingers = json.loads(FINGERS.read_text())
    objects = {}
    for module_name in MODULES:
        fixture, parts = importlib.import_module(module_name).build()
        base = object_id(fixture.name)
        objects[base] = part_entry(fixture, base, fingers)
        for part in parts:
            if len(part.grips) != 1:
                raise GraspFrameError(f"{part.name}: expected one grip pad in the CAD, found {len(part.grips)}")
            objects[object_id(part.name)] = part_entry(part, base, fingers)
    return {"source": "cad/products build() (printed CAD) + cad/fingers (Team V fingers), cad/grasp_frames.py",
            "units": "mm, deg",
            "tcp": "UTOOL on the flange axis at the tip plane, w, p = -180, 0; r = measured part yaw + k, "
                   "TCP = installed pose applied to grips[].tcp_local_mm[k] (k = roll_offset_deg or +180)",
            "gripper_gap_mm": {"closed_max": CLOSED_GAP_MAX_MM, "open_min": OPEN_GAP_MIN_MM},
            "objects": dict(sorted(objects.items()))}


def main() -> None:
    OUT.write_text(json.dumps(frames(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
