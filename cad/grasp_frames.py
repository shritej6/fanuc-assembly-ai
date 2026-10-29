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
- `grips`: the CAD grip pads, in the installed local frame: closing axis, pad centre, contact faces, contact height,
  and the fingertip point (TCP) to command: pad centre, `TCP_ABOVE_ROOT_MM` above the pad's usable bottom.
- `roll_offset_deg`: add the part's measured yaw to get the TCP r (`w, p = -180, 0`). The twin's IK spreads the jaws
  along rot(w, p, r) @ +y (sim/starter/ik.py, scene_config.FINGER_SPREAD_AXIS = +y): at w = -180, p = 0 that is
  (sin r, -cos r, 0) in UF0, so jaws closing along the part's local X need r = yaw + 90 and along local Y need
  r = yaw (both +-180 are the same grasp; pick the one nearest the current r).
Every pad must fit the real gripper (SMC MHZL2-16D2 fingertip gap 4 +- 1 mm closed, 15 +- 2 mm open, measured on
the cell v0.5, sim/starter/scene_config.py): the pad must be wider than the largest closed gap and narrower than the
smallest open gap, or nothing is written.
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
CLOSED_GAP_MAX_MM = 5.0     # 4 +- 1 mm closed fingertip gap (sim/starter/scene_config.py, v0.5 cell report)
OPEN_GAP_MIN_MM = 13.0      # 15 +- 2 mm open fingertip gap (same source)
TCP_ABOVE_ROOT_MM = 1.0     # fingertips stop this far above the pad's usable bottom (the root fillet is below it)
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


def grip_entry(name: str, grip, origin: tuple[float, float, float]) -> dict:
    width = 2 * grip.half_width
    if not CLOSED_GAP_MAX_MM < width < OPEN_GAP_MIN_MM:
        raise GraspFrameError(f"{name}: grip pad {width} mm does not fit the {CLOSED_GAP_MAX_MM}-{OPEN_GAP_MIN_MM} mm gap")
    ox, oy, oz = origin
    across, along = (ox, oy) if grip.axis == "X" else (oy, ox)
    centre_across = grip.center - across
    centre_along = (grip.span[0] + grip.span[1]) / 2 - along
    tcp_z = grip.z_top - oz - grip.height + TCP_ABOVE_ROOT_MM
    tcp = (centre_across, centre_along) if grip.axis == "X" else (centre_along, centre_across)
    return {"closing_axis": grip.axis, "pad_width_mm": round(width, 4),
            "pad_span_mm": [round(s - along, 4) for s in grip.span],
            "pad_top_z_mm": round(grip.z_top - oz, 4), "contact_height_mm": round(grip.height, 4),
            "tcp_local_mm": [round(tcp[0], 4), round(tcp[1], 4), round(tcp_z, 4)],
            "roll_offset_deg": ROLL_OFFSET_DEG[grip.axis]}


def part_entry(part, base: str) -> dict:
    return {"name": part.name, "base": base, "installed_origin_mm": [round(v, 4) for v in part.origin],
            "T_local_from_stl": np.round(local_from_stl(part), 6).tolist(),
            "grips": [grip_entry(part.name, g, part.origin) for g in part.grips]}


def frames() -> dict:
    objects = {}
    for module_name in MODULES:
        fixture, parts = importlib.import_module(module_name).build()
        base = object_id(fixture.name)
        objects[base] = part_entry(fixture, base)
        for part in parts:
            if not part.grips:
                raise GraspFrameError(f"{part.name}: no grip pad in the CAD")
            objects[object_id(part.name)] = part_entry(part, base)
    return {"source": "cad/products build() (printed CAD), cad/grasp_frames.py", "units": "mm, deg",
            "tcp": "fingertip point, w, p = -180, 0; r = measured part yaw + roll_offset_deg",
            "gripper_gap_mm": {"closed_max": CLOSED_GAP_MAX_MM, "open_min": OPEN_GAP_MIN_MM},
            "objects": dict(sorted(objects.items()))}


def main() -> None:
    OUT.write_text(json.dumps(frames(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
