"""Write the seating study jobs, one per mating pair, for measure_seating.py.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
OBJECTS = HERE.parent / "objects" / "objects.yaml"
CONTACT = HERE / "contact.yaml"

CONDITIONS = [  # name, xy offset of the hand (mm), tilt about the hand x axis (deg)
    ("aligned", (0.0, 0.0), 0.0), ("offset_x_0.5", (0.5, 0.0), 0.0), ("offset_x_1.0", (1.0, 0.0), 0.0),
    ("offset_y_1.0", (0.0, 1.0), 0.0), ("tilt_1deg", (0.0, 0.0), 1.0), ("tilt_2deg", (0.0, 0.0), 2.0),
    ("tilt_3deg", (0.0, 0.0), 3.0)]
CAPS_N = [15.0, 5.0, 2.0, 0.5]       # downward force caps, tried from high to low until one fails to seat


def part_record(e: dict) -> dict:
    return {"id": e["id"], "stl": f"cad/stl/{e['name']}.stl", "T_local_from_stl": e["T_local_from_stl"],
            "origin_mm": e["installed_origin_mm"], "mass_g": e["mass_g"], "com_mm": e["com_mm"]}


def jobs() -> dict:
    objs = yaml.safe_load(OBJECTS.read_text())["objects"]
    contact = yaml.safe_load(CONTACT.read_text())
    out = []
    for oid, e in objs.items():
        if e["sheet_order"] == 0:
            continue
        n = e["product"]
        lower = [f"F0{n}"] + [f"M{n}_0{k}" for k in range(1, e["sheet_order"])]
        out.append({"pair": f"{oid}_onto_{lower[-1]}", "moving": part_record(e),
                    "receivers": [part_record(objs[k]) for k in lower],
                    "engagement_mm": e["insertion"]["engagement_mm"],
                    "capture_mm": e["insertion"]["xy_capture_mm"], "yaw_tol_deg": e["insertion"]["yaw_tol_deg"]})
    return {"conditions": [{"name": c, "offset_mm": list(o), "tilt_deg": t} for c, o, t in CONDITIONS],
            "caps_N": CAPS_N, "contact": contact, "jobs": out}


def main() -> None:
    (HERE / "seating_jobs.json").write_text(json.dumps(jobs(), indent=1) + "\n")
    print(f"wrote {len(jobs()['jobs'])} jobs")


if __name__ == "__main__":
    main()
