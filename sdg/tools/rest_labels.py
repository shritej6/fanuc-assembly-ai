"""Tag every loose part in already rendered chunks as `installed` or `fallen`, from its exact pose in the label.

The old sets (A-M) were rendered before the test-bed rule and carry no `rest` field, yet 71 % of their loose parts
lie in a pose the real test-bed never shows except as a fallen part. The rule is the one the new sets render with
(user, 2026-10-28): a loose part is `installed` when its up axis is the up axis it has inside its product (yaw and xy
are free), otherwise `fallen`. Labels are not touched: each chunk gets a sidecar `_REST.json`

    {"rule": ..., "max_tilt_deg": 2.0, "images": {"000234": {"o09": "fallen", ...}, ...}}

listing only loose parts, traps and spares (installed parts, mis-builds on a stack, bases and foreign objects have no
rest state). Pure standard library, so it runs on any data host:

    python3 sdg/tools/rest_labels.py --geometry sdg/data/object_geometry.json --root ~/old-sets/synthetic --sets ABCDFGHIJKLM
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

MAX_TILT_DEG = 2.0
LOOSE_KINDS = ("part", "trap", "spare")
RULE = "installed = same up axis as inside its product (yaw free); anything else = fallen"


def product_up_axes(geometry: dict) -> dict[str, list[float]]:
    """World up expressed in the part frame, for each part as it sits in its product."""
    objs = geometry.get("objects", geometry)
    return {p: o["installed_in_base_frame"][2][:3] for p, o in objs.items() if "installed_in_base_frame" in o}


def tilt_deg(transform: list[list[float]], up: list[float]) -> float:
    dot = sum(a * b for a, b in zip(transform[2][:3], up))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot))))


def rest_states(spec: dict, up_axes: dict[str, list[float]]) -> dict[str, str]:
    out = {}
    for o in spec["objects"]:
        if o["kind"] not in LOOSE_KINDS or o["state"] == "installed" or o.get("build_state"):
            continue
        tilted = tilt_deg(o["transform_uf0"], up_axes[o["object_id"]]) > MAX_TILT_DEG
        out[o["uid"]] = "fallen" if tilted else "installed"
    return out


def label_chunk(chunk: Path, up_axes: dict[str, list[float]]) -> dict[str, int]:
    images, counts = {}, {"installed": 0, "fallen": 0}
    for f in sorted((chunk / "labels").glob("*.json")):
        states = rest_states(json.loads(f.read_text())["spec"], up_axes)
        images[f.stem] = states
        for s in states.values():
            counts[s] += 1
    sidecar = {"rule": RULE, "max_tilt_deg": MAX_TILT_DEG, "images": images}
    tmp = chunk / "_REST.json.tmp"
    tmp.write_text(json.dumps(sidecar, separators=(",", ":")))
    tmp.replace(chunk / "_REST.json")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--geometry", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="the synthetic/ directory holding <SET>/chunk_*")
    ap.add_argument("--sets", required=True, help="set letters, e.g. ABCDFGHIJKLM")
    args = ap.parse_args()
    up_axes = product_up_axes(json.loads(args.geometry.read_text()))
    for s in args.sets:
        total = {"installed": 0, "fallen": 0}
        for chunk in sorted((args.root / s).glob("chunk_*")):
            for k, v in label_chunk(chunk, up_axes).items():
                total[k] += v
        print(s, total, flush=True)


if __name__ == "__main__":
    main()
