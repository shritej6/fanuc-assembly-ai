"""Regenerate every STEP and STL file in one command:  python cad/build_all.py
(Optional model numbers rebuild a subset, e.g. `python cad/build_all.py 1` for the first test print.)

STEP files: each part in its installed orientation, local origin at its lowest seating plane.
STL files: same, except parts with a print_rotation (rear guards) are laid flat for slicing.
Also writes output/part_registry.json (installed offsets, grip zones, assemblies) that
validate_outputs.py uses to re-place the exported files and check the assemblies.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cadquery as cq

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_geometry import BUILD_NOTES, Part  # noqa: E402
from parameters import STL_ANGULAR_TOLERANCE, STL_LINEAR_TOLERANCE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
FIXTURE_DIR = "FIXTURES"
MODELS = ("model_01_rotary", "model_02_indicator", "model_03_pushbutton",
          "model_04_sensor_connector", "model_05_io_connector", "model_06_power_connector")

# Deliberate wrong-part recovery cases: (label, base assembly, wrong part, part it replaces)
WRONG_PART_CASES = (
    ("M2 with M1 rotary knob (wrong part)", "MODEL_02_INDICATOR_LIGHT", "M1_03_Rotary_Knob", "M2_03_Indicator_Cap"),
    ("M6 with M5 I/O insert (wrong part)", "MODEL_06_POWER_CONNECTOR", "M5_02_IO_Insert", "M6_02_Power_Insert"),
)

log = logging.getLogger("fanuc_cad")


def stl_shape(part: Part) -> cq.Workplane:
    """Local shape for the STL; rotated parts are re-seated on Z = 0 and centred in XY."""
    shape = part.local()
    if part.print_rotation is None:
        return shape
    axis, deg = part.print_rotation
    shape = shape.rotate((0, 0, 0), axis, deg)
    bb = shape.val().BoundingBox()
    return shape.translate((-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin))


def export_part(part: Part, folder: str) -> None:
    step_path = OUT / "step" / folder / f"{part.name}.step"
    stl_path = OUT / "stl" / folder / f"{part.name}.stl"
    step_path.parent.mkdir(parents=True, exist_ok=True)
    stl_path.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(part.local(), str(step_path), exportType="STEP")
    cq.exporters.export(stl_shape(part), str(stl_path), exportType="STL",
                        tolerance=STL_LINEAR_TOLERANCE, angularTolerance=STL_ANGULAR_TOLERANCE)


def registry_entry(part: Part, folder: str) -> dict:
    return {"folder": folder, "origin": list(part.origin),
            "grips": [asdict(g) for g in part.grips],
            "print_rotation": None if part.print_rotation is None else
            {"axis": list(part.print_rotation[0]), "deg": part.print_rotation[1]}}


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    selected = [MODELS[int(a) - 1] for a in argv] or list(MODELS)
    registry: dict = {"parts": {}, "assemblies": {}}
    for module_name in selected:
        module = importlib.import_module(module_name)
        t0 = time.time()
        fixture, parts = module.build()
        export_part(fixture, FIXTURE_DIR)
        registry["parts"][fixture.name] = registry_entry(fixture, FIXTURE_DIR)
        for part in parts:
            export_part(part, module.MODEL_DIR)
            registry["parts"][part.name] = registry_entry(part, module.MODEL_DIR)
        registry["assemblies"][module.MODEL_DIR] = [fixture.name] + [p.name for p in parts]
        log.info("%s exported (%d parts + fixture) in %.1f s", module.MODEL_DIR, len(parts), time.time() - t0)

    for label, base, wrong, replaces in WRONG_PART_CASES:
        if base not in registry["assemblies"] or wrong not in registry["parts"]:
            continue
        order = registry["assemblies"][base]
        registry["assemblies"][label] = [wrong if n == replaces else n for n in order]

    registry["build_notes"] = list(BUILD_NOTES)
    (OUT / "part_registry.json").write_text(json.dumps(registry, indent=1))
    for note in BUILD_NOTES:
        log.warning("build note: %s", note)
    log.info("Build complete: %d parts exported to %s", len(registry["parts"]), OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
