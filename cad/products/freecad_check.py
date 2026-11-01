"""Re-open STEP files with the FreeCAD kernel as an independent check.
Authored by Shritej Gunda.
"""

import json
import os

import FreeCAD  # noqa: F401  (initialises the FreeCAD application)
import Part

with open(os.environ["FC_CHECK_IN"]) as fh:
    step_paths = json.load(fh)

results = {}
for path in step_paths:
    try:
        shape = Part.read(path)
        results[path] = {"ok": True, "valid": bool(shape.isValid()), "solids": len(shape.Solids),
                         "volume": float(shape.Volume)}
    except Exception as exc:  # noqa: BLE001 - record each file's error for the report; FreeCAD's error classes vary by build
        results[path] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

with open(os.environ["FC_CHECK_OUT"], "w") as fh:
    json.dump(results, fh)
