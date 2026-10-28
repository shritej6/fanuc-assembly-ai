"""Independent STEP re-open check using FreeCAD's own kernel. Run by validate_outputs.py through:

    freecadcmd -c "exec(open('<this file>').read())"

Reads the STEP list from the JSON file named in env FC_CHECK_IN and writes results to FC_CHECK_OUT.
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
    except Exception as exc:  # report per-file failure instead of aborting the whole check
        results[path] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

with open(os.environ["FC_CHECK_OUT"], "w") as fh:
    json.dump(results, fh)
