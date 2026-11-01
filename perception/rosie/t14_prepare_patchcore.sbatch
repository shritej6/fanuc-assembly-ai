#!/bin/bash
# Derive C/I rest evidence, then prepare O-only training plus installed C/I/T evaluation crops without a GPU.
# Authored by Shritej Gunda.
#   RUN_ROOT=$HOME/fanuc-t14
#   NEW_DATA_ROOT=$HOME/fanuc-testbed-data/synthetic
#   OLD_DATA_ROOT=$HOME/fanuc-assembly-data/synthetic
#   REST_TOOL=$HOME/fanuc-r4-backup/fa-demo/sdg/tools/rest_labels.py
#   REST_GEOMETRY=$HOME/fanuc-r4-backup/fa-demo/sdg/data/object_geometry.json
#   CODE_ROOT=$HOME/fanuc-t14-code-<commit>
#   SOURCE_COMMIT=<commit>
#   export RUN_ROOT NEW_DATA_ROOT OLD_DATA_ROOT REST_TOOL REST_GEOMETRY CODE_ROOT SOURCE_COMMIT
#   sbatch --export=ALL,RUN_ROOT,NEW_DATA_ROOT,OLD_DATA_ROOT,REST_TOOL,REST_GEOMETRY,CODE_ROOT,SOURCE_COMMIT \
#     "$CODE_ROOT/perception/rosie/t14_prepare_patchcore.sbatch"
#SBATCH --job-name=fa-t14-pc-data
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 runtime directory}"
: "${NEW_DATA_ROOT:?set NEW_DATA_ROOT to the O/T synthetic data directory}"
: "${OLD_DATA_ROOT:?set OLD_DATA_ROOT to the immutable C/I synthetic data directory}"
: "${REST_TOOL:?set REST_TOOL to the reviewed rest_labels.py}"
: "${REST_GEOMETRY:?set REST_GEOMETRY to the reviewed object geometry}"
: "${CODE_ROOT:?set CODE_ROOT to the exact T14 source snapshot}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the tested source revision}"
SINGULARITY=/usr/local/singularity-3.10.0/bin/singularity
SIF=/data/containers/msoe-pytorch-23.05-py3.sif
OUTPUT_ROOT="$RUN_ROOT/datasets/attempts/patchcore-${SLURM_JOB_ID}"
REST_DATA_ROOT="$RUN_ROOT/rest/attempts/rest-${SLURM_JOB_ID}"
ARTIFACT="$RUN_ROOT/artifacts/patchcore_prepare-${SLURM_JOB_ID}.json"
APPROVED="$RUN_ROOT/datasets/approved/patchcore-${SLURM_JOB_ID}.txt"
mkdir -p "$RUN_ROOT/datasets/attempts" "$RUN_ROOT/datasets/approved" "$RUN_ROOT/rest/attempts" "$RUN_ROOT/artifacts"

test -x "$SINGULARITY"
test -f "$SIF"
test -d "$NEW_DATA_ROOT/O"
test -d "$NEW_DATA_ROOT/T"
test -d "$OLD_DATA_ROOT/C"
test -d "$OLD_DATA_ROOT/I"
test -f "$REST_TOOL"
test -f "$REST_GEOMETRY"
test -f "$CODE_ROOT/cad/defects/manifest.json"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=all)"
test "$(sha256sum "$REST_TOOL" | awk '{print $1}')" = b89e81c7bcc11f76ec59d88a8127d1997bf5871d849037346d3f0b7084e2d10f
test "$(sha256sum "$REST_GEOMETRY" | awk '{print $1}')" = bf98f0473e9c3e5a5a9fbb1bae801434c5085b094b56d28eb03860386245d91c
test -f "$CODE_ROOT/contracts/taxonomy.json"
test ! -e "$OUTPUT_ROOT"
test ! -e "$REST_DATA_ROOT"
NODE_NAME=$(hostname)
SIF_SHA256=$(sha256sum "$SIF" | awk '{print $1}')

"$SINGULARITY" exec \
    -B "$CODE_ROOT:$CODE_ROOT:ro" -B "$NEW_DATA_ROOT:$NEW_DATA_ROOT:ro" \
    -B "$OLD_DATA_ROOT:$OLD_DATA_ROOT:ro" -B "$REST_TOOL:$REST_TOOL:ro" \
    -B "$REST_GEOMETRY:$REST_GEOMETRY:ro" -B "$RUN_ROOT" \
    "$SIF" python - \
    "$CODE_ROOT" "$NEW_DATA_ROOT" "$OLD_DATA_ROOT" "$REST_TOOL" \
    "$REST_GEOMETRY" "$REST_DATA_ROOT" "$OUTPUT_ROOT" "$ARTIFACT" \
    "$SLURM_JOB_ID" "$NODE_NAME" "$SOURCE_COMMIT" "$SIF_SHA256" <<'PY'
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

code_root = Path(sys.argv[1])
new_data_root = Path(sys.argv[2])
old_data_root = Path(sys.argv[3])
rest_tool = Path(sys.argv[4])
rest_geometry = Path(sys.argv[5])
rest_data_root = Path(sys.argv[6])
output_root = Path(sys.argv[7])
artifact = Path(sys.argv[8])
sys.path.insert(0, str(code_root))

from perception.anomaly.dataset import (  # noqa: E402
    discover_patchcore_samples,
    prepare_patchcore_folders,
)
from perception.anomaly.rest_provenance import (  # noqa: E402
    derive_rest_sidecars,
    verify_rest_sidecars,
)
from perception.anomaly.source_contract import validate_patchcore_source_inventory  # noqa: E402

taxonomy = json.loads((code_root / "contracts/taxonomy.json").read_text())
expected_object_ids = tuple(
    item["object_id"] for item in taxonomy["objects"]
    if item["object_id"].startswith("M")
)
if len(expected_object_ids) != 24 or len(set(expected_object_ids)) != 24:
    raise RuntimeError("taxonomy must contain exactly 24 unique part IDs")
source_inventory = validate_patchcore_source_inventory(new_data_root, old_data_root)
rest_summary = derive_rest_sidecars(
    old_data_root,
    rest_data_root,
    tool_path=rest_tool,
    geometry_path=rest_geometry,
)
samples = discover_patchcore_samples(
    new_data_root,
    old_root=old_data_root,
    rest_root=rest_data_root,
    defect_manifest=code_root / "cad/defects/manifest.json",
)
summary = prepare_patchcore_folders(
    samples,
    output_root,
    expected_object_ids=expected_object_ids,
)
verify_rest_sidecars(rest_data_root, rest_summary)
manifest = output_root / "manifest.jsonl"
ready = output_root / "_READY.json"
payload = {
    "runner_id": sys.argv[9],
    "hostname": sys.argv[10],
    "source_commit": sys.argv[11],
    "singularity_image": "/data/containers/msoe-pytorch-23.05-py3.sif",
    "preparation_runtime_sha256": sys.argv[12],
    "new_data_root": str(new_data_root),
    "old_data_root": str(old_data_root),
    "rest_data_root": str(rest_data_root),
    "rest_tool": str(rest_tool),
    "rest_geometry": str(rest_geometry),
    "rest_derivation": asdict(rest_summary),
    "source_inventory": source_inventory,
    "defect_manifest": str(code_root / "cad/defects/manifest.json"),
    "defect_manifest_sha256": hashlib.sha256(
        (code_root / "cad/defects/manifest.json").read_bytes()
    ).hexdigest(),
    "output_root": str(output_root),
    "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
    "ready_sha256": hashlib.sha256(ready.read_bytes()).hexdigest(),
    "summary": asdict(summary),
}
temporary = artifact.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(artifact)
print(json.dumps(payload, sort_keys=True))
PY

test ! -e "$OUTPUT_ROOT/_INCOMPLETE"
test -s "$OUTPUT_ROOT/_READY.json"
test -s "$OUTPUT_ROOT/manifest.jsonl"
test -s "$ARTIFACT"
printf '%s\n' "$OUTPUT_ROOT" > "$APPROVED"
echo "T14 approved PatchCore dataset: $OUTPUT_ROOT"
