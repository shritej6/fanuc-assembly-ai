#!/bin/bash
# Validate C/I plus evaluation-only T and create deterministic per-part anomalib crops without reserving a GPU.
#   RUN_ROOT=$HOME/fanuc-t14
#   DATA_ROOT=$HOME/fanuc-assembly-data/synthetic
#   CODE_ROOT=$HOME/fanuc-t14-code-<commit>
#   SOURCE_COMMIT=<commit>
#   export RUN_ROOT DATA_ROOT CODE_ROOT SOURCE_COMMIT
#   sbatch --export=ALL,RUN_ROOT,DATA_ROOT,CODE_ROOT,SOURCE_COMMIT \
#     "$CODE_ROOT/perception/rosie/t14_prepare_patchcore.sbatch"
#SBATCH --job-name=fa-t14-pc-data
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 runtime directory}"
: "${DATA_ROOT:?set DATA_ROOT to the synthetic data directory}"
: "${CODE_ROOT:?set CODE_ROOT to the exact T14 source snapshot}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the tested source revision}"
SINGULARITY=/usr/local/singularity-3.10.0/bin/singularity
SIF=/data/containers/msoe-pytorch-23.05-py3.sif
OUTPUT_ROOT="$RUN_ROOT/datasets/attempts/patchcore-${SLURM_JOB_ID}"
ARTIFACT="$RUN_ROOT/artifacts/patchcore_prepare-${SLURM_JOB_ID}.json"
APPROVED="$RUN_ROOT/datasets/approved/patchcore-${SLURM_JOB_ID}.txt"
mkdir -p "$RUN_ROOT/datasets/attempts" "$RUN_ROOT/datasets/approved" "$RUN_ROOT/artifacts"

test -x "$SINGULARITY"
test -f "$SIF"
test -d "$DATA_ROOT/C"
test -d "$DATA_ROOT/I"
test -d "$DATA_ROOT/T"
test -f "$CODE_ROOT/contracts/taxonomy.json"
test ! -e "$OUTPUT_ROOT"
NODE_NAME=$(hostname)
SIF_SHA256=$(sha256sum "$SIF" | awk '{print $1}')

"$SINGULARITY" exec \
    -B "$CODE_ROOT:$CODE_ROOT:ro" -B "$DATA_ROOT:$DATA_ROOT:ro" -B "$RUN_ROOT" \
    "$SIF" python - \
    "$CODE_ROOT" "$DATA_ROOT" "$OUTPUT_ROOT" "$ARTIFACT" \
    "$SLURM_JOB_ID" "$NODE_NAME" "$SOURCE_COMMIT" "$SIF_SHA256" <<'PY'
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

code_root = Path(sys.argv[1])
data_root = Path(sys.argv[2])
output_root = Path(sys.argv[3])
artifact = Path(sys.argv[4])
sys.path.insert(0, str(code_root))

from perception.anomaly.dataset import (  # noqa: E402
    discover_patchcore_samples,
    prepare_patchcore_folders,
)

taxonomy = json.loads((code_root / "contracts/taxonomy.json").read_text())
expected_object_ids = tuple(
    item["object_id"] for item in taxonomy["objects"]
    if item["object_id"].startswith("M")
)
if len(expected_object_ids) != 24 or len(set(expected_object_ids)) != 24:
    raise RuntimeError("taxonomy must contain exactly 24 unique part IDs")
samples = discover_patchcore_samples(data_root)
summary = prepare_patchcore_folders(
    samples,
    output_root,
    expected_object_ids=expected_object_ids,
    normal_test_modulus=5,
)
manifest = output_root / "manifest.jsonl"
ready = output_root / "_READY.json"
payload = {
    "slurm_job_id": sys.argv[5],
    "hostname": sys.argv[6],
    "source_commit": sys.argv[7],
    "singularity_image": "/data/containers/msoe-pytorch-23.05-py3.sif",
    "singularity_sha256": sys.argv[8],
    "data_root": str(data_root),
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
