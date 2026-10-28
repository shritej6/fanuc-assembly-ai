#!/bin/bash
# Build and prove the pinned PatchCore environment on one Rosie V100.
# Submit from a persistent log directory:
#   RUN_ROOT=$HOME/fanuc-t14 SOURCE_COMMIT=$(git rev-parse HEAD) \
#     sbatch --export=ALL,RUN_ROOT,SOURCE_COMMIT perception/rosie/t14_patchcore_env_smoke.sbatch
#SBATCH --job-name=fa-t14-pc-env
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.log
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 runtime directory}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the tested source revision}"
SINGULARITY=/usr/local/singularity-3.10.0/bin/singularity
SIF=/data/containers/msoe-pytorch-23.05-py3.sif
ENV_DIR="$RUN_ROOT/envs/attempts/patchcore-${SLURM_JOB_ID}"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
ARTIFACT="$ARTIFACT_DIR/patchcore_env_smoke-${SLURM_JOB_ID}.json"
AUDIT="$ARTIFACT_DIR/pip_audit-${SLURM_JOB_ID}.json"
APPROVED="$RUN_ROOT/envs/approved/patchcore-${SLURM_JOB_ID}.txt"
mkdir -p "$RUN_ROOT/cache/pip" "$RUN_ROOT/envs/attempts" "$RUN_ROOT/envs/approved" "$ARTIFACT_DIR"

echo "node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
test -x "$SINGULARITY"
test -f "$SIF"
SIF_SHA256=$(sha256sum "$SIF" | awk '{print $1}')
NODE_NAME=$(hostname)
test ! -e "$ENV_DIR"

"$SINGULARITY" exec --nv -B "$RUN_ROOT" "$SIF" bash -s -- \
    "$ENV_DIR" "$ARTIFACT" "$AUDIT" "$RUN_ROOT/cache/pip" \
    "$SLURM_JOB_ID" "$NODE_NAME" "$SOURCE_COMMIT" "$SIF" "$SIF_SHA256" <<'CONTAINER'
set -euo pipefail
ENV_DIR=$1
ARTIFACT=$2
AUDIT=$3
export PIP_CACHE_DIR=$4
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX
unset PIP_REQUIREMENT PIP_CONSTRAINT PIP_TRUSTED_HOST
JOB_ID=$5
NODE_NAME=$6
SOURCE_COMMIT=$7
SIF=$8
SIF_SHA256=$9

python -m venv "$ENV_DIR"
"$ENV_DIR/bin/python" -m pip install \
    torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu126
"$ENV_DIR/bin/python" -m pip install \
    anomalib==2.7.0 pip-audit==2.10.1 \
    --index-url https://pypi.org/simple
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze > "$ENV_DIR/requirements.freeze.txt"
"$ENV_DIR/bin/pip-audit" --strict --format=json --output "$AUDIT"
"$ENV_DIR/bin/python" - \
    "$ARTIFACT" "$AUDIT" "$ENV_DIR/requirements.freeze.txt" \
    "$JOB_ID" "$NODE_NAME" "$SOURCE_COMMIT" "$SIF" "$SIF_SHA256" <<'PY'
import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from anomalib.models import Patchcore

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable inside the T14 PatchCore environment")
if torch.cuda.get_device_capability() != (7, 0):
    raise RuntimeError(f"expected a V100 capability, got {torch.cuda.get_device_capability()}")
if "sm_70" not in torch.cuda.get_arch_list():
    raise RuntimeError(f"Torch wheel lacks sm_70 kernels: {torch.cuda.get_arch_list()}")

probe = torch.ones((1024, 1024), device="cuda")
checksum = float((probe @ probe).sum().item())
torch.cuda.synchronize()
if not math.isfinite(checksum):
    raise RuntimeError("CUDA smoke produced a non-finite result")
model = Patchcore()
artifact = Path(sys.argv[1])
audit = Path(sys.argv[2])
freeze = Path(sys.argv[3])
payload = {
    "anomalib": version("anomalib"),
    "torch": torch.__version__,
    "torchvision": version("torchvision"),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "architectures": torch.cuda.get_arch_list(),
    "cuda_probe_checksum": checksum,
    "model": type(model).__name__,
    "slurm_job_id": sys.argv[4],
    "hostname": sys.argv[5],
    "source_commit": sys.argv[6],
    "singularity_image": sys.argv[7],
    "singularity_sha256": sys.argv[8],
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "pip_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
}
temporary = artifact.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(artifact)
print(json.dumps(payload, sort_keys=True))
PY
CONTAINER

test -s "$ARTIFACT"
test -s "$AUDIT"
touch "$ENV_DIR/.complete"
printf '%s\n' "$ENV_DIR" > "$APPROVED"
echo "T14 PatchCore environment smoke artifact: $ARTIFACT"
echo "T14 approved PatchCore environment: $ENV_DIR"
