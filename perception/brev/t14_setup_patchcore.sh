#!/bin/bash
# Build a hash-linked T14 PatchCore environment on the dedicated Brev L4 VM.
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 Brev runtime directory}"
: "${CODE_ROOT:?set CODE_ROOT to the exact clean T14 Git snapshot}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the tested source revision}"
: "${ATTEMPT_ID:?set ATTEMPT_ID to a unique immutable attempt ID}"
: "${BREV_INSTANCE_ID:?set BREV_INSTANCE_ID from brev ls}"
: "${BREV_MACHINE_TYPE:?set BREV_MACHINE_TYPE from brev ls}"
ENV_DIR="$RUN_ROOT/envs/attempts/patchcore-brev-${ATTEMPT_ID}"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
ARTIFACT="$ARTIFACT_DIR/patchcore_env_brev-${ATTEMPT_ID}.json"
AUDIT="$ARTIFACT_DIR/pip_audit_brev-${ATTEMPT_ID}.json"
PLATFORM="$ARTIFACT_DIR/platform_brev-${ATTEMPT_ID}.json"
APPROVED="$RUN_ROOT/envs/approved/patchcore-brev-${ATTEMPT_ID}.txt"
mkdir -p "$RUN_ROOT/cache/pip" "$RUN_ROOT/envs/attempts" "$RUN_ROOT/envs/approved" "$ARTIFACT_DIR"

[[ "$RUN_ROOT" = /* ]]
[[ "$CODE_ROOT" = /* ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]]
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=all)"
test -z "$(git -C "$CODE_ROOT" clean -ndx)"
test ! -e "$ENV_DIR"
test ! -e "$ARTIFACT"
test ! -e "$AUDIT"
test ! -e "$PLATFORM"
test ! -e "$APPROVED"
test ! -e "$APPROVED.tmp"
command -v nvidia-smi >/dev/null
NODE_NAME=$(hostname)
GPU_ROW=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits)
test "$(printf '%s\n' "$GPU_ROW" | wc -l | tr -d ' ')" = 1

python3 - "$PLATFORM" "$BREV_INSTANCE_ID" "$BREV_MACHINE_TYPE" "$NODE_NAME" "$GPU_ROW" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path

target = Path(sys.argv[1])
os_release = Path("/etc/os-release")
payload = {
    "runtime_kind": "brev-vm",
    "brev_instance_id": sys.argv[2],
    "brev_machine_type": sys.argv[3],
    "hostname": sys.argv[4],
    "gpu_query": sys.argv[5],
    "kernel": platform.release(),
    "machine": platform.machine(),
    "os_release_sha256": hashlib.sha256(os_release.read_bytes()).hexdigest(),
}
temporary = target.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(target)
PY
RUNTIME_SHA256=$(sha256sum "$PLATFORM" | awk '{print $1}')

python3 -m venv "$ENV_DIR"
export PIP_CACHE_DIR="$RUN_ROOT/cache/pip"
export PIP_CONFIG_FILE=/dev/null
unset PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS PIP_NO_INDEX
unset PIP_REQUIREMENT PIP_CONSTRAINT PIP_TRUSTED_HOST
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
"$ENV_DIR/bin/python" -m pip install pip==26.2.2 setuptools==84.2.1 --index-url https://pypi.org/simple
"$ENV_DIR/bin/python" -m pip install torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
"$ENV_DIR/bin/python" -m pip install \
    anomalib==2.7.0 pillow==13.0.0 filelock==4.2.4 pip-audit==2.10.1 \
    --index-url https://pypi.org/simple
"$ENV_DIR/bin/python" -m pip check
"$ENV_DIR/bin/python" -m pip freeze --all > "$ENV_DIR/requirements.freeze.txt"

"$ENV_DIR/bin/python" - "$ENV_DIR/requirements.freeze.txt" "$ENV_DIR/requirements.audit.txt" <<'PY'
import sys
from pathlib import Path

from packaging.version import InvalidVersion, Version

freeze, audit = map(Path, sys.argv[1:])
normalized = []
for line in freeze.read_text().splitlines():
    if line.count("==") != 1:
        raise RuntimeError(f"audit requires one exact version pin per line: {line!r}")
    package, raw_version = line.split("==", 1)
    try:
        normalized.append(f"{package}=={Version(raw_version).public}")
    except InvalidVersion as exc:
        raise RuntimeError(f"invalid frozen version for {package}: {raw_version}") from exc
temporary = audit.with_suffix(".tmp")
temporary.write_text("\n".join(normalized) + "\n")
temporary.replace(audit)
PY
"$ENV_DIR/bin/pip-audit" --strict --no-deps --disable-pip --progress-spinner=off \
    --requirement "$ENV_DIR/requirements.audit.txt" --format=json --output "$AUDIT"

"$ENV_DIR/bin/python" - \
    "$ARTIFACT" "$AUDIT" "$ENV_DIR/requirements.freeze.txt" "$ENV_DIR/requirements.audit.txt" \
    "$ENV_DIR" "$PLATFORM" "$RUNTIME_SHA256" "$SOURCE_COMMIT" <<'PY'
import hashlib
import json
import math
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from anomalib.models import Patchcore

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (8, 9):
    raise RuntimeError(f"expected one Brev L4 capability 8.9, got {torch.cuda.get_device_capability()}")
if torch.cuda.device_count() != 1 or torch.cuda.get_device_name().strip() != "NVIDIA L4":
    raise RuntimeError(f"expected exactly one NVIDIA L4, got {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
if "sm_89" not in torch.cuda.get_arch_list():
    raise RuntimeError(f"Torch wheel lacks sm_89 kernels: {torch.cuda.get_arch_list()}")
probe = torch.ones((1024, 1024), device="cuda")
checksum = float((probe @ probe).sum().item())
torch.cuda.synchronize()
if not math.isfinite(checksum):
    raise RuntimeError("CUDA smoke produced a non-finite result")
artifact, audit, freeze, requirements, env_dir, platform_path = map(Path, sys.argv[1:7])
payload = {
    "anomalib": version("anomalib"),
    "torch": version("torch"),
    "torchvision": version("torchvision"),
    "pillow": version("pillow"),
    "filelock": version("filelock"),
    "pip": version("pip"),
    "setuptools": version("setuptools"),
    "pip_audit": version("pip-audit"),
    "cuda_runtime": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "architectures": torch.cuda.get_arch_list(),
    "cuda_probe_checksum": checksum,
    "model": type(Patchcore()).__name__,
    "env_dir": str(env_dir.resolve()),
    "platform_identity": str(platform_path.resolve()),
    "platform_identity_sha256": hashlib.sha256(platform_path.read_bytes()).hexdigest(),
    "runtime_sha256": sys.argv[7],
    "source_commit": sys.argv[8],
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "audit_requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    "pip_audit_sha256": hashlib.sha256(audit.read_bytes()).hexdigest(),
}
temporary = artifact.with_suffix(".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
temporary.replace(artifact)
print(json.dumps(payload, sort_keys=True, allow_nan=False))
PY

test -s "$ARTIFACT"
test -s "$AUDIT"
test -s "$PLATFORM"
sha256sum "$ARTIFACT" | awk '{print $1}' > "$ENV_DIR/.complete.tmp"
mv "$ENV_DIR/.complete.tmp" "$ENV_DIR/.complete"
printf '%s\n' "$ARTIFACT" > "$APPROVED.tmp"
mv "$APPROVED.tmp" "$APPROVED"
echo "T14 Brev runtime SHA-256: $RUNTIME_SHA256"
echo "T14 approved Brev PatchCore environment: $ENV_DIR"
