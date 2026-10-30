#!/bin/bash
# Build a hash-linked T14 PatchCore environment on the dedicated Brev L4 VM.
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 Brev runtime directory}"
: "${CODE_ROOT:?set CODE_ROOT to the exact clean T14 Git snapshot}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the tested source revision}"
: "${ATTEMPT_ID:?set ATTEMPT_ID to a unique immutable attempt ID}"
: "${BREV_INSTANCE_ID:?set BREV_INSTANCE_ID from brev ls}"
: "${BREV_MACHINE_TYPE:?set BREV_MACHINE_TYPE from brev ls}"
UV_VERSION=0.13.6
PYTHON_VERSION=3.11.18
ENV_DIR="$RUN_ROOT/envs/attempts/patchcore-brev-${ATTEMPT_ID}"
BOOTSTRAP_DIR="$RUN_ROOT/bootstrap/attempts/uv-${ATTEMPT_ID}"
PYTHON_INSTALL_DIR="$RUN_ROOT/python/attempts/cpython-${PYTHON_VERSION}-${ATTEMPT_ID}"
ARTIFACT_DIR="$RUN_ROOT/artifacts"
ARTIFACT="$ARTIFACT_DIR/patchcore_env_brev-${ATTEMPT_ID}.json"
AUDIT="$ARTIFACT_DIR/pip_audit_brev-${ATTEMPT_ID}.json"
BOOTSTRAP_AUDIT="$ARTIFACT_DIR/pip_audit_bootstrap_brev-${ATTEMPT_ID}.json"
PLATFORM="$ARTIFACT_DIR/platform_brev-${ATTEMPT_ID}.json"
APPROVED="$RUN_ROOT/envs/approved/patchcore-brev-${ATTEMPT_ID}.txt"
mkdir -p "$RUN_ROOT/cache/pip/${ATTEMPT_ID}" "$RUN_ROOT/cache/uv/${ATTEMPT_ID}"
mkdir -p "$RUN_ROOT/bootstrap/attempts" "$RUN_ROOT/python/attempts"
mkdir -p "$RUN_ROOT/envs/attempts" "$RUN_ROOT/envs/approved" "$ARTIFACT_DIR"

[[ "$RUN_ROOT" = /* ]]
[[ "$CODE_ROOT" = /* ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]]
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=all)"
test -z "$(git -C "$CODE_ROOT" clean -ndx)"
test ! -e "$ENV_DIR"
test ! -e "$BOOTSTRAP_DIR"
test ! -e "$PYTHON_INSTALL_DIR"
test ! -e "$ARTIFACT"
test ! -e "$AUDIT"
test ! -e "$BOOTSTRAP_AUDIT"
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

while IFS='=' read -r name _; do
    case "$name" in
        PIP_*|UV_*) unset "$name" ;;
    esac
done < <(env)
export PIP_CACHE_DIR="$RUN_ROOT/cache/pip/${ATTEMPT_ID}"
export PIP_CONFIG_FILE=/dev/null
export UV_CACHE_DIR="$RUN_ROOT/cache/uv/${ATTEMPT_ID}"
export UV_PYTHON_INSTALL_DIR="$PYTHON_INSTALL_DIR"
export UV_NO_CONFIG=1
export UV_MANAGED_PYTHON=1
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV

python3 -m venv "$BOOTSTRAP_DIR"
"$BOOTSTRAP_DIR/bin/python" -m pip install \
    pip==26.2.2 setuptools==84.2.1 uv==0.13.6 pip-audit==2.10.1 filelock==4.2.4 \
    --index-url https://pypi.org/simple
"$BOOTSTRAP_DIR/bin/python" -m pip check
"$BOOTSTRAP_DIR/bin/python" -m pip freeze --all > "$BOOTSTRAP_DIR/requirements.freeze.txt"
"$BOOTSTRAP_DIR/bin/python" - \
    "$BOOTSTRAP_DIR/requirements.freeze.txt" "$BOOTSTRAP_DIR/requirements.audit.txt" <<'PY'
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
"$BOOTSTRAP_DIR/bin/pip-audit" --strict --no-deps --disable-pip --progress-spinner=off \
    --requirement "$BOOTSTRAP_DIR/requirements.audit.txt" --format=json --output "$BOOTSTRAP_AUDIT"
test "$("$BOOTSTRAP_DIR/bin/python" -c 'from importlib.metadata import version; print(version("uv"))')" = "$UV_VERSION"
[[ "$("$BOOTSTRAP_DIR/bin/uv" --version)" =~ ^uv[[:space:]]+${UV_VERSION}([[:space:]].*)?$ ]]

"$BOOTSTRAP_DIR/bin/uv" python install --managed-python --no-bin --no-config "$PYTHON_VERSION"
"$BOOTSTRAP_DIR/bin/uv" venv --managed-python --no-python-downloads --no-config --seed \
    --python "$PYTHON_VERSION" "$ENV_DIR"
test "$("$ENV_DIR/bin/python" -c 'import platform; print(platform.python_version())')" = "$PYTHON_VERSION"
TARGET_PYTHON=$("$ENV_DIR/bin/python" -c 'import sys; from pathlib import Path; print(Path(sys.executable).resolve())')
case "$TARGET_PYTHON" in
    "$PYTHON_INSTALL_DIR"/*) ;;
    *) echo "managed Python escaped isolated install directory: $TARGET_PYTHON" >&2; exit 1 ;;
esac

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
    "$ENV_DIR" "$PLATFORM" "$RUNTIME_SHA256" "$SOURCE_COMMIT" \
    "$BOOTSTRAP_DIR/requirements.freeze.txt" "$BOOTSTRAP_DIR/requirements.audit.txt" \
    "$BOOTSTRAP_AUDIT" "$BOOTSTRAP_DIR/bin/uv" "$UV_VERSION" \
    "$PYTHON_INSTALL_DIR" "$PYTHON_VERSION" <<'PY'
import hashlib
import json
import math
import platform
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
bootstrap_freeze, bootstrap_requirements = map(Path, sys.argv[9:11])
bootstrap_audit, uv_executable = map(Path, sys.argv[11:13])
python_install_dir = Path(sys.argv[14]).resolve()
python_executable = Path(sys.executable).resolve()
package_versions = {
    "anomalib": version("anomalib"),
    "torch": version("torch"),
    "torchvision": version("torchvision"),
    "pillow": version("pillow"),
    "filelock": version("filelock"),
    "pip": version("pip"),
    "setuptools": version("setuptools"),
    "pip_audit": version("pip-audit"),
}
expected_versions = {
    "anomalib": "2.7.0",
    "torch": "2.14.1+cu126",
    "torchvision": "0.29.1+cu126",
    "pillow": "13.0.0",
    "filelock": "4.2.4",
    "pip": "26.2.2",
    "setuptools": "84.2.1",
    "pip_audit": "2.10.1",
}
if package_versions != expected_versions:
    raise RuntimeError(f"installed package versions differ from secure pins: {package_versions}")
if platform.python_version() != sys.argv[15]:
    raise RuntimeError(f"expected Python {sys.argv[15]}, got {platform.python_version()}")
if python_install_dir not in python_executable.parents:
    raise RuntimeError(f"target Python escaped managed install directory: {python_executable}")
if sys.argv[13] != "0.13.6":
    raise RuntimeError(f"expected uv 0.13.6, got {sys.argv[13]}")
payload = {
    **package_versions,
    "runtime_kind": "brev-vm",
    "python": platform.python_version(),
    "python_install_dir": str(python_install_dir),
    "python_executable": str(python_executable),
    "python_executable_sha256": hashlib.sha256(python_executable.read_bytes()).hexdigest(),
    "uv": sys.argv[13],
    "uv_executable": str(uv_executable.resolve()),
    "uv_executable_sha256": hashlib.sha256(uv_executable.read_bytes()).hexdigest(),
    "bootstrap_freeze": str(bootstrap_freeze.resolve()),
    "bootstrap_freeze_sha256": hashlib.sha256(bootstrap_freeze.read_bytes()).hexdigest(),
    "bootstrap_audit_requirements": str(bootstrap_requirements.resolve()),
    "bootstrap_audit_requirements_sha256": hashlib.sha256(bootstrap_requirements.read_bytes()).hexdigest(),
    "bootstrap_pip_audit": str(bootstrap_audit.resolve()),
    "bootstrap_pip_audit_sha256": hashlib.sha256(bootstrap_audit.read_bytes()).hexdigest(),
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
test -s "$BOOTSTRAP_AUDIT"
test -s "$BOOTSTRAP_DIR/requirements.freeze.txt"
test -s "$BOOTSTRAP_DIR/requirements.audit.txt"
test -s "$PLATFORM"
sha256sum "$ARTIFACT" | awk '{print $1}' > "$ENV_DIR/.complete.tmp"
mv "$ENV_DIR/.complete.tmp" "$ENV_DIR/.complete"
printf '%s\n' "$ARTIFACT" > "$APPROVED.tmp"
mv "$APPROVED.tmp" "$APPROVED"
echo "T14 Brev runtime SHA-256: $RUNTIME_SHA256"
echo "T14 approved Brev PatchCore environment: $ENV_DIR"
