#!/bin/bash
# User order steps 4-5 on the 8xH100 box, called by h100_testbed.sh after the set-P model.
# PatchCore: own pinned environment; the strict pip-audit must pass first (user), else PatchCore never runs unaudited
# and this stage fails. One part per GPU, parts in waves of $GPUS (24 parts = 3 waves on 8 GPUs), no DDP. The stage
# succeeds only when every one of the 24 M parts of contracts/taxonomy.json has a fresh metrics.json with status
# "measured"; anything less exits nonzero, so the driver records "t14 FAILED" and never writes ALL_DONE.
# Needs the old C / I chunks with _REST.json in $OLD_ROOT (test images only, never training).
set -euo pipefail

REPO="${REPO:-$HOME/fanuc-t27}"
DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-testbed-data}"
OLD_ROOT="${OLD_ROOT:-$HOME/fanuc-oldsets/synthetic}"
PC="${PC:-$HOME/fanuc-t27-runs/patchcore}"   # inside the runs dir the watcher copies to Rosie
PC_VENV="${PC_VENV:-$HOME/fanuc-patchcore-venv}"
STATUS="${STATUS:-$HOME/t27_status.txt}"
UV="$HOME/.local/bin/uv"
GPUS="$(nvidia-smi -L | wc -l)"
mark() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }

# torch 2.10 / pillow 12 / filelock 4.1 / old pip + setuptools failed the strict audit (Rosie job 290706, 81 known
# vulnerabilities); these versions passed it with 0 and run anomalib 2.7.0 (checked 2026-10-29).
# Called as `patchcore_env || ...`, where bash turns set -e off inside the function: every step checks its own exit,
# or a failed audit would be followed by a passing import check and the function would report success.
patchcore_env() {
  "$UV" venv -q -p 3.12 "$PC_VENV" || return 1
  "$UV" pip install -q -p "$PC_VENV/bin/python" torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126 || return 1
  "$UV" pip install -q -p "$PC_VENV/bin/python" anomalib==2.7.0 pillow==13.0.0 filelock==4.2.4 pip==26.2.2 \
    setuptools==84.2.1 pip-audit==2.10.1 || return 1
  "$PC_VENV/bin/python" -m pip check || return 1
  "$PC_VENV/bin/python" -m pip freeze --all | sed -E 's/\+[A-Za-z0-9.]+$//' > "$PC/requirements.audit.txt" || return 1
  "$PC_VENV/bin/pip-audit" --strict --no-deps --disable-pip --progress-spinner=off \
    --requirement "$PC/requirements.audit.txt" --format=json --output "$PC/pip_audit.json" || return 1
  "$PC_VENV/bin/python" -c 'import torch, anomalib; assert torch.cuda.is_available(); print(torch.__version__, anomalib.__version__)'
}

patchcore() {
  rm -rf "$PC/crops" "$PC/runs" "$PC"/runs_*.log   # outputs of an earlier run must not count as this run's
  "$PC_VENV/bin/python" -m perception.anomaly.testbed --new-root "$DATA_ROOT/synthetic" --old-root "$OLD_ROOT" \
    --out "$PC/crops" > "$PC/crops.log"
  local parts   # object IDs hold no spaces, so word splitting gives one part per element
  parts=($("$PC_VENV/bin/python" -c \
    'import json,sys; print(" ".join(o["object_id"] for o in json.load(open(sys.argv[1]))["objects"] if o["object_id"].startswith("M")))' \
    "$REPO/contracts/taxonomy.json"))
  (( ${#parts[@]} == 24 )) || { mark "patchcore expected 24 M parts in the taxonomy, found ${#parts[@]}"; return 1; }
  local i=0
  for part in "${parts[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i % GPUS)) "$PC_VENV/bin/python" -m perception.anomaly.train_patchcore \
      --crops "$PC/crops" --part "$part" --out "$PC/runs" > "$PC/runs_$part.log" 2>&1 &
    i=$((i + 1))
    if (( i % GPUS == 0 )); then wait; fi
  done
  wait
  local unmeasured
  unmeasured=$("$PC_VENV/bin/python" -c '
import json, sys
from pathlib import Path
runs = Path(sys.argv[1])
for part in sys.argv[2:]:
    path = runs / part / "metrics.json"
    status = json.loads(path.read_text()).get("status") if path.is_file() else "no metrics.json"
    if status != "measured":
        print(f"{part}={status}")' "$PC/runs" "${parts[@]}")
  if [[ -n "$unmeasured" ]]; then mark "patchcore parts not measured: $(echo $unmeasured)"; return 1; fi
  mark "patchcore all ${#parts[@]} parts measured"
}

cd "$REPO"
mkdir -p "$PC"
mark "patchcore_env start"
patchcore_env > "$PC/env.log" 2>&1 || { mark "patchcore_env blocked: strict audit or setup did not pass (see $PC/env.log)"; exit 1; }
mark "patchcore_env ok (strict audit passed)"
mark "patchcore start"
patchcore
mark "patchcore ok"
