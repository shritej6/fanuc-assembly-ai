#!/bin/bash
# User order steps 4-5 on the 8xH100 box, called by h100_testbed.sh after the set-P model.
# PatchCore: own pinned environment; the strict pip-audit must pass first (user), else PatchCore is skipped, never run
# unaudited. One part per GPU, parts in waves of $GPUS (24 parts = 3 waves on 8 GPUs), single-GPU per part (no DDP).
# Status lines never say FAILED (that word stops the watcher); the whole run still ends with ALL_DONE.
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
patchcore_env() {
  "$UV" venv -q -p 3.12 "$PC_VENV"
  "$UV" pip install -q -p "$PC_VENV/bin/python" torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cu126
  "$UV" pip install -q -p "$PC_VENV/bin/python" anomalib==2.7.0 pillow==13.0.0 filelock==4.2.4 pip==26.2.2 \
    setuptools==84.2.1 pip-audit==2.10.1
  "$PC_VENV/bin/python" -m pip check
  "$PC_VENV/bin/python" -m pip freeze --all | sed -E 's/\+[A-Za-z0-9.]+$//' > "$PC/requirements.audit.txt"
  "$PC_VENV/bin/pip-audit" --strict --no-deps --disable-pip --progress-spinner=off \
    --requirement "$PC/requirements.audit.txt" --format=json --output "$PC/pip_audit.json"
  "$PC_VENV/bin/python" -c 'import torch, anomalib; assert torch.cuda.is_available(); print(torch.__version__, anomalib.__version__)'
}

patchcore() {
  "$PC_VENV/bin/python" -m perception.anomaly.testbed --new-root "$DATA_ROOT/synthetic" --old-root "$OLD_ROOT" \
    --out "$PC/crops" > "$PC/crops.log"
  mapfile -t parts < <("$PC_VENV/bin/python" -c 'import json,sys; print("\n".join(sorted(json.load(open(sys.argv[1])))))' \
    "$PC/crops/counts.json")
  local i=0
  for part in "${parts[@]}"; do
    CUDA_VISIBLE_DEVICES=$((i % GPUS)) "$PC_VENV/bin/python" -m perception.anomaly.train_patchcore \
      --crops "$PC/crops" --part "$part" --out "$PC/runs" > "$PC/runs_$part.log" 2>&1 &
    i=$((i + 1))
    if (( i % GPUS == 0 )); then wait; fi
  done
  wait
  local done_n
  done_n=$(find "$PC/runs" -name metrics.json | wc -l)
  mark "patchcore parts with metrics: $done_n of ${#parts[@]}"
  [[ $done_n == "${#parts[@]}" ]]
}

cd "$REPO"
mkdir -p "$PC"
mark "patchcore_env start"
if patchcore_env > "$PC/env.log" 2>&1; then
  mark "patchcore_env ok (strict audit passed)"
  mark "patchcore start"
  if patchcore; then mark "patchcore ok"; else mark "patchcore PARTS_INCOMPLETE (see $PC/runs_*.log)"; fi
else
  mark "patchcore SKIPPED: strict audit or setup did not pass (see $PC/env.log, $PC/pip_audit.json)"
fi
