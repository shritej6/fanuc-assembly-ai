#!/bin/bash
# Unattended render queue for a multi-GPU Docker host, one serial queue per GPU.
# Authored by Shritej Gunda.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-assembly-data}"
VENV="${HOST_VENV:-$HOME/fa-venv}"
export DATA_ROOT

setup() {
  [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet "numpy>=1.26" "pyyaml>=6" "trimesh>=4" manifold3d scipy pillow
  (cd "$REPO" && "$VENV/bin/python" sdg/tools/defect_variants.py)   # verified against the manifest
  mkdir -p "$DATA_ROOT"/{specs,synthetic,cache,logs}
  docker pull --quiet "${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
}

render_with_retries() {  # gpu set chunk
  local out
  out="$DATA_ROOT/synthetic/$2/chunk_$(printf "%04d" "$3")"
  for try in 1 2 3; do
    DOCKER_GPUS="device=$1" "$REPO/sdg/spark/render_chunk.sh" "$2" "$3" > /dev/null 2>&1 || true
    [[ -f "$out/_DONE.json" ]] && { echo "$(date +%T) gpu$1 $2/$3 done"; return 0; }
    [[ -f "$out/_FAILED.txt" ]] && mv "$out/_FAILED.txt" "$out/_FAILED_retry$try.txt"
    echo "$(date +%T) gpu$1 $2/$3 attempt $try failed"
  done
  echo "$(date +%T) gpu$1 $2/$3 GAVE UP after 3 attempts"
}

queue() {  # gpu "SET:first-last ..."
  local gpu=$1 item
  for item in $2; do
    local set=${item%%:*} range=${item#*:}
    for c in $(seq "${range%-*}" "${range#*-}"); do
      render_with_retries "$gpu" "$set" "$c"
    done
  done
}

[[ -n "${SKIP_SETUP:-}" ]] || setup
echo "$(date +%T) setup done; $# queues"
gpu=0
for plan in "$@"; do
  queue "$gpu" "$plan" &
  [[ -n "${ONE_GPU:-}" ]] || gpu=$((gpu + 1))
done
wait
date +%T > "$DATA_ROOT/QUEUE_DONE"
echo "$(date +%T) all queues finished"
