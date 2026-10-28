#!/bin/bash
# Render chunks on the GB10 (DGX Spark, arm64) with the official Isaac Sim 6.0.0 container.
# Same generator and specs as Rosie; one chunk at a time (one GPU).
#
#   sdg/spark/render_chunk.sh C 0 1 2        # set C, chunks 0 1 2
#   LIMIT=3 sdg/spark/render_chunk.sh B 0    # smoke test: first 3 scenes only
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-assembly-data}"
PY="${HOST_PY:-$HOME/fa-venv/bin/python}"
IMAGE="${ISAAC_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.0}"
LIMIT="${LIMIT:-0}"
SET="$1"
shift

mkdir -p "$DATA_ROOT"/cache/home "$DATA_ROOT"/{logs,synthetic}
# The image runs as its own user (uid 1234, owner of /isaac-sim) in our group, so our folders must be group-writable.
# Its own files stay its own; only touch what we own. HOME=/cache/home keeps Kit / shader caches between runs.
find "$DATA_ROOT/cache" "$DATA_ROOT/synthetic" "$DATA_ROOT/logs" \( ! -user "$(id -u)" -prune \) -o \
  \( -type d -exec chmod g+rwxs {} + \)
for c in "$@"; do
  CHUNK=$(printf "%04d" "$c")
  SPECS="$DATA_ROOT/specs/$SET/chunk_$CHUNK.jsonl"
  OUT="$DATA_ROOT/synthetic/$SET/chunk_$CHUNK"
  if [[ -f "$OUT/_DONE.json" ]]; then
    echo "chunk $SET/$CHUNK already done"
    continue
  fi
  (cd "$REPO" && "$PY" -m sdg.scene_sampler --set "$SET" --chunk "$c" --out "$SPECS")
  mkdir -p "$OUT" && chmod g+rwxs "$OUT"
  docker run --rm --gpus all --network host -u "1234:$(id -g)" \
    -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e HOME=/cache/home -e ISAAC_SIF="$IMAGE" \
    -v "$REPO:$REPO:ro" -v "$DATA_ROOT:$DATA_ROOT" -v "$DATA_ROOT/cache:/cache" \
    --entrypoint bash "$IMAGE" -c 'umask 002 && exec /isaac-sim/python.sh "$@"' _ \
    "$REPO/sdg/isaac/generate.py" --specs "$SPECS" --out "$OUT" --limit "$LIMIT" \
    2>&1 | tee "$DATA_ROOT/logs/spark_${SET}_${CHUNK}.log"
done
