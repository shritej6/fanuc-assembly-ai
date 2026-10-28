#!/bin/bash
# One Slurm array task = one chunk of one set. Submit with sdg/rosie/submit.sh (it writes the specs first).
# UNTESTED until the Rosie pilot (T08).
#SBATCH --job-name=fa-sdg
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.log
set -euo pipefail

: "${ISAAC_SIF:?set ISAAC_SIF to the Isaac Sim container (see docs/SDG_PLAN.md, section Compute)}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${REPO:?set REPO to the repo checkout on Rosie}"
: "${SET:?set SET (A, B, C or D)}"
LIMIT="${LIMIT:-0}"

CHUNK=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
SPECS="$DATA_ROOT/specs/$SET/chunk_$CHUNK.jsonl"
OUT="$DATA_ROOT/synthetic/$SET/chunk_$CHUNK"
if [[ -f "$OUT/_DONE.json" ]]; then
  echo "chunk $SET/$CHUNK already done"
  exit 0
fi
mkdir -p "$OUT" "$DATA_ROOT/cache/kit" "$DATA_ROOT/cache/ov" "$DATA_ROOT/cache/nv"

echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  set $SET  chunk $CHUNK  limit $LIMIT"
# The SIF is read-only: Kit's shader / asset caches go to DATA_ROOT so they persist between chunks.
singularity exec --nv --writable-tmpfs \
  --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,ISAAC_SIF="$ISAAC_SIF",FANUC_ROBOT_USD="${FANUC_ROBOT_USD:-}" \
  -B "$REPO" -B "$DATA_ROOT" \
  -B "$DATA_ROOT/cache/kit:/isaac-sim/kit/cache" \
  -B "$DATA_ROOT/cache/ov:$HOME/.cache/ov" \
  -B "$DATA_ROOT/cache/nv:$HOME/.nv" \
  "$ISAAC_SIF" /isaac-sim/python.sh "$REPO/sdg/isaac/generate.py" \
  --specs "$SPECS" --out "$OUT" --limit "$LIMIT"
