#!/bin/bash
# Write the scene specs for a set on the login node, then submit one Slurm array task per chunk.
#
#   sdg/rosie/submit.sh A            # all chunks of set A
#   sdg/rosie/submit.sh A 0-3        # chunks 0..3
#   sdg/rosie/submit.sh pilot        # first 375 images of chunk 0 of A, B, C, D (reused by the full run)
#
# Needs: ISAAC_SIF, DATA_ROOT (default ~/fanuc-assembly-data), the host venv from sdg/rosie/setup.sh.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-assembly-data}"
PY="${HOST_PY:-$HOME/fa-venv/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
: "${ISAAC_SIF:?set ISAAC_SIF to the Isaac Sim container (see docs/SDG_PLAN.md, section Compute)}"
export ISAAC_SIF DATA_ROOT REPO

specs() {   # set chunk
  (cd "$REPO" && "$PY" -m sdg.scene_sampler --set "$1" --chunk "$2" \
    --out "$DATA_ROOT/specs/$1/chunk_$(printf "%04d" "$2").jsonl")
}

submit() {  # set array-range limit
  sbatch --parsable --array="$2%$MAX_PARALLEL" --export=ALL,SET="$1",LIMIT="$3" \
    "$REPO/sdg/rosie/render_chunk.sbatch"
}

if [[ "$1" == "pilot" ]]; then
  for SET in A B C D; do
    specs "$SET" 0
    LIMIT=$(cd "$REPO" && "$PY" -c "from sdg.scene_sampler import Inputs; i=Inputs.load(); \
print(i.sets['pilot']['images_per_set'] // len(i.sets['sets']['$SET']['cameras']))")
    echo "pilot $SET: job $(submit "$SET" 0 "$LIMIT")  ($LIMIT scenes)"
  done
  exit 0
fi

SET="$1"
N=$(cd "$REPO" && "$PY" -m sdg.scene_sampler --set "$SET" --n-chunks)
RANGE="${2:-0-$((N - 1))}"
FIRST="${RANGE%-*}"
LAST="${RANGE#*-}"
for ((c = FIRST; c <= LAST; c++)); do
  specs "$SET" "$c"
done
echo "set $SET chunks $RANGE: job $(submit "$SET" "$RANGE" 0)"
