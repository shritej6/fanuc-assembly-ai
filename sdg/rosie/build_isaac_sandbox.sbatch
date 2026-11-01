#!/bin/bash
# Build a user-owned Isaac Sim 6.0.0 sandbox container on Rosie.
# Authored by Shritej Gunda.
#SBATCH --job-name=fa-isaac-build
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

TAG="${ISAAC_TAG:-6.0.0}"
DEST="${ISAAC_SANDBOX:-$HOME/containers/isaac-sim-$TAG}"
export SINGULARITY_CACHEDIR="$HOME/containers/.cache"
export SINGULARITY_TMPDIR="$HOME/containers/.tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR" "$(dirname "$DEST")"

if [[ -x "$DEST/isaac-sim/python.sh" ]]; then
  echo "sandbox already built: $DEST"
  exit 0
fi
rm -rf "$DEST.partial"
singularity build --sandbox "$DEST.partial" "docker://nvcr.io/nvidia/isaac-sim:$TAG"
chmod -R u+rwX "$DEST.partial/isaac-sim"
mv "$DEST.partial" "$DEST"
rm -rf "$SINGULARITY_TMPDIR"
singularity exec "$DEST" ls -l /isaac-sim/python.sh
echo "built $DEST ($(du -sh "$DEST" | cut -f1))"
