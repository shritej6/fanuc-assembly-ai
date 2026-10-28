#!/bin/bash
# Build an Isaac Sim 6.0.0 sandbox (unpacked container) owned by the user.
# Why: /data/containers/isaac-sim-6.1.0.sif has /isaac-sim readable only by root, and the stack pins 6.0.
# An unprivileged sandbox build extracts every file as the calling user, so the result is readable.
#
#   sbatch sdg/rosie/build_isaac_sandbox.sbatch        # then: export ISAAC_SIF=~/containers/isaac-sim-6.0.0
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
