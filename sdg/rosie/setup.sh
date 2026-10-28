#!/bin/bash
# One-time setup on the Rosie login node: host venv for the scene sampler + data folders.
#
#   git clone https://github.com/shritej6/fanuc-assembly-ai.git ~/fanuc-assembly-ai
#   ~/fanuc-assembly-ai/sdg/rosie/setup.sh
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-assembly-data}"
VENV="${HOST_VENV:-$HOME/fa-venv}"

if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/pip" install --quiet "numpy>=1.26" "pyyaml>=6"
mkdir -p "$DATA_ROOT"/{specs,synthetic,cache,models,runs}
chmod 700 "$DATA_ROOT"

echo "venv:      $VENV"
echo "DATA_ROOT: $DATA_ROOT ($(df -h "$DATA_ROOT" | awk 'NR==2 {print $4}') free on its filesystem)"
echo "Isaac containers on Rosie:"
ls -1 /data/containers/isaac-sim*.sif
