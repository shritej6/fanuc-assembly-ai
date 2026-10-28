#!/bin/bash
# Freeze T13 manifests and export COCO/crops. SMOKE=1 is the only native-only mode.
#SBATCH --job-name=fa-t13-prepare
#SBATCH --partition=teaching
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-assembly-data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/fanuc-t13/artifacts}"
PYTHON="${PYTHON:-$HOME/fanuc-t13-venv/bin/python}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "$SMOKE" != "0" && "$SMOKE" != "1" ]]; then
  echo "SMOKE must be 0 or 1" >&2
  exit 2
fi

MODE="full"
if [[ "$SMOKE" == "1" ]]; then
  MODE="smoke"
fi

COMMAND=("$PYTHON" -m perception.rosie.prepare
  --data-root "$DATA_ROOT"
  --output-root "$ARTIFACT_ROOT/$MODE"
  --taxonomy "$REPO/contracts/taxonomy.json")
if [[ "$SMOKE" == "1" ]]; then
  COMMAND+=(--smoke)
fi
if [[ "$DRY_RUN" == "1" ]]; then
  COMMAND+=(--dry-run)
fi
printf ' %q' "${COMMAND[@]}"
printf '\n'
cd "$REPO"
"${COMMAND[@]}"
