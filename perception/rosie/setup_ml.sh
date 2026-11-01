#!/bin/bash
# Build Rosie training environment without the broken snap-packaged uv.
# Authored by Shritej Gunda.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_PATH="${VENV_PATH:-$HOME/fanuc-t13-venv-secure}"
PYTHON="${PYTHON:-python3.12}"
DRY_RUN="${DRY_RUN:-0}"
REQUIREMENTS="${REQUIREMENTS:-$SCRIPT_DIR/requirements-ml-cu126.txt}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run "$PYTHON" -m venv --clear "$VENV_PATH"
run "$VENV_PATH/bin/python" -m pip install --no-cache-dir \
  --require-hashes --only-binary=:all: --no-index --requirement "$REQUIREMENTS"
run "$VENV_PATH/bin/python" -m pip check
run "$VENV_PATH/bin/python" -c \
  'import scipy, torch, torchvision, transformers; assert torch.version.cuda == "12.6"; print(scipy.__version__, torch.__version__, torchvision.__version__, transformers.__version__)'
