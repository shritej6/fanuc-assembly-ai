#!/bin/bash
# Build Rosie training environment without the broken snap-packaged uv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV_PATH="${VENV_PATH:-$HOME/fanuc-t13-venv}"
PYTHON="${PYTHON:-python3.12}"
DRY_RUN="${DRY_RUN:-0}"

run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run "$PYTHON" -m venv "$VENV_PATH"
run "$VENV_PATH/bin/python" -m pip install --no-cache-dir --upgrade pip==25.3
run "$VENV_PATH/bin/python" -m pip install --no-cache-dir \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  torch==2.6.0+cu118 torchvision==0.21.0+cu118
run "$VENV_PATH/bin/python" -m pip install --no-cache-dir \
  accelerate==1.15.0 pycocotools==2.0.11 safetensors==0.8.0 \
  scipy==1.18.1 timm==1.0.30 'torchmetrics[detection]==1.9.0' transformers==4.57.3 \
  jsonschema==4.26.0 numpy==2.5.4 pillow==13.0.0 pyyaml==6.0.3
run "$VENV_PATH/bin/python" -m pip check
run "$VENV_PATH/bin/python" -c \
  'import scipy, torch, torchvision, transformers; assert torch.version.cuda == "11.8"; print(scipy.__version__, torch.__version__, torchvision.__version__, transformers.__version__)'
