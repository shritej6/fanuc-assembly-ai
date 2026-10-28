#!/bin/bash
# One-time Cosmos Transfer 2.5 setup on an x86 GPU pod (A100/H100, driver >= 570 for CUDA 12.8), no Docker inside.
# Same Cosmos commit as the GB10 image, so every host renders with the same code. Then:
#   RUNNER=venv BUNDLE=/workspace/<bundle> bash sdg/cosmos/run_transfer.sh
set -euo pipefail
REPO_DIR=${REPO_DIR:-$HOME/cosmos-transfer2.5}
COSMOS_COMMIT=2ff49d0
[ -s "$HOME/.cache/huggingface/token" ] || { echo "copy the Hugging Face token to ~/.cache/huggingface/token" >&2; exit 1; }
command -v ffmpeg >/dev/null || (apt-get update -qq && apt-get install -y -qq ffmpeg git-lfs >/dev/null)
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh && sh /tmp/uv-install.sh
fi
export PATH="$HOME/.local/bin:$PATH"
[ -d "$REPO_DIR" ] || git clone -q https://github.com/nvidia-cosmos/cosmos-transfer2.5.git "$REPO_DIR"
git -C "$REPO_DIR" checkout -q "$COSMOS_COMMIT"
cd "$REPO_DIR" && uv python install && uv sync --extra=cu128
echo "cosmos ready at $(git -C "$REPO_DIR" rev-parse --short HEAD)"
