#!/usr/bin/env bash
# Run a Python script inside the Isaac Lab 3.0.0-beta2 (Isaac Sim 6.0.0) container.
# Authored by Shritej Gunda.
#
#   GPU=0 REPO=~/t12_data/repo_env sim/assembly_lab/env/isaaclab_run.sh sim/assembly_lab/env/env_smoke.py --minimal
#
# GPU   one GPU index (default 0), seen inside the container as cuda:0.
# REPO  repo checkout on the host, mounted at /workspace/repo and used as the cwd.
# The T12_* settings read by register.py are passed through when set. Shader caches persist per GPU.
set -euo pipefail
GPU="${GPU:-0}"
REPO="$(realpath "${REPO:-$HOME/t12_data/repo_env}")"
DATA="$(realpath "$HOME/t12_data")"
IMAGE="${ISAACLAB_IMAGE:-isaac-lab-base:3.0.0b2p1-sim6.0.0}"
CACHE="$DATA/.isaac_cache/gpu$GPU${CACHE_TAG:-}"    # CACHE_TAG: a second container on the same GPU gets its own cache
mkdir -p "$CACHE"/{kit,ov,pip,glcache,computecache,logs,data}
NAME="isaaclab-gpu${GPU}-$(date +%H%M%S)-$$"
exec docker run --rm --name "$NAME" --gpus "device=$GPU" --network host --user root \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_KIT_ALLOW_ROOT=1 -e HOME=/root \
  -e T12_ASSET_CACHE=/root/t12_data/env_assets -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace/repo \
  -e T12_POOL -e T12_CAMERAS -e T12_IMAGE -e T12_SAMPLER -e T12_CONTACT_YAML -e T12_JITTER -e T12_CONTACT_SET -e T12_SUBSTEP_LOG \
  -v "$REPO":/workspace/repo -v "$REPO":"$REPO" -v "$DATA":/root/t12_data -v "$DATA":"$HOME/t12_data" \
  -v "$CACHE/kit":/isaac-sim/kit/cache -v "$CACHE/ov":/root/.cache/ov -v "$CACHE/pip":/root/.cache/pip \
  -v "$CACHE/glcache":/root/.cache/nvidia/GLCache -v "$CACHE/computecache":/root/.nv/ComputeCache \
  -v "$CACHE/logs":/root/.nvidia-omniverse/logs -v "$CACHE/data":/root/.local/share/ov/data \
  -w /workspace/repo --entrypoint /workspace/isaaclab/isaaclab.sh "$IMAGE" -p "$@"
