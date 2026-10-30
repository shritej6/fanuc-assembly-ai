#!/usr/bin/env bash
# Run a Python script inside the Isaac Lab 3.0.0-beta2 (Isaac Sim 6.0.0) container on fa-sim / fa-mimic.
#
#   GPU=0 REPO=~/t12_data/repo_env ~/t12_data/repo_env/sim/assembly_lab/env/isaaclab_run.sh \
#       sim/assembly_lab/env/env_smoke.py --out ~/t12_data/env_smoke --minimal
#
# GPU   one GPU index (default 0). Each container sees only that GPU, as cuda:0.
# REPO  a repo checkout on the host (default ~/t12_data/repo_env); mounted at /workspace/repo, which is the cwd.
# REPO is also mounted at its own host path, and ~/t12_data at /root/t12_data and at $HOME/t12_data, so host paths work
# inside. PYTHONPATH has the repo root (generate_dataset.py --external_callback sim.assembly_lab.env.register.register_tasks).
# T12_POOL / T12_CAMERAS / T12_IMAGE / T12_SAMPLER / T12_CONTACT_YAML / T12_JITTER are passed through when set (register.py).
# Kit / shader caches persist per GPU under ~/t12_data/.isaac_cache/gpu$GPU (first start compiles shaders, minutes).
set -euo pipefail
GPU="${GPU:-0}"
REPO="$(realpath "${REPO:-$HOME/t12_data/repo_env}")"
DATA="$(realpath "$HOME/t12_data")"
IMAGE="${ISAACLAB_IMAGE:-isaac-lab-base:3.0.0b2p1-sim6.0.0}"
CACHE="$DATA/.isaac_cache/gpu$GPU"
mkdir -p "$CACHE"/{kit,ov,pip,glcache,computecache,logs,data}
NAME="isaaclab-gpu${GPU}-$(date +%H%M%S)-$$"
exec docker run --rm --name "$NAME" --gpus "device=$GPU" --network host --user root \
  -e ACCEPT_EULA=Y -e OMNI_KIT_ACCEPT_EULA=YES -e OMNI_KIT_ALLOW_ROOT=1 -e HOME=/root \
  -e T12_ASSET_CACHE=/root/t12_data/env_assets -e PYTHONUNBUFFERED=1 -e PYTHONPATH=/workspace/repo \
  -e T12_POOL -e T12_CAMERAS -e T12_IMAGE -e T12_SAMPLER -e T12_CONTACT_YAML -e T12_JITTER -e T12_CONTACT_SET \
  -v "$REPO":/workspace/repo -v "$REPO":"$REPO" -v "$DATA":/root/t12_data -v "$DATA":"$HOME/t12_data" \
  -v "$CACHE/kit":/isaac-sim/kit/cache -v "$CACHE/ov":/root/.cache/ov -v "$CACHE/pip":/root/.cache/pip \
  -v "$CACHE/glcache":/root/.cache/nvidia/GLCache -v "$CACHE/computecache":/root/.nv/ComputeCache \
  -v "$CACHE/logs":/root/.nvidia-omniverse/logs -v "$CACHE/data":/root/.local/share/ov/data \
  -w /workspace/repo --entrypoint /workspace/isaaclab/isaaclab.sh "$IMAGE" -p "$@"
