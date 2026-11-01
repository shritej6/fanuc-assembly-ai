#!/bin/bash
# Install the pinned Isaac-GR00T N1.7 stack on a fresh Ubuntu x86_64 GPU box (idempotent).
# Authored by Shritej Gunda.
#   bash policy_learn/groot/install_groot.sh
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=pins.env
source "$HERE/pins.env"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
UV="$HOME/.local/bin/uv"

install_system() {
  sudo apt-get update -qq
  # libaio-dev and the CUDA 12.8 toolkit: deepspeed needs CUDA_HOME/nvcc to start (Isaac-GR00T
  # scripts/deployment/dgpu/install_deps.sh).
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git rsync python3-pip libaio-dev > /dev/null
  if [[ ! -d /usr/local/cuda ]]; then
    if ! apt-cache show cuda-toolkit-12-8 &> /dev/null; then
      local ubuntu keyring
      ubuntu="$(. /etc/os-release && echo "${VERSION_ID//./}")"
      keyring="$(mktemp --suffix=.deb)"
      curl -fsSL -o "$keyring" \
        "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${ubuntu}/x86_64/cuda-keyring_1.1-1_all.deb"
      sudo dpkg -i "$keyring"
      rm -f "$keyring"
      sudo apt-get update -qq
    fi
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends cuda-toolkit-12-8 > /dev/null
  fi
  local major
  major="$(ffmpeg -version | head -1 | sed -E 's/^ffmpeg version n?([0-9]+).*/\1/')"
  if ! [[ "$major" =~ ^[4-7]$ ]]; then
    echo "ffmpeg major version '$major' is not 4-7; torchcodec 0.8.0 cannot load it (Ubuntu 25.10+?)" >&2
    exit 3
  fi
}

install_uv() {
  if [[ -x "$UV" ]] && "$UV" --version | grep -q " $UV_VERSION"; then return; fi
  python3 -m pip install --user --quiet --break-system-packages "uv==$UV_VERSION" 2>/dev/null \
    || python3 -m pip install --user --quiet "uv==$UV_VERSION"
}

checkout_groot() {
  # No submodules (they are the sim benchmarks) and no LFS (demo data only): neither is used here.
  if [[ ! -d "$GROOT_DIR/.git" ]]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none "$GROOT_REPO_URL" "$GROOT_DIR"
  fi
  GIT_LFS_SKIP_SMUDGE=1 git -C "$GROOT_DIR" -c filter.lfs.process= -c filter.lfs.smudge=cat \
    -c filter.lfs.required=false fetch --quiet origin "$GROOT_COMMIT"
  GIT_LFS_SKIP_SMUDGE=1 git -C "$GROOT_DIR" -c filter.lfs.process= -c filter.lfs.smudge=cat \
    -c filter.lfs.required=false checkout --quiet --force "$GROOT_COMMIT"
  [[ "$(git -C "$GROOT_DIR" rev-parse HEAD)" == "$GROOT_COMMIT" ]]
}

sync_env() {
  cd "$GROOT_DIR"
  "$UV" python install "$PYTHON_VERSION"
  "$UV" sync --frozen --python "$PYTHON_VERSION"
  # CUDA_HOME for flash-attn / triton JIT (README: "If fine-tuning fails with CUDA_HOME is unset").
  [[ -d /usr/local/cuda ]] && export CUDA_HOME=/usr/local/cuda
  "$GROOT_DIR/.venv/bin/python" - <<'PY'
import torch, transformers, gr00t, torchcodec  # noqa: F401  torchcodec import fails if ffmpeg is unusable
import flash_attn  # noqa: F401
import deepspeed  # noqa: F401  import runs deepspeed's op checks, which raise without CUDA_HOME
assert torch.cuda.is_available(), "torch sees no GPU"
print(f"torch {torch.__version__} cuda {torch.version.cuda} gpus {torch.cuda.device_count()} "
      f"transformers {transformers.__version__}")
PY
  # Client-side deps of policy_learn.groot (already in the lock; checked so a later import cannot fail).
  "$GROOT_DIR/.venv/bin/python" -c 'import msgpack, msgpack_numpy, zmq, pandas, pyarrow'
}

install_system
install_uv
checkout_groot
sync_env
echo "GR00T ready at $GROOT_DIR commit $GROOT_COMMIT"
