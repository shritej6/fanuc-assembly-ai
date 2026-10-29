#!/bin/bash
# Cosmos side of the render -> Cosmos pipeline (sdg/brev/hub.sh): one worker per GPU on a multi-GPU Cosmos host.
# The hub drops each rendered chunk's bundle into ~/fa-cosmos/inbox/; a free GPU claims it (an atomic mv), restyles
# it with sdg/cosmos/run_transfer.sh and moves it to ~/fa-cosmos/done/, where the hub fetches the results.
# A bundle that fails three times goes to ~/fa-cosmos/failed/ for a look. Needs sdg/cosmos/pod_setup.sh first.
#
#   nohup sdg/brev/cosmos_worker.sh > ~/fa-cosmos/workers.log 2>&1 &     # stops when ~/fa-cosmos/STOP exists
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$HOME/fa-cosmos"
export PATH="$HOME/.local/bin:$PATH"   # uv / uvx (Cosmos calls uvx); nohup shells do not read the login profile
mkdir -p "$ROOT"/{incoming,inbox,work,done,failed}

# COSMOS_RUNTIME=fa-cosmos-runtime (sdg/brev/cosmos_runtime.Dockerfile): run the host venv inside that image, for VMs
# whose OS lacks FFmpeg 6 / CUDA 13 libraries. On these VMs ~/.cache and ~/.cache/huggingface are symlinks to other
# disks, so their real folders are mounted at their real paths too. Our uid has no passwd entry in the image, and
# torch's compile cache needs a user name: USER / LOGNAME give it one.
run_bundle() {  # gpu bundle
  if [[ -z "${COSMOS_RUNTIME:-}" ]]; then
    CUDA_VISIBLE_DEVICES=$1 RUNNER=venv BUNDLE="$2" bash "$REPO/sdg/cosmos/run_transfer.sh"
    return
  fi
  local mounts=(-v /data:/data -v "$HOME:$HOME") p
  for p in "$HOME/.cache" "$HOME/.cache/huggingface"; do mounts+=(-v "$(readlink -f "$p"):$(readlink -f "$p")"); done
  docker run --rm --gpus "device=$1" --ipc=host -u "$(id -u):$(id -g)" -w "$REPO" "${mounts[@]}" \
    -e HOME -e USER="$(id -un)" -e LOGNAME="$(id -un)" -e PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/bin:/bin" \
    -e LD_LIBRARY_PATH=/usr/local/cuda/compat -e RUNNER=venv -e REPO_DIR="${REPO_DIR:?}" -e BUNDLE="$2" \
    "$COSMOS_RUNTIME" bash sdg/cosmos/run_transfer.sh
}

worker() {  # gpu
  local gpu=$1 b name
  while [[ ! -f "$ROOT/STOP" ]]; do
    b=""
    for cand in "$ROOT"/inbox/*; do
      [[ -d "$cand" ]] || continue
      name=$(basename "$cand")
      mv "$cand" "$ROOT/work/$name" 2>/dev/null && { b="$ROOT/work/$name"; break; }
    done
    [[ -z "$b" ]] && { sleep 30; continue; }
    for try in 1 2 3; do
      run_bundle "$gpu" "$b" > /dev/null 2>&1
      [[ $(find "$b/out" -name '*.jpg' | wc -l) -ge $(find "$b/specs" -name '*.json' | wc -l) ]] && break
      echo "$(date +%T) gpu$gpu $name attempt $try incomplete"
    done
    if [[ $(find "$b/out" -name '*.jpg' | wc -l) -ge $(find "$b/specs" -name '*.json' | wc -l) ]]; then
      rm -rf "$b/raw" "$b/batch" && mv "$b" "$ROOT/done/$name" && echo "$(date +%T) gpu$gpu $name done"
    else
      mv "$b" "$ROOT/failed/$name" && echo "$(date +%T) gpu$gpu $name FAILED after 3 attempts"
    fi
  done
}

n=$(nvidia-smi -L | wc -l)
echo "$(date +%T) $n Cosmos workers"
for gpu in $(seq 0 $((n - 1))); do worker "$gpu" & done
wait
