#!/bin/bash
# Prepare a rented Brev GPU VM (Ubuntu 22.04, sudo, NVIDIA driver older than 580) for sdg/brev/cosmos_worker.sh.
# Cosmos here needs CUDA 13 (sdg/cosmos/pod_setup.sh): datacenter GPUs run it on the old driver through NVIDIA's
# forward-compatibility package, and transformer-engine finds the CUDA runtime libraries through ldconfig, which
# RunPod images ship and these VMs do not (found on a driver-550 A100 box, 2026-10-29). The largest empty disk
# becomes /data for the models, the Cosmos repo and the bundles. The Hugging Face token goes to /data/hf/token
# (~/.cache/huggingface links there): the first run stops after the mount if it is missing; copy it, run again.
# Then start the workers with
#   LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat REPO_DIR=/data/cosmos-transfer2.5 sdg/brev/cosmos_worker.sh
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"

if ! mountpoint -q /data; then
  disk=$(lsblk -dbno NAME,SIZE,TYPE | awk '$3 == "disk"' | sort -k2 -n | tail -1 | cut -d" " -f1)
  if sudo blkid "/dev/$disk" > /dev/null; then
    echo "/dev/$disk already has a filesystem; mount it at /data by hand" >&2; exit 1
  fi
  sudo mkfs.ext4 -q -F "/dev/$disk" && sudo mkdir -p /data && sudo mount "/dev/$disk" /data
  sudo chown "$(id -u):$(id -g)" /data
fi
mkdir -p /data/hf /data/fa-cosmos ~/.cache
[[ -e ~/.cache/huggingface ]] || ln -s /data/hf ~/.cache/huggingface
ln -sfn /data/fa-cosmos ~/fa-cosmos
[[ -s /data/hf/token ]] || { echo "/data is ready: copy the Hugging Face token to /data/hf/token, run again" >&2; exit 2; }

wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
  -O /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb > /dev/null && sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends cuda-compat-13-0 cuda-libraries-13-0 ffmpeg git-lfs > /dev/null
echo /usr/local/cuda-13.0/targets/x86_64-linux/lib | sudo tee /etc/ld.so.conf.d/cuda-13-libs.conf > /dev/null
sudo ldconfig

REPO_DIR=/data/cosmos-transfer2.5 UV_CACHE_DIR=/data/uv-cache bash "$REPO/sdg/cosmos/pod_setup.sh"
cd /data/cosmos-transfer2.5
LD_LIBRARY_PATH=/usr/local/cuda-13.0/compat .venv/bin/python -c \
  "import torch, transformer_engine.pytorch; assert torch.cuda.is_available(); print(torch.cuda.device_count(), 'GPUs ok')"
