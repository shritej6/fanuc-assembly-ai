# Runtime around the host's Cosmos venv (sdg/cosmos/pod_setup.sh) on Ubuntu 22.04 VMs: Cosmos's decord needs FFmpeg 6
# (libavformat.so.60), which only Ubuntu 24.04 ships, and the CUDA 13 libraries; both come from official images and
# repositories here. The venv, its uv Python and the bundles are mounted at their host paths (sdg/brev/cosmos_worker.sh).
#   docker build -t fa-cosmos-runtime -f sdg/brev/cosmos_runtime.Dockerfile sdg/brev
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git git-lfs && rm -rf /var/lib/apt/lists/*
