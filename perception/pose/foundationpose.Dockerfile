# Pinned FoundationPose runtime for the pretrained pose check (user: own environment, separate from PatchCore and T13).
# The upstream image is CUDA 11.3 (no sm_90), so this follows upstream's "conda (local)" recipe on NVIDIA's official
# CUDA 12.4 devel image, with the GPU extensions compiled for the H100 only. Build context = the FoundationPose
# checkout (NVlabs/FoundationPose a1b694b):
#   docker build -t fa-foundationpose -f foundationpose.Dockerfile /scratch/FoundationPose
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="9.0" FORCE_CUDA=1 PATH=/opt/fp/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-venv git cmake ninja-build build-essential \
      libeigen3-dev libboost-all-dev pybind11-dev libgl1 libegl1 libglib2.0-0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/fp && pip install --no-cache-dir pip==24.3.1 setuptools==75.6.0 wheel==0.45.1
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
# pytorch3d v0.7.8 and nvdiffrast v0.3.3, by commit
RUN pip install --no-cache-dir --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
RUN pip install --no-cache-dir --no-build-isolation \
      "git+https://github.com/NVlabs/nvdiffrast.git@729261dc64c4241ea36efda84fbf532cc8b425b8"
COPY requirements.txt /tmp/fp-requirements.txt
RUN pip install --no-cache-dir -r /tmp/fp-requirements.txt && pip freeze --all > /opt/fp/freeze.txt
# The validation script rebuilds the jittered render camera with pxr.Gf, exactly as the generator posed it
RUN pip install --no-cache-dir usd-core==26.8 && pip freeze --all > /opt/fp/freeze.txt
# open3d (from requirements.txt) loads libusb at import
RUN apt-get update && apt-get install -y --no-install-recommends libusb-1.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*
# Ubuntu's pybind11 2.9 predates NumPy 2 (mycpp.cluster_poses segfaults on any array); build mycpp against 2.13:
#   docker run ... -w /fp fa-foundationpose bash -c 'CONDA_PREFIX=/opt/fp bash build_all_conda.sh'
RUN pip install --no-cache-dir pybind11==2.13.6 && pip freeze --all > /opt/fp/freeze.txt
ENV CMAKE_PREFIX_PATH=/opt/fp/lib/python3.10/site-packages/pybind11
# requirements.txt leaves warp-lang unpinned; 1.10+ wheels need a CUDA 13 driver (the H100 host has 570 / CUDA 12.8)
RUN pip install --no-cache-dir warp-lang==1.9.1 && pip freeze --all > /opt/fp/freeze.txt
