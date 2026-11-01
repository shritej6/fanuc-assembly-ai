# Runtime image around the host Cosmos venv, with FFmpeg 6 and CUDA 13.
# Authored by Shritej Gunda.
FROM nvidia/cuda:13.0.2-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git git-lfs && rm -rf /var/lib/apt/lists/*
