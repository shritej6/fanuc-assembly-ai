#!/bin/bash
# Serve the official Cosmos Reason 2 8B (nvidia/Cosmos-Reason2-8B, NVIDIA Open Model License, gated on Hugging Face)
# with vLLM's OpenAI-compatible server, on the GPU host itself. Runs in the pinned vllm/vllm-openai image (amd64 and
# arm64), so the host needs only Docker with the NVIDIA container toolkit. NVIDIA's model card asks for >= 32 GB of GPU
# memory and its repo for vllm >= 0.11.0 with --reasoning-parser qwen3 and --max-model-len 8192-16384.
#
#   bash serving/cosmos_reason2/serve.sh          # starts container "cosmos-reason2", waits until /health is 200
#
# The Hugging Face token must already be at $HF_DIR/token (copied file-to-file, never printed); the container reads it
# from the mounted cache, so it never appears on a command line or in the environment. The server listens on
# 127.0.0.1:$PORT only; reach it through `ssh -L`. Weights are pinned to MODEL_REVISION in agent/cosmos_reason2_endpoint.py.
set -euo pipefail

IMAGE="${IMAGE:-vllm/vllm-openai:v0.11.2@sha256:2c908d5a84ed251b6a17d179f42d06df1aff353007779ac5eecd8a0ea3fe9331}"
MODEL_REPO="nvidia/Cosmos-Reason2-8B"
MODEL_REVISION="${MODEL_REVISION:-a9fae2cf89dc64db96b12860417f0eb403013bb9}"
SERVED_NAME="Cosmos-Reason2-8B"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.90}"
HF_DIR="${HF_DIR:-$HOME/.cache/huggingface}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1800}"
NAME=cosmos-reason2

[[ -s "$HF_DIR/token" ]] || { echo "no Hugging Face token at $HF_DIR/token; copy the file there first" >&2; exit 2; }
command -v docker > /dev/null || { echo "docker is missing on this host" >&2; exit 2; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

docker rm -f "$NAME" > /dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --restart unless-stopped \
  -p "127.0.0.1:$PORT:8000" -v "$HF_DIR:/root/.cache/huggingface" \
  "$IMAGE" "$MODEL_REPO" --revision "$MODEL_REVISION" --served-model-name "$SERVED_NAME" \
  --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 1, "video": 0}' --seed 0 > /dev/null

deadline=$((SECONDS + READY_TIMEOUT_S))
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2> /dev/null)" != "true" ]] || ((SECONDS > deadline)); then
    echo "vLLM did not become healthy; last log lines:" >&2
    docker logs --tail 40 "$NAME" >&2
    exit 1
  fi
  sleep 10
done
echo "Cosmos Reason 2 8B healthy on 127.0.0.1:$PORT"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
