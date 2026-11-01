#!/bin/bash
# Serve nvidia/Cosmos-Reason2-8B with the vLLM OpenAI-compatible server on the GPU host, bound to 127.0.0.1.
# Authored by Shritej Gunda.
#   MODE=spark bash serve.sh   # DGX Spark (NGC vLLM image)
#   MODE=brev  bash serve.sh   # rented x86 Brev box (vllm/vllm-openai image)
# The Hugging Face token must already be at $HF_DIR/token; weights are pinned in agent/cosmos_reason2_endpoint.py.
set -euo pipefail

MODE="${MODE:-brev}"
case "$MODE" in
  spark)
    IMAGE="${IMAGE:-nvcr.io/nvidia/vllm:26.09-py3@sha256:557747337846eecf34486866f3d992383c6f8af869fdb0e365b5093fcaee02c2}"
    ENTRY=(vllm serve)                 # NGC images start through nvidia_entrypoint.sh, which runs the given command
    WORK="${WORK:-$HOME/fanuc-t35}"
    GPU_UTIL="${GPU_UTIL:-0.25}"       # of 121 GB unified memory: ~30 GB (8B bf16 weights ~17 GB), shared box
    RESTART=no ;;
  brev)
    IMAGE="${IMAGE:-vllm/vllm-openai:v0.11.2@sha256:2c908d5a84ed251b6a17d179f42d06df1aff353007779ac5eecd8a0ea3fe9331}"
    ENTRY=()                           # the image's entrypoint is already `vllm serve`
    WORK="${WORK:-$HOME/cosmos_reason2}"
    GPU_UTIL="${GPU_UTIL:-0.90}"
    RESTART=unless-stopped ;;
  *) echo "MODE must be spark or brev" >&2; exit 2 ;;
esac
MODEL_REPO="${MODEL_REPO:-nvidia/Cosmos-Reason2-8B}"
MODEL_REVISION="${MODEL_REVISION:-a9fae2cf89dc64db96b12860417f0eb403013bb9}"
SERVED_NAME="${SERVED_NAME:-Cosmos-Reason2-8B}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
HF_DIR="${HF_DIR:-$WORK/hf}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-2400}"
NAME=cosmos-reason2

[[ -s "$HF_DIR/token" ]] || { echo "no Hugging Face token at $HF_DIR/token; copy the file there first" >&2; exit 2; }
command -v docker > /dev/null || { echo "docker is missing on this host" >&2; exit 2; }
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

docker rm -f "$NAME" > /dev/null 2>&1 || true
docker run -d --name "$NAME" --gpus all --ipc=host --restart "$RESTART" \
  -p "127.0.0.1:$PORT:8000" -v "$HF_DIR:/root/.cache/huggingface" -e HF_HOME=/root/.cache/huggingface \
  "$IMAGE" "${ENTRY[@]}" "$MODEL_REPO" --revision "$MODEL_REVISION" --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_UTIL" \
  --limit-mm-per-prompt '{"image": 1, "video": 0}' --seed 0 > /dev/null

deadline=$((SECONDS + READY_TIMEOUT_S))
until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2> /dev/null)" != "true" ]] || ((SECONDS > deadline)); then
    echo "vLLM did not become healthy; last log lines:" >&2
    docker logs --tail 40 "$NAME" >&2
    exit 1
  fi
  sleep 10
done
echo "$SERVED_NAME ($MODEL_REPO@${MODEL_REVISION:0:8}) healthy on 127.0.0.1:$PORT"
