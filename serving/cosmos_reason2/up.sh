#!/bin/bash
# One command, from the laptop at the repo root, Saturday morning: rent a Brev GPU box, serve Cosmos Reason 2 8B on it,
# tunnel it to http://127.0.0.1:8000/v1 and run the health check.
#
#   bash serving/cosmos_reason2/up.sh                 # BOX=fa-c1 by default
#   BOX=fa-c1 bash serving/cosmos_reason2/up.sh down  # close the tunnel and DELETE the box when the run is over
#
# TYPES is the Brev fallback chain (first that provisions wins; all have >= 48 GB, NVIDIA asks for >= 32 GB).
# TOKEN_FILE is a local Hugging Face token file whose account has accepted the Cosmos-Reason2-8B licence on
# huggingface.co; it is copied file-to-file with scp and never printed. Each step prints one line; any failure stops.
set -euo pipefail

BOX="${BOX:-fa-c1}"
TYPES="${TYPES:-massedcompute_L40S,hyperstack_A100_80G,massedcompute_A100_sxm4_80G,hyperstack_H100,scaleway_H100}"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.cache/huggingface/token}"
PORT="${PORT:-8000}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1500}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

box_line() { brev ls 2> /dev/null | awk -v b="$BOX" '$1 == b'; }

if [[ "${1:-}" == "down" ]]; then
  pkill -f "ssh -fN -L $PORT:127.0.0.1:$PORT $BOX" || true
  brev delete "$BOX"
  exit 0
fi

[[ -s "$TOKEN_FILE" ]] || { echo "no Hugging Face token file at $TOKEN_FILE" >&2; exit 2; }
if [[ -z "$(box_line)" ]]; then
  echo "creating $BOX (types: $TYPES)"
  brev create "$BOX" --type "$TYPES"
fi
deadline=$((SECONDS + READY_TIMEOUT_S))
until box_line | grep -q "RUNNING.*COMPLETED.*READY"; do
  ((SECONDS < deadline)) || { echo "$BOX not READY in time: $(box_line)" >&2; exit 1; }
  sleep 20
done
echo "$BOX ready: $(box_line | awk '{print $6}')"
brev refresh > /dev/null

ssh -o StrictHostKeyChecking=accept-new "$BOX" 'mkdir -p ~/.cache/huggingface ~/cosmos_reason2 && chmod 700 ~/.cache/huggingface'
scp -q "$TOKEN_FILE" "$BOX:.cache/huggingface/token"
ssh "$BOX" 'chmod 600 ~/.cache/huggingface/token'
scp -q "$HERE/serve.sh" "$BOX:cosmos_reason2/serve.sh"
echo "starting vLLM on $BOX (first start downloads ~17 GB of weights)"
ssh "$BOX" "PORT=$PORT bash ~/cosmos_reason2/serve.sh"

pkill -f "ssh -fN -L $PORT:127.0.0.1:$PORT $BOX" || true
ssh -fN -L "$PORT:127.0.0.1:$PORT" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "$BOX"
echo "tunnel up: http://127.0.0.1:$PORT/v1"
cd "$REPO" && uv run python -m serving.cosmos_reason2.health --base-url "http://127.0.0.1:$PORT/v1"
