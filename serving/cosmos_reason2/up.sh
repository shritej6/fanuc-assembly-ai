#!/bin/bash
# Serve Cosmos Reason 2 8B on a GPU host, tunnel it to http://127.0.0.1:8000/v1 and run the health check.
# Authored by Shritej Gunda.
#   MODE=spark bash serving/cosmos_reason2/up.sh        # DGX Spark host
#   bash serving/cosmos_reason2/up.sh                   # fallback: rented Brev box $BOX
#   MODE=spark bash serving/cosmos_reason2/up.sh down   # stop the server and tunnel (brev: also deletes the box)
# TOKEN_FILE is a local Hugging Face token file; it is copied with scp and never printed.
set -euo pipefail

MODE="${MODE:-brev}"
TYPES="${TYPES:-massedcompute_L40S,hyperstack_A100_80G,massedcompute_A100_sxm4_80G,hyperstack_H100,scaleway_H100}"
TOKEN_FILE="${TOKEN_FILE:-$HOME/.cache/huggingface/token}"
PORT="${PORT:-8000}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1500}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
case "$MODE" in
  spark) HOST="${HOST:-spark}"; WORK=fanuc-t35 ;;
  brev) HOST="${BOX:-fa-c1}"; WORK=cosmos_reason2 ;;
  *) echo "MODE must be spark or brev" >&2; exit 2 ;;
esac

box_line() { brev ls 2> /dev/null | awk -v b="$HOST" '$1 == b'; }
close_tunnel() { pkill -f "ssh -fN -L $PORT:127.0.0.1:$PORT .*$HOST\$" || true; }

if [[ "${1:-}" == "down" ]]; then
  close_tunnel
  if [[ "$MODE" == brev ]]; then brev delete "$HOST"; else ssh "$HOST" 'docker rm -f cosmos-reason2'; fi
  exit 0
fi

[[ -s "$TOKEN_FILE" ]] || { echo "no Hugging Face token file at $TOKEN_FILE" >&2; exit 2; }
if [[ "$MODE" == brev ]]; then
  [[ -n "$(box_line)" ]] || { echo "creating $HOST (types: $TYPES)"; brev create "$HOST" --type "$TYPES"; }
  deadline=$((SECONDS + READY_TIMEOUT_S))
  until box_line | grep -q "RUNNING.*COMPLETED.*READY"; do
    ((SECONDS < deadline)) || { echo "$HOST not READY in time: $(box_line)" >&2; exit 1; }
    sleep 20
  done
  brev refresh > /dev/null
  echo "$HOST ready"
fi

ssh -o StrictHostKeyChecking=accept-new "$HOST" "mkdir -p ~/$WORK/hf && chmod 700 ~/$WORK/hf"
scp -q "$TOKEN_FILE" "$HOST:$WORK/hf/token"
ssh "$HOST" "chmod 600 ~/$WORK/hf/token"
scp -q "$HERE/serve.sh" "$HOST:$WORK/serve.sh"
echo "starting vLLM on $HOST (first start downloads the image and ~17 GB of weights)"
ssh "$HOST" "MODE=$MODE PORT=$PORT bash ~/$WORK/serve.sh"

close_tunnel
ssh -fN -L "$PORT:127.0.0.1:$PORT" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 "$HOST"
echo "tunnel up: http://127.0.0.1:$PORT/v1"
cd "$REPO" && uv run python -m serving.cosmos_reason2.health --base-url "http://127.0.0.1:$PORT/v1"
