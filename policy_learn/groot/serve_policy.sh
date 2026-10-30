#!/bin/bash
# Serve a fine-tuned checkpoint with Isaac-GR00T's own policy server (ZMQ, port 5555) for the sim eval or the cell.
#   bash policy_learn/groot/serve_policy.sh [checkpoint dir] [gpu]
# Default checkpoint: the newest one under ~/groot_runs/main (or ~/groot_artifacts/groot_final once exported).
# It binds to 127.0.0.1 only (the server has no authentication); reach it with an SSH tunnel from the client:
#   ssh -N -L 5555:127.0.0.1:5555 <training box>      then GrootTcpClient(host="127.0.0.1", port=5555)
set -Eeuo pipefail

GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
PORT="${PORT:-5555}"
ckpt="${1:-}"
if [[ -z "$ckpt" ]]; then
  ckpt="$(ls -d "$HOME"/groot_runs/main/checkpoint-*/ 2>/dev/null | sort -t- -k2 -n | tail -1 | sed 's:/$::')"
  [[ -d "$HOME/groot_artifacts/groot_final" ]] && ckpt="$HOME/groot_artifacts/groot_final"
fi
[[ -d "$ckpt" ]] || { echo "no checkpoint found (pass one as the first argument)" >&2; exit 2; }
echo "serving $ckpt on 127.0.0.1:$PORT (GPU ${2:-0})"
cd "$GROOT_DIR"
CUDA_VISIBLE_DEVICES="${2:-0}" exec "$GROOT_DIR/.venv/bin/python" gr00t/eval/run_gr00t_server.py \
  --model-path "$ckpt" --embodiment-tag NEW_EMBODIMENT --device cuda:0 --host 127.0.0.1 --port "$PORT"
