#!/bin/bash
# Serve a fine-tuned checkpoint with Isaac-GR00T's own policy server (ZMQ, port 5555) for the sim eval or the cell.
#   ATTESTATION_JSON=/path/to/attestation.json bash policy_learn/groot/serve_policy.sh [checkpoint dir] [gpu]
# Default checkpoint: the newest one under ~/groot_runs/main (or ~/groot_artifacts/groot_final once exported).
# It binds to 127.0.0.1 only (the server has no authentication); reach it with an SSH tunnel from the client:
#   ssh -N -L 5555:127.0.0.1:5555 <training box>      then GrootTcpClient(host="127.0.0.1", port=5555)
set -Eeuo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CALLER_PWD="$PWD"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
PORT="${PORT:-5555}"
UPSTREAM_PORT="${UPSTREAM_PORT:-5556}"
ATTESTATION_JSON="${ATTESTATION_JSON:-}"
ckpt="${1:-}"
if [[ -z "$ckpt" ]]; then
  ckpt="$(ls -d "$HOME"/groot_runs/main/checkpoint-*/ 2>/dev/null | sort -t- -k2 -n | tail -1 | sed 's:/$::')"
  [[ -d "$HOME/groot_artifacts/groot_final" ]] && ckpt="$HOME/groot_artifacts/groot_final"
fi
[[ -d "$ckpt" ]] || { echo "no checkpoint found (pass one as the first argument)" >&2; exit 2; }
[[ -f "$ATTESTATION_JSON" ]] || {
  echo "ATTESTATION_JSON must name the reviewed deployment manifest" >&2
  exit 2
}
[[ "$ckpt" = /* ]] || ckpt="$CALLER_PWD/$ckpt"
[[ "$ATTESTATION_JSON" = /* ]] || ATTESTATION_JSON="$CALLER_PWD/$ATTESTATION_JSON"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
echo "serving attested $ckpt on 127.0.0.1:$PORT via upstream :$UPSTREAM_PORT (GPU ${2:-0})"
PYTHONPATH="$REPO_ROOT" "$GROOT_DIR/.venv/bin/python" -m policy_learn.groot.attested_server \
  --checkpoint "$ckpt" --attestation "$ATTESTATION_JSON" --verify-only
cd "$GROOT_DIR"
CUDA_VISIBLE_DEVICES="${2:-0}" "$GROOT_DIR/.venv/bin/python" gr00t/eval/run_gr00t_server.py \
  --model-path "$ckpt" --embodiment-tag NEW_EMBODIMENT --device cuda:0 --host 127.0.0.1 --port "$UPSTREAM_PORT" &
upstream_pid=$!
cleanup() {
  kill "$upstream_pid" 2>/dev/null || true
  wait "$upstream_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM
PYTHONPATH="$REPO_ROOT" "$GROOT_DIR/.venv/bin/python" -m policy_learn.groot.attested_server \
  --checkpoint "$ckpt" --attestation "$ATTESTATION_JSON" --host 127.0.0.1 --port "$PORT" \
  --upstream-port "$UPSTREAM_PORT"
