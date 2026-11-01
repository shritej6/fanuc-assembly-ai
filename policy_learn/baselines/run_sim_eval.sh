#!/usr/bin/env bash
# Run the sim evaluator against an ACT or SmolVLA policy server through the same GrootPolicy guard.
# Authored by Shritej Gunda.
#   PORT=5561 policy_learn/baselines/run_sim_eval.sh <run-name> <train-box> [run_eval.py args...]
# Results land in <EVAL_HOST>:~/t12_data/eval/<run-name>/results.json.
set -euo pipefail
RUN=${1:?run name}; BOX=${2:?training box with the policy server}; shift 2
PORT=${PORT:-5561}; EVAL_HOST=${EVAL_HOST:-fa-mimic}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
ssh -N -o ExitOnForwardFailure=yes -L "$PORT:127.0.0.1:$PORT" "$BOX" & pids+=($!)
ssh -N -o ExitOnForwardFailure=yes -R "$PORT:127.0.0.1:$PORT" "$EVAL_HOST" & pids+=($!)
sleep 5
for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null || { echo "tunnel failed (port $PORT busy?)" >&2; exit 2; }; done
HOST="$EVAL_HOST" REMOTE_DIR="${REMOTE_DIR:-t12_agent12/FANUC_T36}" \
  "$REPO/sim/assembly_lab/eval/run_remote.sh" "$RUN" "groot://127.0.0.1:$PORT" "$@"
