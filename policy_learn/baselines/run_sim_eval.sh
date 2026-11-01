#!/usr/bin/env bash
# Run agent 6's sim evaluator on an ACT / SmolVLA policy server, unchanged: the server speaks GR00T's protocol, so the
# evaluator gets `--policy groot://127.0.0.1:$PORT` and every chunk passes the same GrootPolicy guard as GR00T's.
#   policy_learn/baselines/run_sim_eval.sh <run-name> <train-box> [run_eval.py args...]
#   e.g. PORT=5561 policy_learn/baselines/run_sim_eval.sh act_e1 fa-b1 --only nominal_M4 --layouts 1
# Before: on <train-box>, `python -m policy_learn.baselines.policy_server --port $PORT --checkpoint ...` is running.
# This host (needs to stay awake for the run) chains two SSH tunnels: sim box 127.0.0.1:$PORT -> here -> train box.
# Env: PORT (5561), EVAL_HOST (fa-mimic, run_remote.sh's HOST), GPU / IMAGE passed through to run_remote.sh,
#      REMOTE_DIR (t12_agent12/FANUC_T36, a separate checkout so agent 6's sync dir is never overwritten).
# Results: <EVAL_HOST>:~/t12_data/eval/<run-name>/results.json (same format as GR00T's runs).
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
