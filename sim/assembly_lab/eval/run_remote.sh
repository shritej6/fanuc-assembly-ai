#!/usr/bin/env bash
# Sync this worktree to the sim box and run the evaluator in agent 1's Isaac Lab container (env/isaaclab_run.sh).
#   sim/assembly_lab/eval/run_remote.sh <run-name> <policy> [run_eval.py args...]
#   e.g. sim/assembly_lab/eval/run_remote.sh scripted_v1 scripted
#        sim/assembly_lab/eval/run_remote.sh groot_e1 groot://127.0.0.1:5555
# Env: HOST (fa-mimic), GPU (3), IMAGE (isaac-lab-base:3.0.0b2p1-sim6.0.0), REMOTE_DIR (t12_agent6/FANUC_T32),
#      SKIP_READY=1 to run before agent 1 writes ~/ISAACLAB_READY.
set -euo pipefail
RUN=${1:?run name}; POLICY=${2:?policy}; shift 2
HOST=${HOST:-fa-mimic}; GPU=${GPU:-3}; IMAGE=${IMAGE:-isaac-lab-base:3.0.0b2p1-sim6.0.0}
REMOTE_DIR=${REMOTE_DIR:-t12_agent6/FANUC_T32}
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
[ -n "${SKIP_READY:-}" ] || ssh "$HOST" "test -f ~/ISAACLAB_READY" || { echo "$HOST has no ~/ISAACLAB_READY yet (agent 1)"; exit 2; }
ssh "$HOST" "mkdir -p ~/$REMOTE_DIR ~/t12_data/eval"
rsync -az --delete --exclude .git --exclude .venv --exclude '__pycache__' --exclude 'robot/cell/.venv' \
  --exclude 'sdg/output' --exclude '*.hdf5' "$REPO/" "$HOST:$REMOTE_DIR/"
ssh "$HOST" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader -i $GPU"
ssh "$HOST" "cd ~/$REMOTE_DIR && GPU=$GPU REPO=~/$REMOTE_DIR ISAACLAB_IMAGE=$IMAGE \
  bash sim/assembly_lab/env/isaaclab_run.sh sim/assembly_lab/eval/run_eval.py --run $RUN --policy $POLICY $* \
  > ~/t12_data/eval/$RUN.log 2>&1; echo exit=\$?"
