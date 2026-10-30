#!/usr/bin/env bash
# One command on fa-mimic: generate Mimic demos for some (skill, product) groups, export them to LeRobot v2.1
# (policy_learn/groot/DATA_FORMAT.md), and run agent 5's prepare_dataset.py check on the export.
#
#   ISAAC_PY="<Isaac Lab python, see ~/ISAACLAB_READY>" MIMIC_GROUPS="pick_insert:4" TARGET=50 NAME=t12_pi_p4_small \
#       bash sim/assembly_lab/mimic/pipeline.sh
#
# Env: ISAAC_PY (required), MIMIC_GROUPS ("skill:product ..."), TARGET (accepted demos per worker), WORKERS (per group),
#      GPUS ("0 1 2"), NAME (dataset folder under ~/t12_data/lerobot), MAX_EPISODES (export cap, empty = all),
#      TOOLS_PY (python with h5py / pandas / pyarrow, default ~/t12_tools/bin/python), TOUCH_PROXY (1 = accept the
#      gripper_obj proxy for the touch gate when the env has no sim.robot_touch; stamped on every demo).
set -euo pipefail
cd "$(dirname "$0")/../../.."

: "${ISAAC_PY:?set ISAAC_PY to the Isaac Lab python}"
MIMIC_GROUPS=${MIMIC_GROUPS:-"pick_insert:4"}
TARGET=${TARGET:-50}
WORKERS=${WORKERS:-3}
GPUS=${GPUS:-"0 1 2"}
NAME=${NAME:-t12_small}
MAX_EPISODES=${MAX_EPISODES:-}
TOOLS_PY=${TOOLS_PY:-$HOME/t12_tools/bin/python}
DATA=$HOME/t12_data
EXTRA=()
[[ "${TOUCH_PROXY:-0}" == 1 ]] && EXTRA+=(--allow-touch-proxy)

# shellcheck disable=SC2086
"$TOOLS_PY" -m sim.assembly_lab.mimic.run_generation --python "$ISAAC_PY" --demos "$DATA/demos" \
    --out "$DATA/mimic" --groups $MIMIC_GROUPS --gpus $GPUS --workers-per-group "$WORKERS" --target "$TARGET" \
    "${EXTRA[@]}"

INPUTS=()
for g in $MIMIC_GROUPS; do
    INPUTS+=("$DATA/mimic/${g%%:*}/p${g##*:}"/w*.hdf5)
done
CAP=()
[[ -n "$MAX_EPISODES" ]] && CAP=(--max-episodes "$MAX_EPISODES")
"$TOOLS_PY" -m sim.assembly_lab.mimic.export_lerobot --inputs "${INPUTS[@]}" --out "$DATA/lerobot/$NAME" \
    --workers 16 "${CAP[@]}"
"$TOOLS_PY" -m sim.assembly_lab.mimic.validate_lerobot "$DATA/lerobot/$NAME" --video-every 10
"$TOOLS_PY" -m policy_learn.groot.prepare_dataset --src "$DATA/lerobot/$NAME" --out "$DATA/prepare_check/$NAME"
echo "READY: $DATA/lerobot/$NAME"
