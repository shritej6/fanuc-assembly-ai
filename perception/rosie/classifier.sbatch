#!/bin/bash
# Train the exact 30-class wrist classifier on Rosie T4s.
#SBATCH --job-name=fa-t13-classifier
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="${REPO:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
if [[ -f "$REPO/.t13-code-commit" ]]; then
  IFS= read -r FANUC_CODE_COMMIT < "$REPO/.t13-code-commit"
  export FANUC_CODE_COMMIT
fi
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$HOME/fanuc-t13/artifacts}"
RUN_ROOT="${RUN_ROOT:-$HOME/fanuc-t13/runs}"
PYTHON="${PYTHON:-$HOME/fanuc-t13-venv-secure/bin/python}"
TORCHRUN="${TORCHRUN:-$HOME/fanuc-t13-venv-secure/bin/torchrun}"
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${SLURM_GPUS_ON_NODE:-1}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi

MODE="full"
if [[ "$SMOKE" == "1" ]]; then
  MODE="smoke"
  MAX_STEPS="${MAX_STEPS:-2}"
else
  "$PYTHON" -m perception.rosie.gates "$ARTIFACT_ROOT/full/manifests/metadata.json"
fi

COMMAND=("$TORCHRUN" --standalone --nnodes=1 --nproc_per_node "$NPROC_PER_NODE"
  -m perception.classifier.train
  --config "$REPO/perception/configs/classifier_t4.yaml"
  --crops "$ARTIFACT_ROOT/$MODE/crops/crops.jsonl"
  --manifest-metadata "$ARTIFACT_ROOT/$MODE/manifests/metadata.json"
  --output-dir "$RUN_ROOT/$MODE/classifier")
if [[ "$SMOKE" == "1" ]]; then
  COMMAND+=(--smoke --max-steps "$MAX_STEPS")
fi
if [[ -n "$RESUME_FROM_CHECKPOINT" ]]; then
  COMMAND+=(--resume-from-checkpoint "$RESUME_FROM_CHECKPOINT")
fi
printf 'precision=fp16 bf16=false tf32=false\n'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" != "1" ]]; then
  export NVIDIA_TF32_OVERRIDE=0
  cd "$REPO"
  "${COMMAND[@]}"
fi
