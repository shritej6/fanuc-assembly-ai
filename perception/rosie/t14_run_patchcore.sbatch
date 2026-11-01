#!/bin/bash
# Run one independently auditable PatchCore part. Submit as --array=0-23.
# Authored by Shritej Gunda.
#   sbatch --array=0-23 --export=ALL,RUN_ROOT,CODE_ROOT,SOURCE_COMMIT,DATASET_DIR,DATASET_ARTIFACT,\
#     ENV_DIR,ENV_ARTIFACT,ENV_AUDIT perception/rosie/t14_run_patchcore.sbatch
#SBATCH --job-name=fa-t14-pc-run
#SBATCH --partition=dgx
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%A_%a.log
set -euo pipefail

: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 runtime directory}"
: "${CODE_ROOT:?set CODE_ROOT to the exact T14 source snapshot}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the tested source revision}"
: "${DATASET_DIR:?set DATASET_DIR to one approved prepared dataset}"
: "${DATASET_ARTIFACT:?set DATASET_ARTIFACT to its preparation artifact}"
: "${ENV_DIR:?set ENV_DIR to one approved PatchCore environment}"
: "${ENV_ARTIFACT:?set ENV_ARTIFACT to its smoke artifact}"
: "${ENV_AUDIT:?set ENV_AUDIT to its strict pip-audit JSON}"
: "${BENCHMARK_GATE_SHA256:?set BENCHMARK_GATE_SHA256 from a measured matching benchmark gate}"
: "${BENCHMARK_GATE:?set BENCHMARK_GATE to the matching measured benchmark gate}"
[[ "$BENCHMARK_GATE_SHA256" =~ ^[0-9a-f]{64}$ ]]
test "$(sha256sum "$BENCHMARK_GATE" | awk '{print $1}')" = "$BENCHMARK_GATE_SHA256"
PART_INDEX=${PART_INDEX:-${SLURM_ARRAY_TASK_ID:-}}
: "${PART_INDEX:?set PART_INDEX or submit a Slurm array}"

SINGULARITY=/usr/local/singularity-3.10.0/bin/singularity
SIF=/data/containers/msoe-pytorch-23.05-py3.sif
RUN_ID="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}-${PART_INDEX}"
OUTPUT_ROOT="$RUN_ROOT/runs/attempts/patchcore-${RUN_ID}"
ARTIFACT="$RUN_ROOT/artifacts/patchcore_run-${RUN_ID}.json"
APPROVED="$RUN_ROOT/runs/approved/patchcore-${RUN_ID}.txt"
POINTER_TMP="$APPROVED.tmp"
MODEL_CACHE="$RUN_ROOT/cache/torch"

test -x "$SINGULARITY"
test -f "$SIF"
test -d "$CODE_ROOT"
test -d "$DATASET_DIR"
test -f "$DATASET_DIR/_READY.json"
test -f "$DATASET_DIR/manifest.jsonl"
test -f "$DATASET_ARTIFACT"
test -d "$ENV_DIR"
test -f "$ENV_DIR/.complete"
test -f "$ENV_DIR/requirements.freeze.txt"
test -f "$ENV_ARTIFACT"
test -f "$ENV_AUDIT"
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_ROOT"
test ! -e "$ARTIFACT"
test ! -e "$APPROVED"
test ! -e "$POINTER_TMP"
mkdir -p "$RUN_ROOT/runs/attempts" "$RUN_ROOT/runs/approved" "$RUN_ROOT/artifacts" "$MODEL_CACHE"

SIF_SHA256=$(sha256sum "$SIF" | awk '{print $1}')
NODE_NAME=$(hostname)
"$SINGULARITY" exec --nv --cleanenv \
    -B "$RUN_ROOT" \
    -B "$CODE_ROOT:$CODE_ROOT:ro" \
    -B "$DATASET_DIR:$DATASET_DIR:ro" \
    -B "$DATASET_ARTIFACT:$DATASET_ARTIFACT:ro" \
    -B "$ENV_DIR:$ENV_DIR:ro" \
    -B "$ENV_ARTIFACT:$ENV_ARTIFACT:ro" \
    -B "$ENV_AUDIT:$ENV_AUDIT:ro" \
    "$SIF" python "$CODE_ROOT/perception/rosie/t14_execute_patchcore.py" \
    --code-root "$CODE_ROOT" --source-commit "$SOURCE_COMMIT" \
    --dataset-dir "$DATASET_DIR" --dataset-artifact "$DATASET_ARTIFACT" \
    --env-dir "$ENV_DIR" --env-artifact "$ENV_ARTIFACT" --env-audit "$ENV_AUDIT" \
    --model-cache "$MODEL_CACHE" --part-index "$PART_INDEX" \
    --output-root "$OUTPUT_ROOT" --artifact "$ARTIFACT" --workers 8 \
    --runner-id "$SLURM_JOB_ID" --hostname "$NODE_NAME" --runtime-sha256 "$SIF_SHA256" \
    --run-kind production --benchmark-gate "$BENCHMARK_GATE" \
    --benchmark-gate-sha256 "$BENCHMARK_GATE_SHA256"

test -s "$ARTIFACT"
printf '%s\n' "$ARTIFACT" > "$POINTER_TMP"
mv "$POINTER_TMP" "$APPROVED"
echo "T14 approved PatchCore part artifact: $ARTIFACT"
