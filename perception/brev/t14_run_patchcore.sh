#!/bin/bash
# Run one PatchCore part on Brev: benchmark first, production only after its measured gate.
# Authored by Shritej Gunda.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 Brev runtime directory}"
: "${CODE_ROOT:?set CODE_ROOT to the exact clean T14 Git snapshot}"
: "${SOURCE_COMMIT:?set SOURCE_COMMIT to the tested source revision}"
: "${DATASET_DIR:?set DATASET_DIR to the copied approved prepared dataset}"
: "${DATASET_ARTIFACT:?set DATASET_ARTIFACT to its preparation artifact}"
: "${ENV_DIR:?set ENV_DIR to the approved Brev environment}"
: "${ENV_ARTIFACT:?set ENV_ARTIFACT to its environment artifact}"
: "${ENV_AUDIT:?set ENV_AUDIT to its strict pip-audit JSON}"
: "${RUNTIME_SHA256:?set RUNTIME_SHA256 to the Brev platform identity SHA-256}"
: "${PLATFORM_IDENTITY:?set PLATFORM_IDENTITY to the hashed Brev platform JSON}"
: "${BREV_INSTANCE_ID:?set BREV_INSTANCE_ID from brev ls}"
: "${ATTEMPT_ID:?set ATTEMPT_ID to a unique immutable attempt ID}"
: "${PART_INDEX:?set PART_INDEX in [0, 23]}"
: "${RUN_MODE:?set RUN_MODE to benchmark or production}"
[[ "$RUN_ROOT" = /* ]]
[[ "$CODE_ROOT" = /* ]]
[[ "$DATASET_DIR" = /* ]]
[[ "$DATASET_ARTIFACT" = /* ]]
[[ "$ENV_DIR" = /* ]]
[[ "$ENV_ARTIFACT" = /* ]]
[[ "$ENV_AUDIT" = /* ]]
[[ "$PLATFORM_IDENTITY" = /* ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$ATTEMPT_ID" =~ ^[a-z0-9][a-z0-9-]{0,63}$ ]]
[[ "$PART_INDEX" =~ ^([0-9]|1[0-9]|2[0-3])$ ]]
test "$PART_INDEX" -ge 0
test "$PART_INDEX" -le 23
case "$RUN_MODE" in benchmark|production) ;; *) exit 2 ;; esac
test "$(git -C "$CODE_ROOT" rev-parse HEAD)" = "$SOURCE_COMMIT"
test -z "$(git -C "$CODE_ROOT" status --porcelain --untracked-files=all)"
test -z "$(git -C "$CODE_ROOT" clean -ndx)"
test "$(sha256sum "$ENV_ARTIFACT" | awk '{print $1}')" = "$(tr -d '\n' < "$ENV_DIR/.complete")"
test "$(sha256sum "$PLATFORM_IDENTITY" | awk '{print $1}')" = "$RUNTIME_SHA256"
LIVE_HOSTNAME=$(hostname)
LIVE_GPU_ROW=$(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader,nounits)
test "$(printf '%s\n' "$LIVE_GPU_ROW" | wc -l | tr -d ' ')" = 1
"$ENV_DIR/bin/python" - "$PLATFORM_IDENTITY" "$BREV_INSTANCE_ID" "$LIVE_HOSTNAME" "$LIVE_GPU_ROW" <<'PY'
import json
import sys
from pathlib import Path

platform = json.loads(Path(sys.argv[1]).read_text())
expected = {
    "runtime_kind": "brev-vm",
    "brev_instance_id": sys.argv[2],
    "hostname": sys.argv[3],
    "gpu_query": sys.argv[4],
}
if any(platform.get(key) != value for key, value in expected.items()):
    raise RuntimeError("live Brev hardware disagrees with the approved platform identity")
PY
mkdir -p "$RUN_ROOT/cache/torch" "$RUN_ROOT/locks" "$RUN_ROOT/artifacts"
exec 9>"$RUN_ROOT/locks/patchcore.lock"
flock -n 9 || { echo "another T14 PatchCore process is active" >&2; exit 1; }

if [ "$RUN_MODE" = benchmark ]; then
    test "$PART_INDEX" = 0
    : "${HOURLY_RATE_USD:?set the measured Brev listing rate for the benchmark projection}"
    : "${HOURLY_RATE_EVIDENCE:?set the immutable JSON evidence for the measured Brev listing rate}"
    [[ "$HOURLY_RATE_EVIDENCE" = /* ]]
    : "${BENCHMARK_GATE:?set BENCHMARK_GATE to a fresh gate output path}"
    OUTPUT_ROOT="$RUN_ROOT/benchmarks/attempts/patchcore-${ATTEMPT_ID}"
    ARTIFACT="$RUN_ROOT/artifacts/patchcore_benchmark-${ATTEMPT_ID}.json"
    APPROVED="$RUN_ROOT/benchmarks/approved/patchcore-${ATTEMPT_ID}.txt"
    mkdir -p "$RUN_ROOT/benchmarks/attempts" "$RUN_ROOT/benchmarks/approved"
else
    : "${BENCHMARK_GATE:?set BENCHMARK_GATE to the approved matching benchmark gate}"
    : "${BENCHMARK_GATE_SHA256:?set the recorded benchmark gate SHA-256}"
    [[ "$BENCHMARK_GATE_SHA256" =~ ^[0-9a-f]{64}$ ]]
    test "$(sha256sum "$BENCHMARK_GATE" | awk '{print $1}')" = "$BENCHMARK_GATE_SHA256"
    PYTHONPATH="$CODE_ROOT" "$ENV_DIR/bin/python" "$CODE_ROOT/perception/brev/t14_benchmark_gate.py" verify \
        --gate "$BENCHMARK_GATE" --dataset-artifact "$DATASET_ARTIFACT" \
        --env-artifact "$ENV_ARTIFACT" --source-commit "$SOURCE_COMMIT" \
        --runtime-sha256 "$RUNTIME_SHA256"
    OUTPUT_ROOT="$RUN_ROOT/runs/attempts/${ATTEMPT_ID}/patchcore-${PART_INDEX}"
    ARTIFACT="$RUN_ROOT/artifacts/patchcore_run-${ATTEMPT_ID}-${PART_INDEX}.json"
    APPROVED="$RUN_ROOT/runs/approved/${ATTEMPT_ID}/patchcore-${PART_INDEX}.txt"
    mkdir -p "$RUN_ROOT/runs/attempts/${ATTEMPT_ID}" "$RUN_ROOT/runs/approved/${ATTEMPT_ID}"
fi
test ! -e "$OUTPUT_ROOT"
test ! -e "$ARTIFACT"
test ! -e "$APPROVED"
test ! -e "$APPROVED.tmp"

EXECUTION_KIND=(--run-kind "$RUN_MODE")
if [ "$RUN_MODE" = production ]; then
    EXECUTION_KIND+=(--benchmark-gate "$BENCHMARK_GATE" --benchmark-gate-sha256 "$BENCHMARK_GATE_SHA256")
fi

PYTHONPATH="$CODE_ROOT" "$ENV_DIR/bin/python" "$CODE_ROOT/perception/rosie/t14_execute_patchcore.py" \
    --code-root "$CODE_ROOT" --source-commit "$SOURCE_COMMIT" \
    --dataset-dir "$DATASET_DIR" --dataset-artifact "$DATASET_ARTIFACT" \
    --env-dir "$ENV_DIR" --env-artifact "$ENV_ARTIFACT" --env-audit "$ENV_AUDIT" \
    --model-cache "$RUN_ROOT/cache/torch" --part-index "$PART_INDEX" \
    --output-root "$OUTPUT_ROOT" --artifact "$ARTIFACT" --workers 4 \
    --runner-id "brev-${BREV_INSTANCE_ID}-${ATTEMPT_ID}-part-${PART_INDEX}" --hostname "$(hostname)" \
    --runtime-sha256 "$RUNTIME_SHA256" "${EXECUTION_KIND[@]}"

test -s "$ARTIFACT"
if [ "$RUN_MODE" = benchmark ]; then
    PYTHONPATH="$CODE_ROOT" "$ENV_DIR/bin/python" "$CODE_ROOT/perception/brev/t14_benchmark_gate.py" create \
        --benchmark-artifact "$ARTIFACT" --dataset-artifact "$DATASET_ARTIFACT" \
        --env-artifact "$ENV_ARTIFACT" --source-commit "$SOURCE_COMMIT" \
        --runtime-sha256 "$RUNTIME_SHA256" --hourly-rate-usd "$HOURLY_RATE_USD" \
        --hourly-rate-evidence "$HOURLY_RATE_EVIDENCE" \
        --output "$BENCHMARK_GATE"
    echo "T14 benchmark gate SHA-256: $(sha256sum "$BENCHMARK_GATE" | awk '{print $1}')"
fi
printf '%s\n' "$ARTIFACT" > "$APPROVED.tmp"
mv "$APPROVED.tmp" "$APPROVED"
echo "T14 approved Brev $RUN_MODE part artifact: $ARTIFACT"
