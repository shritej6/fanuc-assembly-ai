#!/bin/bash
# Run all 24 PatchCore parts sequentially after the matching Brev benchmark gate is measured.
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

: "${ATTEMPT_ID:?set ATTEMPT_ID to a unique immutable production attempt ID}"
: "${BENCHMARK_GATE:?set BENCHMARK_GATE to the approved matching benchmark gate}"
: "${CODE_ROOT:?set CODE_ROOT to the exact clean T14 Git snapshot}"
: "${RUN_ROOT:?set RUN_ROOT to the isolated T14 Brev runtime directory}"
: "${T14_GUARDED_ORCHESTRATION:?invoke through t14_guarded_production.py}"
DEADLINE_EPOCH=$(python3 - "$BENCHMARK_GATE" <<'PY'
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

gate = json.loads(Path(sys.argv[1]).read_text())
deadline = datetime.strptime(gate["budget_deadline_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
print(int(deadline.timestamp()))
PY
)

run_with_budget() {
    local remaining
    remaining=$((DEADLINE_EPOCH - $(date +%s)))
    test "$remaining" -gt 0
    timeout --foreground --kill-after=30s "${remaining}s" "$@"
}

for PART_INDEX in $(seq 0 23); do
    export PART_INDEX RUN_MODE=production
    run_with_budget "$CODE_ROOT/perception/brev/t14_run_patchcore.sh"
done
SUMMARY="$RUN_ROOT/summaries/patchcore-brev-${ATTEMPT_ID}.json"
run_with_budget env PYTHONPATH="$CODE_ROOT" python3 "$CODE_ROOT/perception/rosie/t14_finalize_patchcore.py" \
    --approved-dir "$RUN_ROOT/runs/approved/${ATTEMPT_ID}" \
    --output "$SUMMARY"
echo "T14 measured Brev PatchCore generation: $SUMMARY"
