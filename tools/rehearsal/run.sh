#!/usr/bin/env bash
# Rehearse Codex's CELL runtime head against the laggy real-arm double, then print a short table.
#
#   tools/rehearsal/run.sh [--groot] [--ref origin/feature/T37-saturday-cell-integration]
#
# --groot  also runs the real GR00T server through an SSH tunnel to fa-eval (port 5555), closed on exit.
# Exit code is pytest's: non-zero on a FAIL (a new unsafe behaviour) or an XPASS (a known blocker looks fixed:
# re-review it and drop its xfail marker).
set -euo pipefail

REF="origin/feature/T37-saturday-cell-integration"
GROOT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --groot) GROOT=1; shift ;;
    --ref) REF="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}"
WORK="$(mktemp -d "${SCRATCH%/}/cell-rehearsal.XXXXXX")"
REPORT="$WORK/report.jsonl"
TUNNEL_PID=""

cleanup() {
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  git -C "$HARNESS" worktree remove --force "$WORK/head" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

git -C "$HARNESS" fetch --quiet origin "${REF#origin/}"
HEAD_SHA="$(git -C "$HARNESS" rev-parse FETCH_HEAD)"
git -C "$HARNESS" worktree add --quiet --detach "$WORK/head" "$HEAD_SHA"
mkdir -p "$WORK/head/tools/rehearsal" "$WORK/head/tests/rehearsal"
cp "$HARNESS"/tools/rehearsal/*.py "$WORK/head/tools/rehearsal/"
cp "$HARNESS"/tests/rehearsal/*.py "$WORK/head/tests/rehearsal/"

if [[ "$GROOT" == 1 ]]; then
  ssh -o BatchMode=yes -o ExitOnForwardFailure=yes -N -L 5555:127.0.0.1:5555 fa-eval &
  TUNNEL_PID=$!
  sleep 3
  export REHEARSAL_GROOT_URL="groot://127.0.0.1:5555"
fi

cd "$WORK/head"
echo "Rehearsing Codex head ${HEAD_SHA:0:7} ($REF)"
uv sync --quiet --group dev --python 3.13
set +e
REHEARSAL=1 REHEARSAL_REPORT="$REPORT" uv run pytest -q -p no:cacheprovider tests/rehearsal -rxXf 2>&1 \
  | grep -E "^(XPASS|XFAIL|FAILED|ERROR|SKIPPED)|passed|failed|error" | cut -c1-220
STATUS=${PIPESTATUS[0]}
set -e
echo
[[ -s "$REPORT" ]] && uv run python -m tools.rehearsal.report "$REPORT"
echo
echo "Codex head ${HEAD_SHA:0:7}: pytest exit $STATUS (XFAIL = known blocker still present; XPASS/FAIL need a look)"
exit "$STATUS"
