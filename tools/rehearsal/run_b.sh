#!/usr/bin/env bash
# Rehearse the DirectGrootRuntime entry point (robot/cell/policy_direct.py) against the laggy real-arm double.
# Authored by Shritej Gunda.
#
#   tools/rehearsal/run_b.sh [--groot] [--ref <git ref>]
#
# --groot  also drives the real GR00T server at 127.0.0.1:5555 through an existing SSH tunnel.
# Exits non-zero when a scenario FAILs. KNOWN = a documented limitation (drop not detected on the cell).
set -eu

REF="origin/feature/T39-direct-groot-runtime"
GROOT=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --groot) GROOT=(--groot groot://127.0.0.1:5555); shift ;;
    --ref) REF="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRATCH="${TMPDIR:-/tmp}"
WORK="$(mktemp -d "${SCRATCH%/}/cell-rehearsal-b.XXXXXX")"
cleanup() {
  git -C "$HARNESS" worktree remove --force "$WORK/head" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

git -C "$HARNESS" fetch --quiet origin "${REF#origin/}"
HEAD_SHA="$(git -C "$HARNESS" rev-parse FETCH_HEAD)"
git -C "$HARNESS" worktree add --quiet --detach "$WORK/head" "$HEAD_SHA"
mkdir -p "$WORK/head/tools/rehearsal"
touch "$WORK/head/tools/__init__.py" "$WORK/head/tools/rehearsal/__init__.py"
cp "$HARNESS"/tools/rehearsal/{laggy_cell.py,direct_b.py} "$WORK/head/tools/rehearsal/"

cd "$WORK/head"
echo "Rehearsing path B head ${HEAD_SHA:0:7} ($REF)"
uv run --quiet --python 3.13 --with pyzmq --with msgpack --with msgpack-numpy --with opencv-python-headless \
  --with requests python -m tools.rehearsal.direct_b ${GROOT[@]+"${GROOT[@]}"} 2>&1 | grep -v "^INFO"
exit "${PIPESTATUS[0]}"
