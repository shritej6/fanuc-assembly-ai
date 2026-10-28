#!/bin/bash
# Render -> Cosmos pipeline hub, run on the render host. Renders with sdg/brev/render_queue.sh and, as each chunk
# finishes, exports a Cosmos share of it (sdg.cosmos.restyle export --chunks), sends the bundle to a Cosmos host
# (sdg/brev/cosmos_worker.sh, round robin over ~/fa-cosmos-hosts.txt, one ssh target per line; hosts can be added
# while it runs), fetches finished bundles and imports them (--min-fscore 0.85: edge F gate + per-part composite).
# Writes $DATA_ROOT/PIPELINE_DONE when every chunk is rendered and every sent bundle is imported.
#
#   nohup sdg/brev/hub.sh "N:0-19 O:0-23 ..." "R:0-19 P:0-23 ..." > ~/hub.log 2>&1 &
#   (arguments go to render_queue.sh: one render plan per GPU)
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-assembly-data}"
PY="${HOST_VENV:-$HOME/fa-venv}/bin/python"
SHARE="${COSMOS_SHARE:-0.55}"
HOSTS="$HOME/fa-cosmos-hosts.txt"
BUNDLES="$DATA_ROOT/cosmos/stream"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
export DATA_ROOT
mkdir -p "$BUNDLES"

"$REPO/sdg/brev/render_queue.sh" "$@" > "$DATA_ROOT/render_queue.log" 2>&1 &

send_finished_chunks() {
  local done_file chunk set name host
  mapfile -t hosts < <(grep -v '^\s*$' "$HOSTS" 2>/dev/null)
  [[ ${#hosts[@]} -eq 0 ]] && return
  for done_file in "$DATA_ROOT"/synthetic/*/chunk_*/_DONE.json; do
    chunk=$(dirname "$done_file"); set=$(basename "$(dirname "$chunk")")
    [[ -f "$chunk/_COSMOS_SENT" ]] && continue
    name="${set}__$(basename "$chunk")"
    rm -rf "${BUNDLES:?}/$name"
    (cd "$REPO" && "$PY" -m sdg.cosmos.restyle export --sets "$set" --chunks "$(basename "$chunk")" \
      --fraction "$SHARE" --data-root "$DATA_ROOT" --out "$BUNDLES/$name") > /dev/null || { echo "$(date +%T) export $name failed"; continue; }
    host=${hosts[$(( $(cat "$BUNDLES/.next" 2>/dev/null || echo 0) % ${#hosts[@]} ))]}
    if rsync -a -e "${SSH[*]}" "$BUNDLES/$name" "$host:fa-cosmos/incoming/" \
       && "${SSH[@]}" -n "$host" "mv ~/fa-cosmos/incoming/$name ~/fa-cosmos/inbox/$name"; then
      echo "$host" > "$chunk/_COSMOS_SENT"
      echo $(( $(cat "$BUNDLES/.next" 2>/dev/null || echo 0) + 1 )) > "$BUNDLES/.next"
      echo "$(date +%T) sent $name to $host"
    else
      echo "$(date +%T) sending $name to $host failed; retry next round"
    fi
  done
}

import_finished_bundles() {
  local host b name
  while read -r host; do
    [[ -z "$host" ]] && continue
    for name in $("${SSH[@]}" -n "$host" 'ls ~/fa-cosmos/done 2>/dev/null'); do
      [[ -f "$BUNDLES/$name/_IMPORTED.json" ]] && continue
      rsync -a -e "${SSH[*]}" "$host:fa-cosmos/done/$name/" "$BUNDLES/$name/" < /dev/null || continue
      b="$BUNDLES/$name"
      if (cd "$REPO" && "$PY" -m sdg.cosmos.restyle import "$b" --min-fscore 0.85 --data-root "$DATA_ROOT") > "$b/_import.tmp"; then
        mv "$b/_import.tmp" "$b/_IMPORTED.json" && echo "$(date +%T) imported $name $(cat "$b/_IMPORTED.json")"
      else
        echo "$(date +%T) import $name failed (see $b/_import.tmp)"
      fi
    done
  done < <(grep -v '^\s*$' "$HOSTS" 2>/dev/null)
}

while true; do
  send_finished_chunks
  import_finished_bundles
  sent=$(find "$DATA_ROOT/synthetic" -name _COSMOS_SENT | wc -l)
  imported=$(find "$BUNDLES" -name _IMPORTED.json | wc -l)
  echo "$(date '+%F %T') sent $sent imported $imported" > "$DATA_ROOT/hub_status.txt"
  if [[ -f "$DATA_ROOT/QUEUE_DONE" && $sent -eq $imported && $sent -gt 0 ]]; then
    date +%T > "$DATA_ROOT/PIPELINE_DONE"; echo "$(date +%T) pipeline done: $imported bundles"; exit 0
  fi
  sleep 60
done
