#!/bin/bash
# Help a running Brev render from the GB10 by rendering its queues from the end.
# Authored by Shritej Gunda.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
HOST=$1; shift
DATA_ROOT="${DATA_ROOT:-$HOME/fa-t25-render}"
REMOTE_DATA="${REMOTE_DATA:-fanuc-assembly-data}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
export DATA_ROOT

started_there() {  # set chunk_dir_name
  "${SSH[@]}" -n "$HOST" "test -e $REMOTE_DATA/synthetic/$1/$2 || test -e $REMOTE_DATA/specs/$1/$2.jsonl"
}

queue() {  # "SET:hi-lo ..."
  local item set range c chunk out
  for item in $1; do
    set=${item%%:*} range=${item#*:}
    for c in $(seq "${range%-*}" -1 "${range#*-}"); do
      chunk=chunk_$(printf "%04d" "$c"); out="$DATA_ROOT/synthetic/$set/$chunk"
      if started_there "$set" "$chunk"; then echo "$(date +%T) $set/$chunk started on $HOST: queue met, stop"; return; fi
      "$REPO/sdg/spark/render_chunk.sh" "$set" "$c" > /dev/null 2>&1
      [[ -f "$out/_DONE.json" ]] || { echo "$(date +%T) $set/$chunk render failed here; left to $HOST"; continue; }
      if started_there "$set" "$chunk"; then echo "$(date +%T) $set/$chunk: $HOST started it meanwhile, kept here"; return; fi
      "${SSH[@]}" -n "$HOST" "mkdir -p $REMOTE_DATA/synthetic/$set $REMOTE_DATA/specs/$set" \
        && rsync -a -e "${SSH[*]}" "$DATA_ROOT/specs/$set/$chunk.jsonl" "$HOST:$REMOTE_DATA/specs/$set/" \
        && rsync -a -e "${SSH[*]}" "$out/" "$HOST:$REMOTE_DATA/synthetic/$set/.incoming_$chunk/" \
        && "${SSH[@]}" -n "$HOST" "mv $REMOTE_DATA/synthetic/$set/.incoming_$chunk $REMOTE_DATA/synthetic/$set/$chunk" \
        && echo "$(date +%T) $set/$chunk rendered here, handed to $HOST" \
        || echo "$(date +%T) $set/$chunk hand-over FAILED (kept in $out)"
    done
  done
}

for plan in "$@"; do queue "$plan" & done
wait
echo "$(date +%T) tail render finished"
