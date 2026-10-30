#!/bin/bash
# Help a running Brev render (sdg/brev/hub.sh) from the GB10: render its queues from the END, newest first, and move
# each finished chunk to the render host, whose queue then skips it (render_chunk.sh skips chunks with _DONE.json) and
# whose hub sends it to Cosmos as usual. A chunk is taken only while the render host has not started it, and handed
# over only if it still has not, so both never write the same chunk; the GB10 stops where the two meet.
#
#   queues share the GB10 GPU:  sdg/spark/tail_render.sh fa-r4 "Q:3-0 P:11-0 ..." "Q:7-4 P:23-12 ..."
#   (items in the order to render: write ranges high-low to walk a set backwards)
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
