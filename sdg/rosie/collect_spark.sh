#!/bin/bash
# Copy finished GB10 chunks and specs into the Rosie data folder.
# Authored by Shritej Gunda.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-assembly-data}"
SPARK="${SPARK:-hackathon1@spark1.ai.msoe.edu}"
SSH="ssh -i ${SPARK_KEY:-$HOME/.ssh/av_gb10} -o BatchMode=yes"
REMOTE=fanuc-assembly-data

mapfile -t DONE < <($SSH "$SPARK" "cd $REMOTE/synthetic && ls -d */chunk_*/ | while read d; do [ -f \$d/_DONE.json ] && echo \${d%/}; done")
echo "GB10 finished chunks: ${#DONE[@]}"
for d in "${DONE[@]}"; do
  mkdir -p "$DATA_ROOT/synthetic/$d"
  rsync -rlt --no-perms -e "$SSH" "$SPARK:$REMOTE/synthetic/$d/" "$DATA_ROOT/synthetic/$d/"
done
# Specs only for the sets the GB10 rendered: its folder also holds old smoke-test specs of Rosie's sets.
for s in $(printf "%s\n" "${DONE[@]}" | cut -d/ -f1 | sort -u); do
  mkdir -p "$DATA_ROOT/specs/$s"
  rsync -rlt --no-perms -e "$SSH" "$SPARK:$REMOTE/specs/$s/" "$DATA_ROOT/specs/$s/"
done
rsync -rlt --no-perms -e "$SSH" "$SPARK:$REMOTE/watchdog/sheets/" "$DATA_ROOT/watchdog/sheets/"
echo "copied into $DATA_ROOT/synthetic"
