#!/bin/bash
# Restyle a bundle made by `sdg.cosmos.restyle export` with Cosmos Transfer 2.5 (image-to-image, edge control).
# Runs the Cosmos container on this host's GPU (H100 80 GB, or the GB10 with the Blackwell image); the model loads
# once and every image not yet in out/ runs in one batch. Needs ~/.cache/huggingface/token with the licences accepted.
#   BUNDLE=~/fanuc-assembly-data/cosmos/pilot bash sdg/cosmos/run_transfer.sh
# Writes $BUNDLE/out/<name>.jpg (at the render's size), run.log, timing.txt, cosmos_commit.txt.
set -euo pipefail
BUNDLE=$(cd "${BUNDLE:?set BUNDLE to the exported bundle}" && pwd)
REPO_DIR=${REPO_DIR:-$HOME/cosmos-transfer2.5}
IMAGE=${COSMOS_IMAGE:-cosmos-transfer2.5:2ff49d0}
NUM_STEPS=${NUM_STEPS:-35}

git -C "$REPO_DIR" rev-parse HEAD > "$BUNDLE/cosmos_commit.txt"
mkdir -p "$BUNDLE/out" "$BUNDLE/batch" "$BUNDLE/raw"
rm -f "$BUNDLE"/batch/*.json
for spec in "$BUNDLE"/specs/*.json; do
  name=$(basename "$spec" .json)
  [ -f "$BUNDLE/out/$name.jpg" ] && continue
  sed 's#"\.\./inputs/#"/bundle/inputs/#' "$spec" > "$BUNDLE/batch/$name.json"
done
todo=$(find "$BUNDLE/batch" -name '*.json' | wc -l)
echo "$(date '+%F %T') $todo images to restyle" | tee -a "$BUNDLE/timing.txt"
[ "$todo" -eq 0 ] && exit 0

start=$(date +%s)
docker run --rm --gpus all --ipc=host -v "$REPO_DIR":/workspace -v "$HOME/.cache":/root/.cache -v "$BUNDLE":/bundle \
  -e HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")" "$IMAGE" \
  bash -c 'python examples/inference.py -i /bundle/batch/*.json -o /bundle/raw --num-steps '"$NUM_STEPS" \
  >> "$BUNDLE/run.log" 2>&1 || echo "$(date '+%F %T') inference exited non-zero (see run.log)" | tee -a "$BUNDLE/timing.txt"

# The Cosmos container runs as root; give its files back before converting them.
docker run --rm --entrypoint chown -v "$BUNDLE":/bundle "$IMAGE" -R "$(id -u):$(id -g)" /bundle
for f in "$BUNDLE"/raw/*; do
  name=$(basename "${f%.*}")
  [ -f "$BUNDLE/specs/$name.json" ] || continue
  case "$f" in
    *.mp4) docker run --rm -v "$BUNDLE":/bundle --entrypoint ffmpeg "$IMAGE" -loglevel error -y \
             -i "/bundle/raw/$(basename "$f")" -frames:v 1 -q:v 2 "/bundle/out/$name.jpg" ;;
    *.jpg|*.png) docker run --rm -v "$BUNDLE":/bundle --entrypoint python "$IMAGE" -c \
             "from PIL import Image; Image.open('/bundle/raw/$(basename "$f")').convert('RGB').save('/bundle/out/$name.jpg', quality=95)" ;;
  esac
done
docker run --rm --entrypoint chown -v "$BUNDLE":/bundle "$IMAGE" -R "$(id -u):$(id -g)" /bundle
done_n=$(find "$BUNDLE/out" -name '*.jpg' | wc -l)
echo "$(date '+%F %T') $done_n images in out/, batch took $(( $(date +%s) - start )) s for $todo" | tee -a "$BUNDLE/timing.txt"
