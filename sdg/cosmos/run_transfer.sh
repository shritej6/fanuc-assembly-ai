#!/bin/bash
# Restyle an exported bundle with Cosmos Transfer 2.5.
# Authored by Shritej Gunda.
set -euo pipefail
BUNDLE=$(cd "${BUNDLE:?set BUNDLE to the exported bundle}" && pwd)
REPO_DIR=${REPO_DIR:-$HOME/cosmos-transfer2.5}
RUNNER=${RUNNER:-docker}
IMAGE=${COSMOS_IMAGE:-cosmos-transfer2.5:2ff49d0}
NUM_STEPS=${NUM_STEPS:-35}
HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
export HF_TOKEN

if [ "$RUNNER" = docker ]; then INSIDE=/bundle; else INSIDE=$BUNDLE; fi
git -C "$REPO_DIR" rev-parse HEAD > "$BUNDLE/cosmos_commit.txt"
mkdir -p "$BUNDLE/out" "$BUNDLE/batch" "$BUNDLE/raw"
rm -f "$BUNDLE"/batch/*.json
for spec in "$BUNDLE"/specs/*.json; do
  name=$(basename "$spec" .json)
  [ -f "$BUNDLE/out/$name.jpg" ] && continue
  sed "s#\"\.\./inputs/#\"$INSIDE/inputs/#" "$spec" > "$BUNDLE/batch/$name.json"
done
todo=$(find "$BUNDLE/batch" -name '*.json' | wc -l)
echo "$(date '+%F %T') $todo images to restyle" | tee -a "$BUNDLE/timing.txt"
[ "$todo" -eq 0 ] && exit 0

start=$(date +%s)
if [ "$RUNNER" = docker ]; then
  docker run --rm --gpus all --ipc=host -v "$REPO_DIR":/workspace -v "$HOME/.cache":/root/.cache -v "$BUNDLE":/bundle \
    -e HF_TOKEN "$IMAGE" bash -c "python examples/inference.py -i /bundle/batch/*.json -o /bundle/raw --num-steps $NUM_STEPS" \
    >> "$BUNDLE/run.log" 2>&1 || echo "$(date '+%F %T') inference exited non-zero (see run.log)" | tee -a "$BUNDLE/timing.txt"
  # The container runs as root; give its files back.
  docker run --rm --entrypoint chown -v "$BUNDLE":/bundle "$IMAGE" -R "$(id -u):$(id -g)" /bundle
else
  (cd "$REPO_DIR" && .venv/bin/python examples/inference.py -i "$BUNDLE"/batch/*.json -o "$BUNDLE/raw" \
     --num-steps "$NUM_STEPS") >> "$BUNDLE/run.log" 2>&1 \
    || echo "$(date '+%F %T') inference exited non-zero (see run.log)" | tee -a "$BUNDLE/timing.txt"
fi

# Cosmos writes <name>.jpg for an image-to-image sample; keep only the samples of this bundle.
for f in "$BUNDLE"/raw/*.jpg; do
  name=$(basename "$f" .jpg)
  [ -f "$BUNDLE/specs/$name.json" ] && cp "$f" "$BUNDLE/out/$name.jpg"
done
done_n=$(find "$BUNDLE/out" -name '*.jpg' | wc -l)
echo "$(date '+%F %T') $done_n images in out/, batch took $(( $(date +%s) - start )) s for $todo" | tee -a "$BUNDLE/timing.txt"
