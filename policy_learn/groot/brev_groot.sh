#!/bin/bash
# GR00T N1.7 fine-tune for the FANUC ER-4iA cell on one 8xH100 box, one command, resume-safe:
#   nohup bash policy_learn/groot/brev_groot.sh > ~/t18_groot.log 2>&1 &
# Before: our repo at $REPO, Agent 4's LeRobot export(s) under $SRC_ROOT (default ~/t12_data/lerobot/*), the HF token
# at ~/.cache/huggingface/token (copied file-to-file; never printed), with access to the gated nvidia/Cosmos-Reason2-2B.
# Every stage appends one line to $STATUS (`<utc> <stage> start|ok|FAILED ...`), so a watcher can report progress.
# START_AT=<stage> reruns from that stage; SKIP="<stage> ..." leaves stages out. Stages: env hf prepare smoke train
# eval export. A rerun of train resumes from the newest checkpoint in $RUNS/main (HF Trainer state + optimizer).
# Throughput gate: after GATE_STEPS steps the measured s/it is projected to MAX_STEPS; if that exceeds
# MAX_TRAIN_HOURS the run restarts once with the largest multiple of SAVE_STEPS that fits (never below MIN_STEPS).
# Off-box copies, every SYNC_SECONDS while training and after export: HF_SYNC_REPO=<user>/<repo> uploads each new
# checkpoint's model files to a PRIVATE Hugging Face repo with the box's token; SYNC_DEST=<host>:<dir> rsyncs full
# checkpoints (optimizer included, so training can resume elsewhere). Without either, the status file says so and the
# main session pulls with: rsync -a <box>:~/groot_runs/ <dest>/
# EVAL_GPU=<i> keeps GPU i out of training and runs GR00T's open-loop eval on the held-out episodes for every new
# checkpoint there (one status line each); Agent 6 can serve the newest checkpoint there with serve_policy.sh.
set -Eeuo pipefail

REPO="${REPO:-$HOME/fanuc-t18}"
HERE="$REPO/policy_learn/groot"
# shellcheck source=pins.env
source "$HERE/pins.env"
GROOT_DIR="${GROOT_DIR:-$HOME/Isaac-GR00T}"
SRC_ROOT="${SRC_ROOT:-$HOME/t12_data/lerobot}"
DATA="${DATA:-$HOME/groot_data}"
RUNS="${RUNS:-$HOME/groot_runs}"
MODELS="${MODELS:-$HOME/groot_models}"
ART="${ART:-$HOME/groot_artifacts}"
STATUS="${STATUS:-$HOME/t18_status.txt}"
PER_GPU_BATCH="${PER_GPU_BATCH:-80}"     # 8 GPUs x 80 = 640, the batch of NVIDIA's LIBERO fine-tunes
MAX_STEPS="${MAX_STEPS:-10000}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
MIN_STEPS="${MIN_STEPS:-3000}"
GATE_STEPS="${GATE_STEPS:-50}"
MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-3.0}"
STATE_DROPOUT="${STATE_DROPOUT:-0.2}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SYNC_SECONDS="${SYNC_SECONDS:-600}"
EVAL_GPU="${EVAL_GPU:-}"
HF_SYNC_REPO="${HF_SYNC_REPO:-}"
SYNC_DEST="${SYNC_DEST:-}"
export GROOT_ACTION_HORIZON="${GROOT_ACTION_HORIZON:-16}" GROOT_EEF_REP="${GROOT_EEF_REP:-relative}"
export HF_HUB_ENABLE_HF_TRANSFER=0 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
[[ -d /usr/local/cuda ]] && export CUDA_HOME=/usr/local/cuda
PY="$GROOT_DIR/.venv/bin/python"
CFG="$HERE/fanuc_er4ia_config.py"
OUT="$RUNS/main"
BASE="$MODELS/GR00T-N1.7-3B"

mark() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
trap 'mark "${STAGE:-setup} FAILED (line $LINENO)"' ERR

START_AT="${START_AT:-env}"
SKIP="${SKIP:-}"
STAGES=(env hf prepare smoke train eval export)
for name in "$START_AT" $SKIP; do
  [[ " ${STAGES[*]} " == *" $name "* ]] || { echo "unknown stage $name in START_AT/SKIP" >&2; exit 2; }
done
started=0
stage() {
  STAGE="$1"; shift
  [[ "$STAGE" == "$START_AT" ]] && started=1
  if (( ! started )); then mark "$STAGE skipped (START_AT=$START_AT)"; return; fi
  if [[ " $SKIP " == *" $STAGE "* ]]; then mark "$STAGE skipped (SKIP)"; return; fi
  mark "$STAGE start"
  "$@"
  mark "$STAGE ok"
}

ALL_GPUS="$(nvidia-smi -L | wc -l)"
train_devices() { seq 0 $((ALL_GPUS - 1)) | grep -vxF -e "${EVAL_GPU:--1}" | paste -sd, -; }
TRAIN_DEVICES="$(train_devices)"
NGPU="$(tr ',' '\n' <<< "$TRAIN_DEVICES" | wc -l)"

stage_env() { GROOT_DIR="$GROOT_DIR" bash "$HERE/install_groot.sh"; }

stage_hf() {
  [[ -s "$HOME/.cache/huggingface/token" ]] || { mark "hf no token file at ~/.cache/huggingface/token"; return 1; }
  local note
  note="$("$PY" - "$BACKBONE_ID" "$BACKBONE_REV" "$BASE_MODEL_ID" "$BASE_MODEL_REV" "$BASE" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download
backbone, backbone_rev, base, base_rev, base_dir = sys.argv[1:]
api = HfApi()
live = api.model_info(backbone).sha            # raises GatedRepoError / 401 without access
snapshot_download(backbone)                    # GR00T loads the backbone by name from the HF cache
snapshot_download(base, revision=base_rev, local_dir=base_dir)
pin = "matches pin" if live == backbone_rev else f"WARNING main moved from pinned {backbone_rev[:8]}"
print(f"backbone {backbone}@{live[:8]} ({pin}); base {base}@{base_rev[:8]} in {base_dir}")
PY
)"
  mark "hf $note"
}

stage_prepare() {
  local srcs=() d
  for d in "$SRC_ROOT"/*/; do [[ -f "$d/meta/info.json" ]] && srcs+=(--src "${d%/}"); done
  (( ${#srcs[@]} )) || { mark "prepare no LeRobot dataset under $SRC_ROOT"; return 1; }
  (cd "$REPO" && "$PY" -m policy_learn.groot.prepare_dataset "${srcs[@]}" --out "$DATA" > "$RUNS/prepare.log")
  for d in "$DATA"/train/*/ "$DATA"/val/*/; do
    [[ -d "$d" ]] || continue
    (cd "$GROOT_DIR" && "$PY" -m gr00t.data.stats --dataset-path "${d%/}" --embodiment-tag NEW_EMBODIMENT \
      --modality-config-path "$CFG" > /dev/null)
  done
  mark "prepare $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); print(" ".join(f"{x["name"]}:train={x["train_episodes"]}ep/{x["train_frames"]}fr,val={x["val_episodes"]}ep" for x in r))' "$DATA/prepare_report.json")"
}

train_paths() { local d out=(); for d in "$DATA"/train/*/; do out+=("${d%/}"); done; (IFS=:; echo "${out[*]}"); }

launch() {   # $1 output dir, $2 max steps, $3 save steps; extra args are passed on
  local out="$1" steps="$2" save="$3"; shift 3
  local args=(gr00t/experiment/launch_finetune.py --base-model-path "$BASE" --dataset-path "$(train_paths)"
    --embodiment-tag NEW_EMBODIMENT --modality-config-path "$CFG" --num-gpus "$NGPU" --output-dir "$out"
    --max-steps "$steps" --save-steps "$save" --save-total-limit 3 --global-batch-size $((PER_GPU_BATCH * NGPU))
    --learning-rate "$LEARNING_RATE" --state-dropout-prob "$STATE_DROPOUT" --dataloader-num-workers 8
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 "$@")
  compgen -G "$out/checkpoint-*/trainer_state.json" > /dev/null && args+=(--resume-from-checkpoint)
  cd "$GROOT_DIR"
  if (( NGPU == 1 )); then
    CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" "$PY" "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="$TRAIN_DEVICES" "$GROOT_DIR/.venv/bin/torchrun" --nproc_per_node="$NGPU" \
      --master_port="${MASTER_PORT:-29500}" "${args[@]}"
  fi
}

stage_smoke() {   # 4 steps + one save on the training GPUs: data, model, multi-GPU and checkpoint writing all work
  rm -rf "$RUNS/smoke"
  launch "$RUNS/smoke" 4 4 > "$RUNS/smoke.log" 2>&1
  local ckpt="$RUNS/smoke/checkpoint-4"
  [[ -f "$ckpt/trainer_state.json" ]] || { mark "smoke no checkpoint written"; return 1; }
  mark "smoke ok on $NGPU GPUs ($TRAIN_DEVICES), checkpoint $(du -sh "$ckpt" | cut -f1)"
  rm -rf "$RUNS/smoke"
}

# Prints "<seconds per step> <total steps>" from the HF Trainer progress bar once GATE_STEPS steps are logged.
measured_speed() {
  python3 - "$1" "$GATE_STEPS" <<'PY'
import re, sys
text = open(sys.argv[1], errors="replace").read().replace("\r", "\n")
bars = re.findall(r"\| *(\d+)/(\d+) \[[^\]]*?([\d.]+)(s/it|it/s)\]", text)
if bars:
    total = bars[0][1]
    done, _, rate, unit = [b for b in bars if b[1] == total][-1]
    if int(done) >= int(sys.argv[2]):
        print(f"{float(rate) if unit == 's/it' else 1 / float(rate):.3f} {total}")
PY
}

kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill "$1" 2>/dev/null || true; }

latest_ckpt() { { ls -d "$OUT"/checkpoint-*/ 2>/dev/null || true; } | sort -t- -k2 -n | tail -1 | sed 's:/$::'; }

sync_ckpt() {    # $1 checkpoint dir; failures are reported in $STATUS and retried on the next round
  local ckpt="$1" name; name="$(basename "$1")"
  if [[ -n "$HF_SYNC_REPO" ]]; then
    if "$PY" - "$HF_SYNC_REPO" "$ckpt" "$name" <<'PY' > /dev/null 2>> "$RUNS/sync.err"
import sys
from huggingface_hub import HfApi
repo, folder, name = sys.argv[1:]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=name,
                  ignore_patterns=["optimizer.pt", "*.pth", "scheduler.pt"])
PY
    then mark "sync $name ok -> hf:$HF_SYNC_REPO (model files)"; else mark "sync $name FAILED -> hf (see $RUNS/sync.err)"; fi
  fi
  if [[ -n "$SYNC_DEST" ]]; then
    if rsync -a --partial "$ckpt" "$STATUS" "$SYNC_DEST/" 2>> "$RUNS/sync.err"; then
      mark "sync $name ok -> $SYNC_DEST"; else mark "sync $name FAILED -> $SYNC_DEST (see $RUNS/sync.err)"; fi
  fi
  [[ -n "$HF_SYNC_REPO$SYNC_DEST" ]] || mark "sync $name NOT off-box (no HF_SYNC_REPO/SYNC_DEST): pull $OUT now"
}

openloop() {     # $1 checkpoint dir, $2 GPU index; one status line with GR00T's unnormalised action MSE / MAE
  local ckpt="$1" val n ids log="$RUNS/openloop_$(basename "$1").log" mse mae
  for val in "$DATA"/val/*/; do
    [[ -d "$val" ]] || { mark "eval $(basename "$ckpt") no held-out episodes"; return 0; }
    n="$(wc -l < "$val/meta/episodes.jsonl")"; ids="$(seq 0 $(( (n < 10 ? n : 10) - 1 )) | paste -sd' ')"
    # shellcheck disable=SC2086
    if ! (cd "$GROOT_DIR" && CUDA_VISIBLE_DEVICES="$2" "$PY" gr00t/eval/open_loop_eval.py --dataset-path "${val%/}" \
      --embodiment-tag NEW_EMBODIMENT --model-path "$ckpt" --traj-ids $ids --steps 2000 \
      --execution-horizon "$GROOT_ACTION_HORIZON" --save-plot-path "$RUNS/plots/$(basename "$ckpt")") > "$log" 2>&1
    then
      mark "eval $(basename "$ckpt") open-loop FAILED (see $log)"; return 1
    fi
    mse="$(grep -o 'Average MSE across all trajs: [^ ]*' "$log" | tail -1 | awk '{print $NF}')"
    mae="$(grep -o 'Average MAE across all trajs: [^ ]*' "$log" | tail -1 | awk '{print $NF}')"
    mark "eval $(basename "$ckpt") $(basename "$val") open-loop on held-out eps [$ids]: MSE=${mse:-?} MAE=${mae:-?}"
  done
}

watch_loop() {   # runs beside training: sync (and with EVAL_GPU, open-loop eval) every new checkpoint
  local seen="" ckpt
  while sleep "$SYNC_SECONDS"; do
    ckpt="$(latest_ckpt)"
    [[ -n "$ckpt" && "$ckpt" != "$seen" && -f "$ckpt/trainer_state.json" ]] || continue
    sleep 30   # let the trainer finish writing the folder
    sync_ckpt "$ckpt"
    if [[ -n "$EVAL_GPU" ]]; then openloop "$ckpt" "$EVAL_GPU" || true; fi   # failure already in $STATUS
    seen="$ckpt"
  done
}

# Starts training, applies the throughput gate once; returns the trainer's exit status.
run_gated() {
  local log="$RUNS/train.log" pid speed=""
  launch "$OUT" "$MAX_STEPS" "$SAVE_STEPS" >> "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null && [[ -z "$speed" ]]; do sleep 30; speed="$(measured_speed "$log")"; done
  [[ -z "$speed" ]] && { mark "train throughput not measured (ended before $GATE_STEPS steps)"; wait "$pid"; return; }
  local sec="${speed% *}" fit
  fit="$(python3 -c "import sys; s,h,k=map(float,sys.argv[1:]); print(int(h*3600/s//k*k))" "$sec" "$MAX_TRAIN_HOURS" "$SAVE_STEPS")"
  mark "train throughput $sec s/step x $MAX_STEPS steps = $(python3 -c "print(f'{$sec*$MAX_STEPS/3600:.2f}')") h (limit $MAX_TRAIN_HOURS h)"
  if (( fit >= MAX_STEPS )); then wait "$pid"; return; fi
  kill_tree "$pid"; wait "$pid" || true
  if (( fit < MIN_STEPS )); then mark "train $fit steps fit in $MAX_TRAIN_HOURS h < MIN_STEPS $MIN_STEPS; stopped"; return 1; fi
  if compgen -G "$OUT/checkpoint-*" > /dev/null; then mark "train over limit but checkpoints exist; not restarted"; return 1; fi
  rm -rf "$OUT"; MAX_STEPS="$fit"
  mark "train restarted with MAX_STEPS=$MAX_STEPS to fit $MAX_TRAIN_HOURS h"
  launch "$OUT" "$MAX_STEPS" "$SAVE_STEPS" >> "$log" 2>&1
}

stage_train() {
  mkdir -p "$OUT"
  MAX_STEPS=$(( (MAX_STEPS + SAVE_STEPS - 1) / SAVE_STEPS * SAVE_STEPS ))   # the last step is always saved
  watch_loop & local watcher=$!
  local rc=0
  run_gated || rc=$?
  kill_tree "$watcher"
  local ckpt; ckpt="$(latest_ckpt)"
  [[ -n "$ckpt" ]] && sync_ckpt "$ckpt"
  return "$rc"
}

stage_eval() { openloop "$(latest_ckpt)" "${EVAL_GPU:-0}"; }

stage_export() {
  local ckpt final="$ART/groot_final"; ckpt="$(latest_ckpt)"
  rm -rf "$final"; mkdir -p "$final"
  rsync -a --exclude optimizer.pt --exclude '*.pth' --exclude scheduler.pt "$ckpt/" "$final/"
  cp "$DATA/prepare_report.json" "$final/t18_prepare_report.json"
  printf 'groot_commit=%s\nbase=%s@%s\ncheckpoint=%s\naction_horizon=%s\neef_rep=%s\nfps=5\n' "$GROOT_COMMIT" \
    "$BASE_MODEL_ID" "$BASE_MODEL_REV" "$(basename "$ckpt")" "$GROOT_ACTION_HORIZON" "$GROOT_EEF_REP" > "$final/t18_manifest.txt"
  cp "$STATUS" "$final/t18_status.txt"
  sync_ckpt "$final"
  mark "export $(basename "$ckpt") -> $final ($(du -sh "$final" | cut -f1))"
}

mkdir -p "$RUNS" "$MODELS" "$ART"
mark "begin repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || cat "$REPO/.t18-code-commit" 2>/dev/null || echo unknown) gpus=$ALL_GPUS train_gpus=$TRAIN_DEVICES eval_gpu=${EVAL_GPU:-none} batch=$((PER_GPU_BATCH * NGPU)) steps=$MAX_STEPS horizon=$GROOT_ACTION_HORIZON rep=$GROOT_EEF_REP sync=${HF_SYNC_REPO:+hf:$HF_SYNC_REPO}${SYNC_DEST:+ rsync:$SYNC_DEST}"
stage env stage_env
stage hf stage_hf
stage prepare stage_prepare
stage smoke stage_smoke
stage train stage_train
stage eval stage_eval
stage export stage_export
mark "ALL_DONE"
