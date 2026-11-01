#!/bin/bash
# One-command, resume-safe ACT and SmolVLA sim comparison baselines for GR00T N1.7 on one GPU box.
# Authored by Shritej Gunda.
#   nohup bash policy_learn/baselines/brev_baselines.sh > ~/baselines.log 2>&1 &
# Trains on the same data and split as GR00T; these baselines are never deployed to the real cell.
# Stages: env hf prepare view smoke train eval export. START_AT and SKIP work as in brev_groot.sh.
set -Eeuo pipefail

REPO="${REPO:-$HOME/fanuc-t36}"
HERE="$REPO/policy_learn/baselines"
# shellcheck source=pins.env
source "$HERE/pins.env"
VENV="${VENV:-$HOME/lerobot-venv}"
SRC_ROOT="${SRC_ROOT:-$HOME/t12_data/lerobot}"
PREP="${PREP:-$HOME/groot_data}"          # prepare_dataset.py output (identical to the GR00T box's)
DATA="${DATA:-$HOME/baseline_data}"       # lerobot_view.py output: train/ and val/
RUNS="${RUNS:-$HOME/baseline_runs}"
MODELS="${MODELS:-$HOME/baseline_models}"
ART="${ART:-$HOME/baseline_artifacts}"
STATUS="${STATUS:-$HOME/t36_status.txt}"
POLICIES="${POLICIES:-act smolvla}"
MAX_TRAIN_HOURS="${MAX_TRAIN_HOURS:-3.0}"
GATE_STEPS="${GATE_STEPS:-60}"
MIN_STEPS="${MIN_STEPS:-5000}"
NUM_WORKERS="${NUM_WORKERS:-16}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
DEVICE="${DEVICE:-cuda}"                   # cpu only for a local plumbing run of the driver
SYNC_SECONDS="${SYNC_SECONDS:-900}"
HF_SYNC_REPO="${HF_SYNC_REPO-shritej81632/fanuc-baselines}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
# Training configs (estimate, not tuned on this data). Both predict 16 rows = 3.2 s at the data's 5 Hz,
# GR00T's action horizon, and the evaluator plays all 16 before the next observation (one observation per chunk).
ACT_BATCH="${ACT_BATCH:-32}"; ACT_MAX_STEPS="${ACT_MAX_STEPS:-80000}"; ACT_SAVE="${ACT_SAVE:-5000}"
SMOLVLA_BATCH="${SMOLVLA_BATCH:-64}"; SMOLVLA_MAX_STEPS="${SMOLVLA_MAX_STEPS:-20000}"; SMOLVLA_SAVE="${SMOLVLA_SAVE:-2500}"
# Instructions are 53-55 SmolVLM2 tokens on the test export, so the base's 48 would cut off the
# target numbers; the smoke stage refuses to train if any instruction exceeds SMOLVLA_TOKENS.
SMOLVLA_TOKENS="${SMOLVLA_TOKENS:-96}"
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled HF_HUB_ENABLE_HF_TRANSFER=0
PY="$VENV/bin/python"

mark() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
trap 'mark "${STAGE:-setup} FAILED (line $LINENO)"' ERR

START_AT="${START_AT:-env}"
SKIP="${SKIP:-}"
STAGES=(env hf prepare view smoke train eval export)
for name in "$START_AT" $SKIP; do
  [[ " ${STAGES[*]} " == *" $name "* ]] || { echo "unknown stage $name in START_AT/SKIP" >&2; exit 2; }
done
for p in $POLICIES; do [[ "$p" == act || "$p" == smolvla ]] || { echo "unknown policy $p" >&2; exit 2; }; done
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

stage_env() {
  command -v uv > /dev/null || { mark "env needs uv (pip install --user uv), not installed"; return 1; }
  [[ -x "$PY" ]] || uv venv -q -p "$PYTHON_VERSION" "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install -q "lerobot[smolvla]==$LEROBOT_VERSION" "$AV_SPEC" pyzmq msgpack msgpack-numpy \
    pandas pyarrow
  mark "env $("$PY" -c 'import lerobot, torch, av; print(f"lerobot {lerobot.__version__} torch {torch.__version__} cuda {torch.cuda.is_available()} av {av.__version__}")')"
}

stage_hf() {
  local note
  note="$("$PY" - "$SMOLVLA_BASE_ID" "$SMOLVLA_BASE_REV" "$MODELS/smolvla_base" "$SMOLVLM_ID" "$SMOLVLM_REV" <<'PY'
import sys
from huggingface_hub import HfApi, snapshot_download
base, base_rev, base_dir, vlm, vlm_rev = sys.argv[1:]
snapshot_download(base, revision=base_rev, local_dir=base_dir)
live = HfApi().model_info(vlm).sha
snapshot_download(vlm)              # SmolVLA loads the VLM by name (main) from the HF cache
pin = "matches pin" if live == vlm_rev else f"WARNING main moved from pinned {vlm_rev[:8]}"
print(f"smolvla base {base}@{base_rev[:8]} in {base_dir}; vlm {vlm}@{live[:8]} ({pin})")
PY
)"
  mark "hf $note"
}

stage_prepare() {   # same prepare step as GR00T: the same split GR00T trains on
  local srcs=() d
  for d in "$SRC_ROOT"/*/; do [[ -f "$d/meta/info.json" ]] && srcs+=(--src "${d%/}"); done
  (( ${#srcs[@]} )) || { mark "prepare no LeRobot dataset under $SRC_ROOT"; return 1; }
  mkdir -p "$RUNS"
  (cd "$REPO" && "$PY" -m policy_learn.groot.prepare_dataset "${srcs[@]}" --out "$PREP" > "$RUNS/prepare.log")
  mark "prepare $(cat "$PREP/prepare_report.json" | tr -d '\n ' | head -c 600)"
}

stage_view() {
  (cd "$REPO" && "$PY" -m policy_learn.baselines.lerobot_view --prepared "$PREP" --out "$DATA" > "$RUNS/view.log")
  mark "view $(tr -d '\n ' < "$DATA/view_report.json" | head -c 600)"
}

policy_args() {   # $1 policy -> the LeRobot train arguments that define it
  case "$1" in
    act) echo "--policy.type=act --policy.chunk_size=16 --policy.n_action_steps=16 --batch_size=$ACT_BATCH" ;;
    smolvla) echo "--policy.path=$MODELS/smolvla_base --policy.chunk_size=16 --policy.n_action_steps=16" \
      "--policy.tokenizer_max_length=$SMOLVLA_TOKENS --batch_size=$SMOLVLA_BATCH" ;;
  esac
}

launch() {   # $1 policy, $2 output dir, $3 steps, $4 save freq, $5 log freq; extra args are passed on
  local p="$1" out="$2" steps="$3" save="$4" logf="$5"; shift 5
  local offline=0; [[ "$p" == smolvla ]] && offline=1
  if [[ -f "$out/checkpoints/last/pretrained_model/train_config.json" ]]; then
    HF_HUB_OFFLINE=$offline "$PY" -m lerobot.scripts.train --resume=true \
      --config_path="$out/checkpoints/last/pretrained_model/train_config.json" "$@"
    return
  fi
  # shellcheck disable=SC2046
  HF_HUB_OFFLINE=$offline "$PY" -m lerobot.scripts.train --dataset.repo_id=local/fanuc_baselines_train \
    --dataset.root="$DATA/train" --dataset.video_backend="$VIDEO_BACKEND" --dataset.image_transforms.enable=true \
    $(policy_args "$p") --policy.device="$DEVICE" --policy.push_to_hub=false --output_dir="$out" --job_name="t36_$p" \
    --steps="$steps" --save_freq="$save" --log_freq="$logf" --num_workers="$NUM_WORKERS" --wandb.enable=false \
    --seed=1000 "$@"
}

# Prints the seconds per step of the last logged window (updt_s + data_s) from a LeRobot train log.
measured_speed() {
  python3 - "$1" <<'PY'
import re, sys
hits = re.findall(r"updt_s:([\d.]+) data_s:([\d.]+)", open(sys.argv[1], errors="replace").read())
if hits:
    print(f"{float(hits[-1][0]) + float(hits[-1][1]):.4f}")
PY
}

stage_smoke() {   # per policy: GATE_STEPS steps + one save with the real batch; sizes the real run
  local p log sec fit max save longest
  if [[ " $POLICIES " == *" smolvla "* ]]; then
    longest="$(HF_HUB_OFFLINE=1 "$PY" - "$SMOLVLM_ID" "$DATA/train/meta/tasks.jsonl" "$DATA/val/meta/tasks.jsonl" <<'PY'
import json, sys
from pathlib import Path
from transformers import AutoProcessor
tok = AutoProcessor.from_pretrained(sys.argv[1]).tokenizer
texts = [json.loads(line)["task"] for f in sys.argv[2:] if Path(f).is_file() for line in open(f) if line.strip()]
print(max(len(tok(t + "\n")["input_ids"]) for t in texts))   # SmolVLA appends "\n" before tokenizing
PY
)"
    (( longest <= SMOLVLA_TOKENS )) || { mark "smoke smolvla: longest instruction $longest tokens > SMOLVLA_TOKENS $SMOLVLA_TOKENS"; return 1; }
    mark "smoke smolvla longest instruction $longest tokens (limit $SMOLVLA_TOKENS)"
  fi
  for p in $POLICIES; do
    rm -rf "$RUNS/smoke_$p"; log="$RUNS/smoke_$p.log"
    launch "$p" "$RUNS/smoke_$p" "$GATE_STEPS" "$GATE_STEPS" $(( GATE_STEPS / 3 > 0 ? GATE_STEPS / 3 : 1 )) > "$log" 2>&1 \
      || { mark "smoke $p FAILED (see $log)"; return 1; }
    [[ -f "$RUNS/smoke_$p/checkpoints/last/pretrained_model/model.safetensors" ]] \
      || { mark "smoke $p no checkpoint written"; return 1; }
    sec="$(measured_speed "$log")"; [[ -n "$sec" ]] || { mark "smoke $p speed not in log"; return 1; }
    if [[ "$p" == act ]]; then max=$ACT_MAX_STEPS save=$ACT_SAVE; else max=$SMOLVLA_MAX_STEPS save=$SMOLVLA_SAVE; fi
    fit="$(python3 -c "import sys; s,h,k,m=map(float,sys.argv[1:]); print(int(min(m, h*3600/(s*1.1)//k*k)))" \
      "$sec" "$MAX_TRAIN_HOURS" "$save" "$max")"
    (( fit >= MIN_STEPS )) || { mark "smoke $p: $sec s/step fits only $fit steps in $MAX_TRAIN_HOURS h < $MIN_STEPS"; return 1; }
    echo "$fit" > "$RUNS/steps_$p"
    mark "smoke $p ok: $sec s/step measured, train $fit steps = $(python3 -c "print(f'{$sec*$fit/3600:.2f}')") h (+10% margin; limit $MAX_TRAIN_HOURS h), loss $(grep -o 'loss:[0-9.]*' "$log" | tail -1)"
    rm -rf "$RUNS/smoke_$p"
  done
}

sync_ckpt() {   # $1 policy, $2 pretrained_model dir, $3 name in the repo
  [[ -n "$HF_SYNC_REPO" ]] || { mark "sync $1/$3 NOT off-box (HF_SYNC_REPO empty): pull $2 now"; return 0; }
  if "$PY" - "$HF_SYNC_REPO" "$2" "$1/$3" <<'PY' > /dev/null 2>> "$RUNS/sync.err"
import sys
from huggingface_hub import HfApi
repo, folder, dest = sys.argv[1:]
api = HfApi()
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(repo_id=repo, folder_path=folder, path_in_repo=dest)
PY
  then mark "sync $1/$3 ok -> hf:$HF_SYNC_REPO (private)"; else mark "sync $1/$3 FAILED (see $RUNS/sync.err)"; fi
}

latest_step() { ls -d "$RUNS/$1"/checkpoints/[0-9]*/ 2>/dev/null | sort | tail -1 | sed 's:/$::'; }

watch_loop() {   # $1 policy: sync every new checkpoint while training
  local seen="" ckpt
  while sleep "$SYNC_SECONDS"; do
    ckpt="$(latest_step "$1")"
    [[ -n "$ckpt" && "$ckpt" != "$seen" && -f "$ckpt/pretrained_model/model.safetensors" ]] || continue
    sleep 30; sync_ckpt "$1" "$ckpt/pretrained_model" "$(basename "$ckpt")"; seen="$ckpt"
  done
}

kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill "$1" 2>/dev/null || true; }

stage_train() {
  local p steps save watcher rc=0
  for p in $POLICIES; do
    [[ -s "$RUNS/steps_$p" ]] || { mark "train $p has no sized step count (run the smoke stage)"; return 1; }
    steps="$(cat "$RUNS/steps_$p")"; save=$ACT_SAVE; [[ "$p" == smolvla ]] && save=$SMOLVLA_SAVE
    local extra=(); [[ "$p" == smolvla ]] && extra=(--policy.scheduler_decay_steps="$steps")
    mark "train $p start: $steps steps, save every $save"
    watch_loop "$p" & watcher=$!
    launch "$p" "$RUNS/$p" "$steps" "$save" 200 ${extra[@]+"${extra[@]}"} >> "$RUNS/train_$p.log" 2>&1 || rc=$?
    kill_tree "$watcher"
    (( rc == 0 )) || { mark "train $p FAILED rc=$rc (see $RUNS/train_$p.log)"; return "$rc"; }
    python3 - "$RUNS/train_$p.log" "$RUNS/loss_$p.tsv" <<'PY'
import re, sys
rows = re.findall(r"step:(\S+) smpl:\S+ ep:\S+ epch:([\d.]+) loss:([\d.]+)", open(sys.argv[1]).read())
open(sys.argv[2], "w").write("step\tepoch\tloss\n" + "".join(f"{s}\t{e}\t{l}\n" for s, e, l in rows))
PY
    mark "train $p done: loss curve $RUNS/loss_$p.tsv, last $(tail -1 "$RUNS/loss_$p.tsv" | tr '\t' ' ')"
  done
}

stage_eval() {   # open-loop on the held-out episodes, through GrootPolicy + guard, like the sim evaluator
  local p port=5561 pid
  [[ -d "$DATA/val" ]] || { mark "eval no held-out episodes"; return 0; }
  for p in $POLICIES; do
    (cd "$REPO" && HF_HUB_OFFLINE=1 "$PY" -m policy_learn.baselines.policy_server --port $port --device "$DEVICE" \
      --checkpoint "$RUNS/$p/checkpoints/last/pretrained_model" > "$RUNS/serve_$p.log" 2>&1) & pid=$!
    for _ in $(seq 60); do grep -q "serving on" "$RUNS/serve_$p.log" && break; sleep 5; done
    (cd "$REPO" && "$PY" -m policy_learn.baselines.openloop --policy "groot://127.0.0.1:$port" --dataset "$DATA/val" \
      --episodes "$EVAL_EPISODES" --out "$RUNS/openloop_$p.json" > "$RUNS/openloop_$p.log" 2>&1) \
      || { kill_tree "$pid"; mark "eval $p FAILED (see $RUNS/openloop_$p.log)"; return 1; }
    kill_tree "$pid"
    mark "eval $p open-loop $("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); x=r["xyz_err_mm"]; print(f"eps={r["episodes"]} queries={r["queries"]} held={r["queries_held"]} xyz_mean={x["mean"]:.1f}mm rot_mean={r["rot_err_deg"]["mean"]:.2f}deg grip={r["gripper_agreement"]:.3f}" if x else f"eps={r["episodes"]} ALL {r["queries"]} queries held by the guard")' "$RUNS/openloop_$p.json")"
  done
}

stage_export() {
  local p final
  for p in $POLICIES; do
    final="$ART/${p}_final"; rm -rf "$final"; mkdir -p "$final"
    cp -r "$RUNS/$p/checkpoints/last/pretrained_model/." "$final/"
    cp "$PREP/prepare_report.json" "$DATA/view_report.json" "$final/"
    cp "$RUNS/loss_$p.tsv" "$final/" 2>/dev/null || true
    cp "$RUNS/openloop_$p.json" "$final/" 2>/dev/null || true
    printf 'lerobot=%s\npolicy=%s\nsteps=%s\nfps=5\nrows_per_chunk=16\nsmolvla_base=%s@%s\n' "$LEROBOT_VERSION" "$p" \
      "$(cat "$RUNS/steps_$p" 2>/dev/null || echo unknown)" "$SMOLVLA_BASE_ID" "$SMOLVLA_BASE_REV" > "$final/t36_manifest.txt"
    cp "$STATUS" "$final/t36_status.txt"
    sync_ckpt "$p" "$final" final
    mark "export $p -> $final ($(du -sh "$final" | cut -f1))"
  done
}

mkdir -p "$RUNS" "$MODELS" "$ART"
mark "begin repo=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown) policies=$POLICIES gpus=$( (nvidia-smi -L 2>/dev/null || true) | wc -l | tr -d " ") sync=${HF_SYNC_REPO:-none}"
stage env stage_env
stage hf stage_hf
stage prepare stage_prepare
stage view stage_view
stage smoke stage_smoke
stage train stage_train
stage eval stage_eval
stage export stage_export
mark "ALL_DONE"
