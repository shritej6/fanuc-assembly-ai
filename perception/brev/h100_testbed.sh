#!/bin/bash
# Train the test-bed models on one 8xH100 Brev box, in the user's order: RT-DETR (with fallen classes) on N R Q,
# wrist classifier on O, set-P build-state classifier on P. Run on the box after the code is in $REPO and the sets
# are in $DATA_ROOT/synthetic/{N,R,Q,O,P}:
#   nohup bash perception/brev/h100_testbed.sh > ~/t27_h100.log 2>&1 &
# Every stage appends one line to $STATUS (`<utc> <stage> start|ok|FAILED`), so a watcher can report progress.
# START_AT=<stage> reruns from that stage after a fix, reusing the env and artifacts of the earlier stages.
# SKIP="<stage> ..." leaves out stages that already ran on this box (e.g. a finished detector when only the
# classifiers rerun). ALL_DONE is written only when $STATUS holds an "ok" for every required stage, from this
# run or an earlier one; otherwise the run ends with PARTIAL_DONE and the missing stages.
set -Eeuo pipefail   # -E: the ERR trap also fires inside stage functions, so a failure always reaches $STATUS

REPO="${REPO:-$HOME/fanuc-t27}"
DATA_ROOT="${DATA_ROOT:-$HOME/fanuc-testbed-data}"
ART="${ART:-$HOME/fanuc-t27-artifacts}"
RUNS="${RUNS:-$HOME/fanuc-t27-runs}"
STATUS="${STATUS:-$HOME/t27_status.txt}"
VENV_PATH="${VENV_PATH:-$HOME/fanuc-t13-venv-secure}"
GPUS="$(nvidia-smi -L | wc -l)"
# 40 GB A100s cannot hold the H100 detector batch; their config keeps the same images per step via accumulation.
# --id=0, not `| head -1`: head closing the pipe gives nvidia-smi SIGPIPE, which pipefail turns into a silent exit.
GPU_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits --id=0)"
if (( GPU_MIB < 70000 )); then DEFAULT_DETECTOR=rtdetr_a100_40g.yaml; else DEFAULT_DETECTOR=rtdetr_h100.yaml; fi
DETECTOR_CONFIG="${DETECTOR_CONFIG:-$DEFAULT_DETECTOR}"
export REPO DATA_ROOT VENV_PATH
export PYTHON_BIN="$VENV_PATH/bin/python"

mark() { echo "$(date -u +%FT%TZ) $*" >> "$STATUS"; }
trap 'mark "${STAGE:-setup} FAILED"' ERR

START_AT="${START_AT:-env}"
SKIP="${SKIP:-}"   # stages run on another box, e.g. SKIP=detector
started=0
for name in "$START_AT" $SKIP; do   # an unknown name would silently run or skip the wrong stages
  case "$name" in
    env|prepare_main|prepare_build|detector|classifier|build_classifier|t14) ;;
    *) echo "unknown stage $name in START_AT/SKIP" >&2; exit 2 ;;
  esac
done

stage() {
  STAGE="$1"; shift
  [[ "$STAGE" == "$START_AT" ]] && started=1
  if (( ! started )); then mark "$STAGE skipped (START_AT=$START_AT)"; return; fi
  if [[ " $SKIP " == *" $STAGE "* ]]; then mark "$STAGE skipped (SKIP)"; return; fi
  mark "$STAGE start"
  "$@"
  mark "$STAGE ok"
}

setup_env() {
  # Codex's lock is for Python 3.12 + CUDA 12.6 wheels; uv's standalone CPython comes from PyPI, no extra apt source.
  python3 -m pip --version > /dev/null 2>&1 || sudo apt-get install -y python3-pip   # Crusoe's image has no pip
  python3 -m pip install --user --quiet uv
  "$HOME/.local/bin/uv" python install 3.12
  PYTHON="$("$HOME/.local/bin/uv" python find 3.12)" bash "$REPO/perception/rosie/setup_ml.sh"
  if ! "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count() > 0'; then
    # An older driver than the CUDA 12.6 wheels expect: NVIDIA's compat package (its own apt repo, as on the Cosmos boxes).
    sudo apt-get install -y cuda-compat-12-6
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    "$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count(), "GPUs")'
  fi
}

prepare() {   # $1 profile, $2 artifact dir
  PROFILE="$1" SMOKE=0 ARTIFACT_ROOT="$2" PYTHON="$PYTHON_BIN" bash "$REPO/perception/rosie/prepare.sbatch"
  "$PYTHON_BIN" -m perception.rosie.gates "$2/full/manifests/metadata.json"
}

train() {     # $1 launcher, $2 config, $3 artifact dir, $4 run dir
  SMOKE=0 NPROC_PER_NODE="$GPUS" CONFIG="$REPO/perception/configs/$2" ARTIFACT_ROOT="$3" RUN_ROOT="$4" \
    PYTHON="$PYTHON_BIN" TORCHRUN="$VENV_PATH/bin/torchrun" bash "$REPO/perception/rosie/$1"
}

cd "$REPO"
git_commit="$(cat "$REPO/.t13-code-commit")"
mark "begin commit=$git_commit gpus=$GPUS gpu_mib=$GPU_MIB detector_config=$DETECTOR_CONFIG"
stage env setup_env
stage prepare_main prepare testbed "$ART/main"
stage prepare_build prepare testbed_build "$ART/build"
stage detector train detector.sbatch "$DETECTOR_CONFIG" "$ART/main" "$RUNS/main"
stage classifier train classifier.sbatch classifier_h100.yaml "$ART/main" "$RUNS/main"
stage build_classifier train classifier.sbatch classifier_build_h100.yaml "$ART/build" "$RUNS/build"
# PatchCore + FoundationPose validation (user order steps 4-5) run here when their launcher has been delivered by now.
required=(env prepare_main prepare_build detector classifier build_classifier)
if [[ -f "$REPO/perception/brev/h100_t14.sh" ]]; then
  stage t14 bash "$REPO/perception/brev/h100_t14.sh"
  required+=(t14)
fi
missing=()
for name in "${required[@]}"; do
  grep -q "Z $name ok\$" "$STATUS" || missing+=("$name")
done
if (( ${#missing[@]} )); then mark "PARTIAL_DONE missing: ${missing[*]}"; else mark "ALL_DONE"; fi
