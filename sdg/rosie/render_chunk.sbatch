#!/bin/bash
# One Slurm array task = one chunk of one set. Submit with sdg/rosie/submit.sh (it writes the specs first).
# UNTESTED until the Rosie pilot (T08).
#SBATCH --job-name=fa-sdg
#SBATCH --partition=teaching
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=%x_%A_%a.log
set -euo pipefail

: "${ISAAC_SIF:?set ISAAC_SIF to the Isaac Sim container (see docs/SDG_PLAN.md, section Compute)}"
: "${DATA_ROOT:?set DATA_ROOT}"
: "${REPO:?set REPO to the repo checkout on Rosie}"
: "${SET:?set SET (A, B, C or D)}"
LIMIT="${LIMIT:-0}"

CHUNK=$(printf "%04d" "$SLURM_ARRAY_TASK_ID")
SPECS="$DATA_ROOT/specs/$SET/chunk_$CHUNK.jsonl"
OUT="$DATA_ROOT/synthetic/$SET/chunk_$CHUNK"
cd "$REPO"
if python3 -m sdg.chunk_marks check "$OUT" "$SPECS" 2> /dev/null; then
  echo "chunk $SET/$CHUNK already done from these specs"
  exit 0
fi
mkdir -p "$OUT" "$DATA_ROOT/cache/ov" "$DATA_ROOT/cache/nv"

echo "node $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader)  set $SET  chunk $CHUNK  limit $LIMIT"
# Isaac writes generated files under /isaac-sim at startup. The admin-capped --writable-tmpfs (64 MB) is too small
# and node /tmp is nearly full, so each task gets its own overlay file, removed when the task ends.
OVERLAY="$DATA_ROOT/cache/overlays/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.img"
mkdir -p "$(dirname "$OVERLAY")"
trap 'rm -f "$OVERLAY"' EXIT
singularity overlay create --size 3072 "$OVERLAY"
# --nv mounts the NVIDIA libraries but not the Vulkan / EGL loader manifests that Docker's NVIDIA runtime injects;
# without them Isaac's RTX renderer fails with vkCreateInstance ERROR_INCOMPATIBLE_DRIVER and falls back to CPU.
GPU_BINDS=()
for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json \
         /usr/share/glvnd/egl_vendor.d/10_nvidia.json /usr/share/vulkan/implicit_layer.d/nvidia_layers.json; do
  [[ -f "$f" ]] && GPU_BINDS+=(-B "$f:$f")
done
# Singularity 3.10's library list predates newer driver libraries (e.g. libnvidia-gpucomp, needed by the Vulkan
# driver), so mount every host NVIDIA library of the running driver version and put them on the library path.
# Each library is mounted under its real name and under every soname link pointing to it: NGX (the RTX denoiser)
# loads libnvidia-ngx.so.1 and OptiX libnvoptix.so.1, and without them every frame keeps its ray-tracing noise.
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
LIBDIR=/usr/lib/x86_64-linux-gnu
for lib in "$LIBDIR"/libnvidia-*.so."$DRV" "$LIBDIR"/libnvoptix.so."$DRV" "$LIBDIR"/lib{GLX,EGL}_nvidia.so."$DRV"; do
  [[ -f "$lib" ]] && GPU_BINDS+=(-B "$lib:/opt/nvhost/$(basename "$lib")")
done
for link in "$LIBDIR"/libnvidia-*.so.* "$LIBDIR"/libnvoptix.so.* "$LIBDIR"/lib{GLX,EGL}_nvidia.so.*; do
  [[ -L "$link" && "$(readlink -f "$link")" == *".so.$DRV" ]] && GPU_BINDS+=(-B "$(readlink -f "$link"):/opt/nvhost/$(basename "$link")")
done
# OptiX denoiser weights (nvoptix.bin) and NGX application profiles.
[[ -d /usr/share/nvidia ]] && GPU_BINDS+=(-B /usr/share/nvidia:/usr/share/nvidia)
# Kit's shader / asset caches go to DATA_ROOT so they persist between chunks.
singularity exec --nv ${GPU_BINDS[@]+"${GPU_BINDS[@]}"} --overlay "$OVERLAY" \
  --env ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,ISAAC_SIF="$ISAAC_SIF",FANUC_ROBOT_USD="${FANUC_ROBOT_USD:-}" \
  --env LD_LIBRARY_PATH=/.singularity.d/libs:/opt/nvhost \
  -B "$REPO" -B "$DATA_ROOT" \
  -B "$DATA_ROOT/cache/ov:$HOME/.cache/ov" \
  -B "$DATA_ROOT/cache/nv:$HOME/.nv" \
  "$ISAAC_SIF" /isaac-sim/python.sh "$REPO/sdg/isaac/generate.py" \
  --specs "$SPECS" --out "$OUT" --limit "$LIMIT"
# Isaac can exit 0 after an exception (SimulationApp.close ends the process), so success is the chunk itself.
if [[ -f "$OUT/_FAILED.txt" ]]; then
  echo "chunk $SET/$CHUNK FAILED:"; tail -20 "$OUT/_FAILED.txt"; exit 1
fi
if [[ "$LIMIT" == "0" ]]; then
  python3 -m sdg.chunk_marks check "$OUT" "$SPECS" || exit 1
fi
