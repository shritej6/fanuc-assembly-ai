#!/usr/bin/env bash
# Scale Mimic generation over several Brev hosts from the controller (the Mac with the fa-* ssh aliases).
# Every step is idempotent; workers run detached on the hosts (nohup), so the controller may sleep in between.
#
#   M=sim/assembly_lab/mimic/scale_generation.sh
#   $M repo                           # repo_a7 = fa-sim:~/t12_data/repo_run10 (agent 3's skills + agent 1's env) + mimic/
#   $M seeds                          # agent 3's closed seed files (+ drop tags) from fa-mimic -> every host
#   MIMIC_GROUPS="pick_insert:4 pick_insert:6" TARGET=50 MAXATT=400 \
#     $M launch r1 fa-sim:1,2,3:2 fa-mimic:0,1,2,3:2 fa-gen1:0,1,2,3:2 fa-a6k-1:0,1:2 fa-l40s-1:0:2
#   $M status r1                      # per-host roll-up of the workers' measured stats
#   $M export r1                      # per host: LeRobot v2.1 export of its kept demos -> fa-mimic; validate + prepare
#
# host spec = <ssh alias>:<comma-separated GPUs>:<workers per GPU>. launch refuses a GPU that runs a container that is
# not ours (run_generation.py checks it on the host). Nothing here stops or touches other agents' containers.
set -euo pipefail
HOSTS=${HOSTS:-"fa-sim fa-mimic fa-gen1 fa-a6k-1 fa-l40s-1"}
GATHER=${GATHER:-fa-mimic}
SKILLS=${SKILLS:-"pick_insert"}
SCR=${SCR:-${TMPDIR:-/tmp}/t12_a7}
SRC_REPO_HOST=${SRC_REPO_HOST:-fa-sim}
SRC_REPO=${SRC_REPO:-t12_data/repo_run10}  # agent 3's recording copy: the skills + gates + env the seeds came from
HERE="$(cd "$(dirname "$0")/../../.." && pwd)"
SSH=(ssh -o ControlPath=none -o ConnectTimeout=15 -n)
RSYNC=(rsync -a -e "ssh -o ControlPath=none -o ConnectTimeout=15")
TOOLS='~/.local/bin/uv run -q --no-project --with h5py --with numpy --with pandas --with pyarrow --with pillow --with pyyaml --with imageio-ffmpeg'

seeds() {
  # agent 3 gathers its closed per-part files on $GATHER and lists them in run10_closed.txt; stage_seeds.py copies
  # only those, with a .drop.json per file from run10_drop_tags.jsonl (run `repo` first: it needs repo_a7 there)
  "${SSH[@]}" "$GATHER" "cd ~/t12_data/repo_a7 && python3 -m sim.assembly_lab.mimic.stage_seeds"
  mkdir -p "$SCR/seeds_all"
  "${RSYNC[@]}" "${GATHER}:t12_data/a7/seed_stage/" "$SCR/seeds_all/"
  for h in $HOSTS; do
    [[ "$h" == "$GATHER" ]] && { "${SSH[@]}" "$h" 'mkdir -p ~/t12_data/a7 && rsync -a ~/t12_data/a7/seed_stage/ ~/t12_data/a7/seeds/'; continue; }
    "${SSH[@]}" "$h" 'mkdir -p ~/t12_data/a7/seeds'
    "${RSYNC[@]}" "$SCR/seeds_all/" "${h}:t12_data/a7/seeds/"
  done
}

repo() {
  mkdir -p "$SCR/repo_a7"
  "${RSYNC[@]}" --delete --exclude .git --exclude __pycache__ "${SRC_REPO_HOST}:${SRC_REPO}/" "$SCR/repo_a7/"
  "${RSYNC[@]}" "$HERE/sim/assembly_lab/mimic/" "$SCR/repo_a7/sim/assembly_lab/mimic/" --exclude __pycache__
  for h in $HOSTS; do
    "${RSYNC[@]}" --delete --exclude __pycache__ "$SCR/repo_a7/" "${h}:t12_data/repo_a7/"
    echo "[repo] $h ok"
  done
}

launch() {
  local run=$1; shift
  : "${MIMIC_GROUPS:?set MIMIC_GROUPS, e.g. MIMIC_GROUPS=\"pick_insert:4 pick_insert:6\"}"
  for spec in "$@"; do
    IFS=: read -r h gpus wpg <<< "$spec"
    local cmd="cd ~/t12_data/repo_a7 && mkdir -p ~/t12_data/a7/gen/$run && nohup python3 -m sim.assembly_lab.mimic.run_generation \
      --run $run --gpus ${gpus//,/ } --workers-per-gpu $wpg --groups $MIMIC_GROUPS --target ${TARGET:-50} \
      --max-attempts ${MAXATT:-400} --seed $RANDOM --stagger-s ${STAGGER:-30} \
      > ~/t12_data/a7/gen/$run/launcher_\$(hostname).log 2>&1 < /dev/null &"
    "${SSH[@]}" "$h" "$cmd"
    echo "[launch] $h gpus $gpus x $wpg workers: $MIMIC_GROUPS"
  done
}

status() {
  local run=$1
  for h in $HOSTS; do
    echo "== $h"
    "${SSH[@]}" "$h" "cd ~/t12_data/repo_a7 2>/dev/null && [ -d ~/t12_data/a7/gen/$run ] && \
      python3 -m sim.assembly_lab.mimic.run_generation --rollup ~/t12_data/a7/gen/$run && \
      docker ps --format '{{.Names}} {{.Status}}' | grep -c isaaclab | sed 's/^/isaaclab containers: /'" 2>/dev/null || true
  done
}

export_run() {
  local run=$1
  for h in $HOSTS; do
    for s in $SKILLS; do
      local name="${run}_${s}_${h}"
      "${SSH[@]}" "$h" "set -e; d=~/t12_data/a7/gen/$run/$s; ls \$d/p*/*.hdf5 >/dev/null 2>&1 || exit 0; \
        cd ~/t12_data/repo_a7; mkdir -p ~/t12_data/a7/bin; \
        ln -sf \$($TOOLS python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') ~/t12_data/a7/bin/ffmpeg; \
        rm -rf ~/t12_data/lerobot/$name; PATH=~/t12_data/a7/bin:\$PATH $TOOLS python -m sim.assembly_lab.mimic.export_lerobot \
          --inputs \$d/p*/*.hdf5 --out ~/t12_data/lerobot/$name --workers 8" || echo "[export] $h $s failed"
      if [[ "$h" != "$GATHER" ]]; then
        mkdir -p "$SCR/lerobot"
        "${RSYNC[@]}" "${h}:t12_data/lerobot/$name" "$SCR/lerobot/" 2>/dev/null || continue
        "${RSYNC[@]}" "$SCR/lerobot/$name" "${GATHER}:t12_data/lerobot/"
      fi
    done
  done
  "${SSH[@]}" "$GATHER" "set -e; cd ~/t12_data/repo_a7; srcs=''; for d in ~/t12_data/lerobot/${run}_*; do \
      [ -f \$d/meta/info.json ] || continue; $TOOLS python -m sim.assembly_lab.mimic.validate_lerobot \$d --video-every 10; \
      srcs=\"\$srcs --src \$d\"; done; rm -rf ~/t12_data/prepare_check/$run; \
      $TOOLS python -m policy_learn.groot.prepare_dataset \$srcs --out ~/t12_data/prepare_check/$run; \
      echo READY: ~/t12_data/lerobot/${run}_*"
}

case "${1:-}" in
  seeds) seeds ;;
  repo) repo ;;
  launch) shift; launch "$@" ;;
  status) status "$2" ;;
  export) export_run "$2" ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
