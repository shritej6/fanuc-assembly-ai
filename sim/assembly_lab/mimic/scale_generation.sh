#!/usr/bin/env bash
# Scale Mimic generation over several Brev hosts from the controller (the Mac with the fa-* ssh aliases).
# Every step is idempotent; workers run detached on the hosts (nohup), so the controller may sleep in between.
#
#   M=sim/assembly_lab/mimic/scale_generation.sh
#   $M repo                           # repo_a7 = fa-sim run10 copy + repo_sync2 env (md5-verified) + mimic/
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
SRC_REPO=${SRC_REPO:-t12_data/repo_run10}  # agent 3's recording copy: the skills + gates the seeds came from
ENV_REPO=${ENV_REPO:-t12_data/repo_sync2}  # the current env (real gripper mass); overlaid on SRC_REPO
ENV_MD5=${ENV_MD5:-dbcfe5fd48d73cde763ffdc4316291a5}  # md5 of ${ENV_REPO}.md5, the manifest of its files
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
    COPYFILE_DISABLE=1 tar -C "$SCR/seeds_all" -cf - . | \
      ssh -o ControlPath=none -o ConnectTimeout=15 "$h" 'mkdir -p ~/t12_data/a7/seeds && tar -C ~/t12_data/a7/seeds -xf -'
  done
}

repo() {
  # repo_a7 = agent 3's run10 copy (skills, robot/, policy_learn) + the current env (ENV_REPO, verified against its md5
  # manifest, whose own md5 must be ENV_MD5) + this checkout's mimic/. ENV_MANIFEST_MD5 is stamped into every demo.
  mkdir -p "$SCR/repo_a7"
  "${RSYNC[@]}" --delete --exclude .git --exclude __pycache__ "${SRC_REPO_HOST}:${SRC_REPO}/" "$SCR/repo_a7/"
  "${RSYNC[@]}" --exclude .git --exclude __pycache__ "${SRC_REPO_HOST}:${ENV_REPO}/" "$SCR/repo_a7/"
  "${RSYNC[@]}" "${SRC_REPO_HOST}:${ENV_REPO}.md5" "$SCR/env_manifest.md5"
  python3 - "$SCR/env_manifest.md5" "$SCR/repo_a7" "$ENV_MD5" <<'PY'
import hashlib, sys
from pathlib import Path
manifest, root, want = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
got = hashlib.md5(manifest.read_bytes()).hexdigest()
if got != want:
    sys.exit(f"env manifest md5 {got} != expected {want}")
bad = [name for digest, name in (line.split(None, 1) for line in manifest.read_text().splitlines() if line.strip())
       if not name.strip().startswith("sim/assembly_lab/mimic/")  # replaced by this checkout's mimic/ below
       and hashlib.md5((root / name.strip()).read_bytes()).hexdigest() != digest]
if bad:
    sys.exit(f"env files differ from the manifest: {bad[:10]}")
(root / "ENV_MANIFEST_MD5").write_text(want + "\n")
print(f"[repo] env verified against manifest {want}")
PY
  rm -rf "$SCR/repo_a7/sim/assembly_lab/mimic"
  mkdir -p "$SCR/repo_a7/sim/assembly_lab/mimic"
  COPYFILE_DISABLE=1 tar -C "$HERE/sim/assembly_lab/mimic" --exclude __pycache__ -cf - . | tar -C "$SCR/repo_a7/sim/assembly_lab/mimic" -xf -
  for h in $HOSTS; do push_repo "$h"; done
}

push_repo() {
  # tar over ssh, then md5-check every file on the host: macOS openrsync was seen to skip changed files silently.
  # Files are overwritten in place (running workers keep the directory they mounted).
  local h=$1
  (cd "$SCR/repo_a7" && find . -type f ! -path '*/__pycache__/*' -print0 | xargs -0 md5 -r) > "$SCR/repo_a7.md5"
  COPYFILE_DISABLE=1 tar -C "$SCR/repo_a7" --exclude __pycache__ -cf - . | \
    ssh -o ControlPath=none -o ConnectTimeout=15 "$h" 'mkdir -p ~/t12_data/repo_a7 && tar -C ~/t12_data/repo_a7 -xf -'
  local bad
  bad=$(ssh -o ControlPath=none -o ConnectTimeout=15 "$h" 'cd ~/t12_data/repo_a7 && md5sum -c - 2>/dev/null | grep -vc ": OK$"' \
    < "$SCR/repo_a7.md5" || true)
  [[ "$bad" == "0" ]] || { echo "[repo] $h: $bad files differ after the copy" >&2; return 1; }
  echo "[repo] $h ok (all files md5-verified)"
}

launch() {
  local run=$1; shift
  : "${MIMIC_GROUPS:?set MIMIC_GROUPS, e.g. MIMIC_GROUPS=\"pick_insert:4 pick_insert:6\"}"
  for spec in "$@"; do
    IFS=: read -r h gpus wpg <<< "$spec"
    # only the redirected launcher goes to the background, so the ssh session closes at once
    # CACHE_TAG: our own Kit/shader cache dirs, apart from other agents' caches on shared hosts
    local cmd="cd ~/t12_data/repo_a7 && mkdir -p ~/t12_data/a7/gen/$run && { CACHE_TAG=${CACHE_TAG:-_a7} nohup python3 -m sim.assembly_lab.mimic.run_generation \
      --run $run --gpus ${gpus//,/ } --workers-per-gpu $wpg --groups $MIMIC_GROUPS --target ${TARGET:-50} \
      --max-attempts ${MAXATT:-400} --seed $RANDOM --stagger-s ${STAGGER:-30} \
      > ~/t12_data/a7/gen/$run/launcher_\$(hostname)_${gpus//,/}.log 2>&1 < /dev/null & }"
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
      # only closed chunks: the newest chunk of a running worker is still being written
      "${SSH[@]}" "$h" "set -e; cd ~/t12_data/repo_a7; \
        files=\$(python3 -m sim.assembly_lab.mimic.run_generation --closed ~/t12_data/a7/gen/$run | grep /$s/ || true); \
        [ -n \"\$files\" ] || exit 0; mkdir -p ~/t12_data/a7/bin; \
        ln -sf \$($TOOLS python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') ~/t12_data/a7/bin/ffmpeg; \
        rm -rf ~/t12_data/lerobot/$name; PATH=~/t12_data/a7/bin:\$PATH $TOOLS python -m sim.assembly_lab.mimic.export_lerobot \
          --inputs \$files --out ~/t12_data/lerobot/$name --workers 8" || echo "[export] $h $s failed"
      if [[ "$h" != "$GATHER" ]]; then
        # relay host -> controller -> gather as one tar stream, replacing the previous export of this host
        "${SSH[@]}" "$h" "test -f ~/t12_data/lerobot/$name/meta/info.json" || continue
        ssh -o ControlPath=none -o ConnectTimeout=15 -n "$h" "tar -C ~/t12_data/lerobot -cf - $name" | \
          ssh -o ControlPath=none -o ConnectTimeout=15 "$GATHER" \
            "mkdir -p ~/t12_data/lerobot && rm -rf ~/t12_data/lerobot/$name && tar -C ~/t12_data/lerobot -xf -"
      fi
    done
  done
  "${SSH[@]}" "$GATHER" "set -e; cd ~/t12_data/repo_a7; srcs=''; for d in ~/t12_data/lerobot/${run}_*; do \
      [ -f \$d/meta/info.json ] || continue; $TOOLS python -m sim.assembly_lab.mimic.validate_lerobot \$d --video-every 10; \
      srcs=\"\$srcs --src \$d\"; done; [ -n \"\$srcs\" ] || { echo no exports yet; exit 0; }; \
      rm -rf ~/t12_data/prepare_check/$run; \
      $TOOLS python -m policy_learn.groot.prepare_dataset \$srcs --out ~/t12_data/prepare_check/$run; \
      echo READY: ~/t12_data/lerobot/${run}_*"
}

case "${1:-}" in
  seeds) seeds ;;
  repo) repo ;;
  push) shift; for h in "$@"; do push_repo "$h"; done ;;
  launch) shift; launch "$@" ;;
  status) status "$2" ;;
  export) export_run "$2" ;;
  *) sed -n '2,15p' "$0"; exit 2 ;;
esac
