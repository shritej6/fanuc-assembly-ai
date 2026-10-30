"""Launch mimic/generate.py workers on this host's GPUs (one Isaac Lab container per worker) and roll up the results.

Runs with the host's plain python3 (stdlib only). Each worker is one container started through env/isaaclab_run.sh
(GPU pinned, repo mounted) generating one (skill, product) group into its own recorder file; several workers can share
a GPU. Throughput and acceptance come only from the workers' <out>.stats.json files (accepted demos, attempts,
measured wall time), never from estimates.

    python3 -m sim.assembly_lab.mimic.run_generation --run r1 --gpus 2 3 --workers-per-gpu 2 \
        --groups pick_insert:4 pick_insert:6 --target 50 --max-attempts 400
    python3 -m sim.assembly_lab.mimic.run_generation --rollup ~/t12_data/a7/gen/r1      # report only

Seeds: <seeds>/<skill>/product_<n>_*.hdf5 (agent 3's kept recorder files, gathered by scale_generation.sh; the
.meta.jsonl next to each file carries agent 3's per-episode metadata while the writer is still open).
Outputs: <out>/<run>/<skill>/p<n>/<host>_g<gpu>_w<k>.hdf5 (+ .stats.json, .rejects.jsonl, .log).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA = Path.home() / "t12_data"


def rollup(root: Path) -> dict:
    per: dict[str, dict] = {}
    for f in sorted(root.glob("*/p*/*.hdf5.stats.json")):
        s = json.loads(f.read_text())
        key = f"{s['skill']}:{s['product']}"
        for name in (s["skill"], key):
            g = per.setdefault(name, {"accepted": 0, "attempts": 0, "worker_s": 0.0, "workers": 0, "failed_gates": {}})
            g["accepted"] += s["accepted"]
            g["attempts"] += s["attempts"]
            g["worker_s"] += s.get("wall_s", 0.0)
            g["workers"] += 1
            for k, v in s["failed_gates"].items():
                g["failed_gates"][k] = g["failed_gates"].get(k, 0) + v
    for g in per.values():
        g["acceptance_rate"] = round(g["accepted"] / g["attempts"], 4) if g["attempts"] else None
        g["demos_per_worker_hour"] = round(g["accepted"] / g["worker_s"] * 3600, 2) if g["worker_s"] else None
    return per


def seed_files(seeds: Path, skill: str, product: str) -> list[str]:
    files = sorted(str(p) for p in (seeds / skill).glob(f"product_{product}_*.hdf5")
                   if Path(f"{p}.meta.jsonl").exists())
    if not files:
        raise FileNotFoundError(f"no seed demos for {skill}:{product} under {seeds / skill}")
    return files


def _has_seeds(seeds: Path, skill: str, product: str) -> bool:
    try:
        seed_files(seeds, skill, product)
    except FileNotFoundError as e:
        print(f"skipping {skill}:{product}: {e}", flush=True)
        return False
    return True


def foreign_on_gpu(gpu: int, mine: str) -> list[str]:
    """Running isaaclab containers on this GPU whose command does not write under our output root."""
    names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True,
                           check=True).stdout.split()
    out = []
    for n in names:
        if not n.startswith(f"isaaclab-gpu{gpu}-"):
            continue
        args = subprocess.run(["docker", "inspect", "--format", "{{.Args}}", n], capture_output=True, text=True,
                              check=False).stdout
        if mine not in args:
            out.append(n)
    return out


def launch(a) -> list[subprocess.Popen]:
    host = socket.gethostname()
    root = a.out / a.run
    busy = {g: foreign_on_gpu(g, str(root)) for g in a.gpus}
    if any(busy.values()) and not a.allow_shared:
        sys.exit(f"GPUs in use by containers that are not ours: { {g: v for g, v in busy.items() if v} }")
    slots = [(g, w) for w in range(a.workers_per_gpu) for g in a.gpus]
    procs = []
    groups = [g for g in a.groups if _has_seeds(a.seeds, *g.split(":"))]
    if not groups:
        sys.exit(f"no seed demos for any of {a.groups} under {a.seeds}")
    for i, (gpu, w) in enumerate(slots):
        skill, product = groups[i % len(groups)].split(":")
        out = root / skill / f"p{product}" / f"{host}_g{gpu}_w{w}.hdf5"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [str(a.repo / "sim/assembly_lab/env/isaaclab_run.sh"), "-m", "sim.assembly_lab.mimic.generate",
               "--skill", skill, "--product", product, "--seeds", *seed_files(a.seeds, skill, product),
               "--out", str(out), "--target", str(a.target), "--max-attempts", str(a.max_attempts),
               "--seed", str(a.seed + 1000 * gpu + 100 * w + i), "--cameras", *a.cameras,
               *(["--allow-touch-proxy"] if a.allow_touch_proxy else [])]
        env = dict(os.environ, GPU=str(gpu), REPO=str(a.repo))
        log = open(f"{out}.log", "w")  # noqa: SIM115 - one file per worker, closed when this launcher exits
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
        print(f"gpu {gpu} slot {w}: {skill}:{product} -> {out}", flush=True)
        time.sleep(a.stagger_s)  # Isaac Sim start-up is heavy; do not start every container at once
    return procs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollup", type=Path, default=None, help="only print the roll-up of a run directory")
    ap.add_argument("--run", default="r1", help="run name (output sub-directory)")
    ap.add_argument("--repo", type=Path, default=DATA / "repo_a7", help="repo copy mounted into the containers")
    ap.add_argument("--seeds", type=Path, default=DATA / "a7" / "seeds")
    ap.add_argument("--out", type=Path, default=DATA / "a7" / "gen")
    ap.add_argument("--groups", nargs="+", default=["pick_insert:4"], help="skill:product, e.g. pick_insert:4")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--target", type=int, default=50, help="accepted demos per worker")
    ap.add_argument("--max-attempts", type=int, default=400, help="per worker")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cameras", nargs="*", default=["table", "side", "wrist"])
    ap.add_argument("--stagger-s", type=float, default=30.0)
    ap.add_argument("--allow-shared", action="store_true", help="start even if a foreign container uses a GPU")
    ap.add_argument("--allow-touch-proxy", action="store_true", help="forwarded to generate.py (stamped per demo)")
    a = ap.parse_args()
    if a.rollup:
        print(json.dumps(rollup(a.rollup.expanduser()), indent=2))
        return
    a.repo, a.seeds, a.out = a.repo.expanduser(), a.seeds.expanduser(), a.out.expanduser()
    procs = launch(a)
    codes = [p.wait() for p in procs]
    report = {"host": socket.gethostname(), "exit_codes": codes, "per_group": rollup(a.out / a.run)}
    (a.out / a.run / f"report_{socket.gethostname()}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
