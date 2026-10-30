"""Launch mimic/generate.py workers over GPUs and roll up the measured results.

Each worker is one Isaac Sim process (AssemblyCellEnv is single-env) generating one (skill, product) group into its
own file; several workers can share a GPU (L40 48 GB). Throughput and acceptance come only from the workers'
<out>.stats.json files (accepted demos, attempts, wall time), never from estimates.

    python -m sim.assembly_lab.mimic.run_generation --python "$ISAAC_PY" --demos ~/t12_data/demos \
        --out ~/t12_data/mimic --gpus 0 1 2 --workers-per-group 3 --target 150 --groups pick_insert:4
    python -m sim.assembly_lab.mimic.run_generation --rollup ~/t12_data/mimic      # report only
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from .catalog import REPO

SEED_FILE = "{demos}/{skill}/product_{product}.hdf5"  # agent 3's skills/record.py layout


def rollup(root: Path) -> dict:
    per: dict[str, dict] = {}
    for f in sorted(root.glob("*/p*/w*.hdf5.stats.json")):
        s = json.loads(f.read_text())
        g = per.setdefault(s["skill"], {"accepted": 0, "attempts": 0, "worker_s": 0.0, "failed_gates": {}})
        g["accepted"] += s["accepted"]
        g["attempts"] += s["attempts"]
        g["worker_s"] += s.get("wall_s", 0.0)
        for k, v in s["failed_gates"].items():
            g["failed_gates"][k] = g["failed_gates"].get(k, 0) + v
    for g in per.values():
        g["acceptance_rate"] = round(g["accepted"] / g["attempts"], 4) if g["attempts"] else None
        g["demos_per_worker_hour"] = round(g["accepted"] / g["worker_s"] * 3600, 1) if g["worker_s"] else None
    return per


def launch(a) -> list[subprocess.Popen]:
    procs, slot = [], 0
    for group in a.groups:
        skill, product = group.split(":")
        seeds = Path(SEED_FILE.format(demos=a.demos, skill=skill, product=product))
        if not seeds.exists():
            raise FileNotFoundError(f"no seed demos for {group}: {seeds}")
        for w in range(a.workers_per_group):
            gpu = a.gpus[slot % len(a.gpus)]
            slot += 1
            out = a.out / skill / f"p{product}" / f"w{w}.hdf5"
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd = [*shlex.split(a.python), "-m", "sim.assembly_lab.mimic.generate", "--skill", skill,
                   "--product", product, "--seeds", str(seeds), "--out", str(out), "--target", str(a.target),
                   "--max-attempts", str(a.max_attempts), "--seed", str(1000 * int(product) + w),
                   *(["--allow-touch-proxy"] if a.allow_touch_proxy else [])]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                       PYTHONPATH=f"{REPO}:{os.environ.get('PYTHONPATH', '')}")
            log = open(f"{out}.log", "w")  # noqa: SIM115 - closed by the child's exit; one file per worker
            procs.append(subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT))
            print(f"gpu {gpu}: {group} worker {w} -> {out}", flush=True)
            time.sleep(a.stagger_s)  # Isaac Sim start-up is heavy; do not start every process at once
    return procs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollup", type=Path, default=None, help="only print the roll-up of an output root")
    ap.add_argument("--python", default="python", help="Isaac Lab's python (see ~/ISAACLAB_READY)")
    ap.add_argument("--demos", type=Path, default=Path("~/t12_data/demos"))
    ap.add_argument("--out", type=Path, default=Path("~/t12_data/mimic"))
    ap.add_argument("--groups", nargs="+", default=[], help="skill:product, e.g. pick_insert:4")
    ap.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers-per-group", type=int, default=3)
    ap.add_argument("--target", type=int, default=150, help="accepted demos per worker")
    ap.add_argument("--max-attempts", type=int, default=1500, help="per worker")
    ap.add_argument("--stagger-s", type=float, default=20.0)
    ap.add_argument("--allow-touch-proxy", action="store_true", help="forwarded to generate.py (stamped per demo)")
    a = ap.parse_args()
    if a.rollup:
        print(json.dumps(rollup(a.rollup.expanduser()), indent=2))
        return
    a.demos, a.out = a.demos.expanduser(), a.out.expanduser()
    procs = launch(a)
    codes = [p.wait() for p in procs]
    report = {"exit_codes": codes, "per_skill": rollup(a.out)}
    (a.out / "generation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
