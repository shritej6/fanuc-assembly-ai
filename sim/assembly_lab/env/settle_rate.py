"""Measure how often the reset settles for each product over random layouts.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--resets", type=int, default=8)
ap.add_argument("--products", default="1,2,3,4,5,6")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()

import sampler
from cell_env import AssemblyCellEnv, CellEnvCfg


def main() -> None:
    out = args.out.expanduser().resolve()
    products = [int(p) for p in args.products.split(",")]
    pool = [u for p in products for u in [f"F0{p}"] + [f"M{p}_0{k}" for k in range(1, 5)]]
    env = AssemblyCellEnv(CellEnvCfg(device=args.device, pool=pool, cameras=()))
    rng = np.random.default_rng(7)
    report = {}
    for p in products:
        ids = [f"F0{p}"] + [f"M{p}_0{k}" for k in range(1, 5)]
        rows = []
        for r in range(args.resets):
            n_in = r % 4                                   # 0..3 parts installed, in sheet order
            installed = {f"M{p}_0{j}": f"F0{p}" for j in range(1, n_in + 1)}
            lay = sampler.random_layout(rng, ids, installed)
            t0 = time.time()
            obs = env.reset({"objects": lay["objects"], "instruction": "settle"})
            s = obs["sim"]
            rows.append({"installed": n_in, "settled": s["settled"], "settle_s": s["settle_s"],
                         "wall_s": round(time.time() - t0, 1), "detail": s["settle_detail"]})
            print(f"[settle] product {p} reset {r} installed {n_in}: {rows[-1]}", flush=True)
        ok = [x for x in rows if x["settled"]]
        report[p] = {"rate": len(ok) / len(rows), "n": len(rows),
                     "mean_settle_s": round(float(np.mean([x["settle_s"] for x in ok])), 3) if ok else None,
                     "failures": [x for x in rows if not x["settled"]]}
        print(f"[settle] product {p}: {len(ok)}/{len(rows)} settled", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=float))
    env.close()


if __name__ == "__main__":
    main()
