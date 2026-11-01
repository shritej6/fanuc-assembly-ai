"""Solve every arm pose of the given sets offline, as the render host will. Exit 1 if any pose has no solution.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import sys

from sdg.scene_sampler import Inputs, chunk_range, n_chunks, sample_scene
from sdg.wrist_ik import reachable


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sets", nargs="+")
    args = ap.parse_args()
    inputs = Inputs.load()
    failed = 0
    for s in args.sets:
        poses = bad = 0
        for c in range(n_chunks(inputs, s)):
            for i in chunk_range(inputs, s, c):
                arm = sample_scene(inputs, s, i)["arm"]
                if arm["mode"] == "parked":
                    continue
                poses += 1
                if not reachable(arm["tcp_uf0"]):
                    bad += 1
                    print(f"  {s} index {i} chunk {c}: no IK for {arm['tcp_uf0']}")
        print(f"set {s}: {poses} arm poses, {bad} without an IK solution")
        failed += bad
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
