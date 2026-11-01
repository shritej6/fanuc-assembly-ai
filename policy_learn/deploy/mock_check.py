"""Run the official run.py --mock check with GrootPolicy in place of the stock policy.
Authored by Shritej Gunda."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CELL = REPO / "robot" / "cell"
sys.path[:0] = [str(CELL), str(REPO)]

import run  # robot/cell/run.py, unchanged

from policy_learn.deploy.groot_policy import GrootPolicy

INSTANCES: list[GrootPolicy] = []


class _Tracked(GrootPolicy):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        INSTANCES.append(self)


def _summary() -> None:
    for pol in INSTANCES:
        held = sum(e["held"] for e in pol.guard_log)
        print(f"  guard: {len(pol.guard_log)} chunks changed ({held} held, {len(pol.guard_log) - held} clamped)")
        for e in pol.guard_log[:5]:
            print(f"    rollout {e['rollout'] + 1} act {e['step']}: {'HOLD' if e['held'] else 'clamp'}: {'; '.join(e['reasons'])[:160]}")


if __name__ == "__main__":
    run.Policy = _Tracked  # run.main() and run.verdict() read this module global
    sys.argv = [str(CELL / "run.py"), "--mock", *[a for a in sys.argv[1:] if a != "--mock"]]
    try:
        run.main()
    finally:
        _summary()
