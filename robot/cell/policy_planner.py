"""Entry point that runs the official run.py with the scripted PlannerRuntime as its Policy.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import sys

import policy_direct as direct

from policy_learn.deploy.planner_runtime import PlannerRuntime


class Policy(PlannerRuntime):
    """Same construction as policy_direct.Policy (both run.py versions)."""

    def __init__(self, checkpoint: str | None = None, runtime: str | None = None, **kw):
        super().__init__(checkpoint, **kw)
        self.runtime = runtime
        direct.INSTANCES.append(self)

    close = direct.Policy.close


def main() -> None:
    import run  # robot/cell/run.py, unchanged

    run.Policy = Policy
    try:
        run.main()
    finally:
        direct.report()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as e:
        sys.exit(f"error: {e}")
