"""HYBRID 14:00 entry point (user-approved 2026-10-31): policy_direct.py with HybridRuntime as the Policy, so GR00T
drives the approach and the carry and the planner takes the final grasp and insert (policy_learn/deploy/
hybrid_runtime.py). run.py, policy.py and cell_client.py are not edited. Same flags and report as policy_direct.py:

    cd robot/cell
    DEPS="--with pyzmq --with msgpack --with msgpack-numpy"
    GROOT_EXEC_HORIZON=8 uv run $DEPS python policy_hybrid.py --mock --checkpoint groot://127.0.0.1:5557
    GROOT_EXEC_HORIZON=8 uv run $DEPS python policy_hybrid.py --checkpoint groot://127.0.0.1:5557   # the slot
"""
from __future__ import annotations

import sys

import policy_direct as direct

from policy_learn.deploy.hybrid_runtime import HybridRuntime


class Policy(HybridRuntime):
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
