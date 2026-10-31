"""Fallback 14:00 entry point: the official run.py with DirectGrootRuntime as its Policy. run.py, policy.py and
cell_client.py are not edited; run.py's `from policy import Policy` global is swapped before run.main() reads it
(the same way policy_learn/deploy/mock_check.py does).

    cd robot/cell
    DEPS="--with pyzmq --with msgpack --with msgpack-numpy"
    uv run $DEPS python policy_direct.py --mock --checkpoint groot://127.0.0.1:5555 --trials 1 --trial-s 120 \
        --instruction "$(python -c 'import yaml;print(yaml.safe_load(open("submission/placement.yaml"))["rollouts"][0]["prompt"])')"
    uv run $DEPS python policy_direct.py --checkpoint groot://127.0.0.1:5555      # the slot: the approved run

Every run.py flag passes through. DIRECT_ROLLOUT=<1..3> forces the placement.yaml rollout; DIRECT_PLACEMENT=<path>
reads another placement file (both are read and validated, and every rollout is planned, when the Policy is built,
before any trial). Empty-gripper detection is OFF on the cell (no gripper width in the observation).
After the run, every step event and guard hold is printed and written to direct_runtime_report.json
(DIRECT_REPORT=<path> to change it).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

from policy_learn.deploy.direct_runtime import EMPTY_GRIPPER_ON_CELL, DirectGrootRuntime  # noqa: E402

INSTANCES: list[DirectGrootRuntime] = []


class Policy(DirectGrootRuntime):
    """checkpoint: "groot://127.0.0.1:5555" (the GR00T server through the ssh tunnel); None reads $GROOT_POLICY.

    Works with both run.py versions: the upstream one (Policy(checkpoint)) and PR #24's (Policy(checkpoint,
    runtime="mock" | "cell"), then policy.close()). `runtime` only labels the report: this policy behaves the same
    on the mock and on the cell."""

    def __init__(self, checkpoint: str | None = None, runtime: str | None = None, **kw):
        super().__init__(checkpoint, **kw)
        self.runtime = runtime
        INSTANCES.append(self)

    def close(self) -> None:
        close = getattr(getattr(self.model, "client", self.model), "close", None)
        if callable(close):
            close()


def report() -> None:
    out = Path(os.environ.get("DIRECT_REPORT", HERE / "direct_runtime_report.json"))
    note = ("empty-gripper detection: OFF on the cell (the observation has no gripper width; joints[6] is binary), "
            "so a part slipping out of closed fingers is not detected" if not EMPTY_GRIPPER_ON_CELL else "")
    data = [{"runtime": p.runtime, "empty_gripper_detection": note or "on", "events": p.events,
             "guard_log": p.guard_log} for p in INSTANCES]
    if note:
        print(f"  NOTE: {note}")
    for pol in data:
        for e in pol["events"]:
            if e["type"] in ("trial_start", "step_start", "grasped", "released", "step_done", "trial_end"):
                detail = {k: v for k, v in e.items() if k not in ("t", "type", "rollout")}
                print(f"  [trial {e['rollout'] + 1}] {e['type']}: {detail}")
        held = sum(g["held"] for g in pol["guard_log"])
        print(f"  guard: {len(pol['guard_log'])} chunks changed ({held} held)")
    if data:
        out.write_text(json.dumps(data, indent=1, default=str))
        print(f"  report: {out}")


def main() -> None:
    import run  # robot/cell/run.py, unchanged

    run.Policy = Policy  # run.main() and run.verdict() read this module global
    try:
        run.main()
    finally:
        report()


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError) as e:  # as run.py: the cell's answer (or no answer) in one line
        sys.exit(f"error: {e}")
