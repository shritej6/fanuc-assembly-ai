"""Test driver (run in robot/cell's environment by test_direct_runtime.py): the official `run.py --mock` with
robot/cell/policy_direct.Policy, its GR00T replaced by the test-only ScriptedGroot. Never for the arm.

    cd robot/cell && uv run python ../../tests/policy_learn/direct_mock_driver.py <trial seconds> <report.json>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CELL = HERE.parents[1] / "robot" / "cell"
sys.path[:0] = [str(CELL), str(HERE.parents[1]), str(HERE)]

import policy_direct
from direct_fakes import ScriptedGroot

BasePolicy = policy_direct.Policy


class ScriptedPolicy(BasePolicy):
    def __init__(self, checkpoint=None):
        fake = ScriptedGroot()
        super().__init__(model=fake)
        fake.runtime = self


if __name__ == "__main__":
    trial_s, report = sys.argv[1], sys.argv[2]
    os.environ["DIRECT_REPORT"] = report
    prompt = yaml.safe_load((CELL / "submission" / "placement.yaml").read_text())["rollouts"][0]["prompt"]
    policy_direct.Policy = ScriptedPolicy
    sys.argv = [str(CELL / "run.py"), "--mock", "--trials", "1", "--trial-s", trial_s, "--instruction", prompt]
    policy_direct.main()
