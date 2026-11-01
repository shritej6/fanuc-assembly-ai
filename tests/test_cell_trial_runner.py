"""A missing measured TCP holds one CELL trial without terminating the run.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import logging

from assembly.execution import CellTrialHeld
from robot.cell.trial_runner import run_cell_trials


def test_cell_trial_hold_sends_no_action_and_runner_continues(caplog) -> None:
    calls = 0

    def fake_run_all(cell, policy) -> None:
        nonlocal calls
        del cell, policy
        calls += 1
        if calls == 1:
            raise CellTrialHeld("CELL refuses action without a measured current TCP")

    with caplog.at_level(logging.ERROR):
        run_cell_trials(object(), object(), run_all_fn=fake_run_all)

    assert calls == 2
    assert "Held/Andon" in caplog.text
    assert "measured current TCP" in caplog.text
