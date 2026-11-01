"""Real-cell rehearsal of Codex's CELL runtime against a lagging, noisy, faulty arm.

Opt-in (several minutes): ``REHEARSAL=1 uv run pytest -q tests/rehearsal`` or ``tools/rehearsal/run.sh``.
Tests documenting a CURRENT blocker are ``xfail(strict=True)``: when Codex fixes it they XPASS and the run
fails, which is the signal to re-review and drop the marker.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("REHEARSAL") != "1", reason="set REHEARSAL=1 (slow, opt-in)")


def _params() -> list[object]:
    from tools.rehearsal.report import SCENARIOS

    params = []
    for scenario in SCENARIOS:
        marks = []
        if scenario.blocker:
            marks.append(pytest.mark.xfail(strict=True, reason=f"KNOWN BLOCKER: {scenario.blocker}"))
        if "groot" in scenario.tags:
            marks.append(pytest.mark.skipif(not os.environ.get("REHEARSAL_GROOT_URL"),
                                            reason="set REHEARSAL_GROOT_URL=groot://127.0.0.1:5555"))
        params.append(pytest.param(scenario.name, marks=marks, id=scenario.name))
    return params


@pytest.mark.parametrize("name", _params() if os.environ.get("REHEARSAL") == "1" else [])
def test_scenario_is_expected_safe(name: str) -> None:
    from tools.rehearsal.report import SCENARIOS, outcome_for, record

    scenario = next(s for s in SCENARIOS if s.name == name)
    outcome = outcome_for(name)
    verdict = scenario.check(outcome)
    record(scenario, outcome, verdict)
    print(f"REHEARSAL {name}: safe={verdict[0]} {verdict[1]}")
    assert verdict[0], verdict[1]


@pytest.mark.xfail(strict=True, reason="KNOWN BLOCKER: REVIEWED_REAL_CELL_FACTORY is None; no real-cell factory")
def test_real_cell_factory_is_pinned() -> None:
    from assembly import cell_runtime

    assert getattr(cell_runtime, "REVIEWED_REAL_CELL_FACTORY", None)


def test_saturday_tilt_cone_allows_three_plus_two_degree_wiggle() -> None:
    import numpy as np

    from assembly.cell_runtime import _saturday_limits
    from policy_learn.deploy.guards import ChunkGuard

    guard = ChunkGuard(_saturday_limits())
    start = [400.0, 0.0, -250.0, -180.0, 0.0, 0.0]
    wiggle = np.array([[400.0, 0.0, -250.0, -180.0, 5.0, 0.0, 0.0]])  # 3 deg tilt + 2 deg twist margin
    result = guard.guard(wiggle, start, 0.0, 0.2)
    assert not result.held, result.reasons


def test_saturday_base_tilt_limit_stays_three_degrees() -> None:
    """The product BASE must sit flat (3 deg); the 6 deg cone is the tool's, never the base's (PR 24, W3)."""
    from assembly.cell_runtime import SATURDAY_GROOT_WATCHDOG_LIMITS

    assert SATURDAY_GROOT_WATCHDOG_LIMITS.base_tilt_limit_deg == 3.0
    assert 4.0 > SATURDAY_GROOT_WATCHDOG_LIMITS.base_tilt_limit_deg  # a 4 deg base tilt must trip
