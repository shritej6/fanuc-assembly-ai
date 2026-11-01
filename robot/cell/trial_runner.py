"""Real-cell trial containment for typed fail-closed policy holds.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from assembly.execution import CellTrialHeld

LOG = logging.getLogger(__name__)


def run_cell_trials(
    cell: object,
    policy: object,
    *,
    run_all_fn: Callable[[object, object], None],
) -> None:
    """Keep later trials available after one trial intentionally emits no action."""
    while True:
        try:
            run_all_fn(cell, policy)
            return
        except CellTrialHeld as error:
            LOG.error(
                "CELL trial entered Held/Andon; no action was sent: %s. "
                "Waiting for that trial to end before continuing.",
                error,
            )
