"""Shared PackML lifecycle evidence rules for terminal run events."""

from __future__ import annotations

from collections.abc import Sequence

from assembly.state_machine import PackMLState
from industry.events import PackMLEvent, ProductResult, ProductResultEvent, RunEvent


class LifecycleEvidenceError(ValueError):
    """A terminal event lacks an active matching PackML lifecycle."""


def validate_terminal_lifecycle(events: Sequence[RunEvent], terminal: ProductResultEvent) -> None:
    """Require the latest earlier matching state to prove the terminal result."""
    matching = tuple(
        event
        for event in events
        if isinstance(event, PackMLEvent)
        and event.run_id == terminal.run_id
        and event.rollout == terminal.rollout
        and event.product == terminal.product
    )
    expected = _terminal_state(terminal.result)
    if not matching or matching[-1].state is not expected:
        detail = f"terminal result lacks active matching PackML lifecycle evidence: expected {expected.value}"
        raise LifecycleEvidenceError(detail)
    if terminal.result is ProductResult.BLOCKED and not _andon_reason(matching[-1].reason):
        raise LifecycleEvidenceError("blocked terminal lacks active Held/Andon lifecycle reason")


def _terminal_state(result: ProductResult) -> PackMLState:
    if result in {ProductResult.GOOD_FIRST_PASS, ProductResult.GOOD_AFTER_REBUILD}:
        return PackMLState.COMPLETE
    if result is ProductResult.BLOCKED:
        return PackMLState.HELD
    return PackMLState.ABORTED


def _andon_reason(reason: str | None) -> bool:
    return isinstance(reason, str) and bool(reason.strip())
