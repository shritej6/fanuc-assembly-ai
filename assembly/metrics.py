"""Shared measured metrics derived from T09 controller events."""

from assembly.controller_ports import ControllerEvent


def count_assist_episodes(events: tuple[ControllerEvent, ...]) -> int:
    """Count distinct Andon episodes without inflating repeated alarm events."""
    count = 0
    active = False
    for event in events:
        if event.kind == "andon" and not active:
            count += 1
            active = True
        elif event.kind in {"resume", "complete", "abort"}:
            active = False
    return count
