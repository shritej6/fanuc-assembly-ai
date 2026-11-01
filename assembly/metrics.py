"""Shared measured metrics derived from controller events.
Authored by Shritej Gunda.
"""

from assembly.controller_ports import ControllerEvent


def count_assist_episodes(events: tuple[ControllerEvent, ...]) -> int:
    """Count distinct Andon episodes without inflating repeated alarm events."""
    assists = 0
    active = False
    for event in events:
        if event.kind == "andon" and not active:
            assists += 1
            active = True
        elif event.kind in {"resume", "complete", "abort"}:
            active = False
    return assists
