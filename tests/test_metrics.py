"""Shared assembly metrics have one definition across output adapters.
Authored by Shritej Gunda.
"""

from assembly.controller_ports import ControllerEvent
from assembly.metrics import count_assist_episodes


def test_assists_count_andon_episodes_not_duplicate_alarm_events() -> None:
    events = (
        ControllerEvent("andon", 4, None, "camera disagreement"),
        ControllerEvent("retry", 4, "M4_01", "attempt 2"),
        ControllerEvent("andon", 4, None, "same active episode"),
        ControllerEvent("resume", 4, None, "operator cleared fault"),
        ControllerEvent("andon", 4, None, "spare needed"),
        ControllerEvent("complete", 4, None, "finished"),
        ControllerEvent("andon", 5, None, "next product check"),
        ControllerEvent("abort", 5, None, "cycle aborted"),
    )

    assert count_assist_episodes(events) == 3
