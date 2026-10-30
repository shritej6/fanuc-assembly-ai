"""Official TCP-bound regressions for equivalent tool-angle interpolation."""

from __future__ import annotations

from dataclasses import replace

import pytest

from assembly.motion import (
    AbsolutePoseUF0,
    MotionConfig,
    MotionPhase,
    OrientedPartPose,
    RestingState,
    Waypoint,
    compile_pick_insert,
)
from robot.adapter import TCP_LIMITS


def _pose(angles: tuple[float, float, float]) -> OrientedPartPose:
    return OrientedPartPose(
        tcp=AbsolutePoseUF0(250.0, -100.0, -300.0, *angles, "UF0", "calibrated"),
        resting_state=RestingState.INSTALLED,
        up_face="installed:+z",
    )


def _config() -> MotionConfig:
    return MotionConfig(30.0, 50.0, 20.0, 40.0, 15.0, 10.0, 3, -180.0, "test fixture")


def _wrapped_delta(start: float, end: float) -> float:
    return (end - start + 180.0) % 360.0 - 180.0


@pytest.mark.parametrize("field_index", (3, 4, 5), ids=("w", "p", "r"))
@pytest.mark.parametrize(
    ("start_angle", "target_angle"),
    ((-355.0, -55.16443320004055), (355.0, 55.16443320004055)),
    ids=("lower-bound", "upper-bound"),
)
def test_shortest_equivalent_tool_path_stays_inside_official_bounds(
    field_index: int,
    start_angle: float,
    target_angle: float,
) -> None:
    start_values = [-180.0, 0.0, 0.0]
    target_values = list(start_values)
    start_values[field_index - 3] = start_angle
    target_values[field_index - 3] = target_angle
    current = Waypoint(200.0, 0.0, -250.0, *start_values, 0.0)
    target = _pose(tuple(target_values))

    chunks = compile_pick_insert(
        current,
        target,
        replace(target, tcp=replace(target.tcp, x=340.0)),
        gripper_roll_deg=0.0,
        config=_config(),
    )

    points = [point for chunk in chunks for point in chunk.waypoints]
    assert all(
        low <= angle <= high
        for point in points
        for angle, (low, high, _unit) in zip((point.w, point.p, point.r), TCP_LIMITS[3:6], strict=True)
    )
    approach = [point for chunk in chunks if chunk.phase is MotionPhase.APPROACH for point in chunk.waypoints]
    emitted_angle = (approach[-1].w, approach[-1].p, approach[-1].r)[field_index - 3]
    assert _wrapped_delta(start_angle, emitted_angle) == pytest.approx(_wrapped_delta(start_angle, target_angle))
