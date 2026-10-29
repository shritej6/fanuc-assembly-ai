"""Deterministic absolute-UF0 pick-and-insert motion compilation."""

from __future__ import annotations

import math
from dataclasses import replace
from itertools import pairwise

import pytest

from assembly.motion import (
    AbsolutePoseUF0,
    MotionConfig,
    MotionPhase,
    OrientedPartPose,
    RestingState,
    Waypoint,
    compile_pick_insert,
    compile_pick_to_insert,
    compile_reinsert,
    compile_release_retreat,
    compile_transfer_held,
)


def _pose(x: float, y: float, z: float, yaw: float) -> OrientedPartPose:
    return OrientedPartPose(
        tcp=AbsolutePoseUF0(
            x=x,
            y=y,
            z=z,
            w=-180.0,
            p=0.0,
            r=yaw,
            frame="UF0",
            source="calibrated",
        ),
        resting_state=RestingState.INSTALLED,
        up_face="installed:+z",
    )


def _config() -> MotionConfig:
    return MotionConfig(
        approach_mm=30.0,
        lift_mm=50.0,
        hover_mm=20.0,
        retreat_mm=40.0,
        max_linear_step_mm=15.0,
        max_yaw_step_deg=10.0,
        max_waypoints_per_chunk=3,
    )


def _current() -> Waypoint:
    return Waypoint(200.0, 0.0, -250.0, -180.0, 0.0, 0.0, 0.0)


def test_compile_pick_insert_emits_ordered_bounded_motion() -> None:
    """A missing or reordered phase, oversized step, or oversized chunk is unsafe."""
    grasp = _pose(250.0, -100.0, -300.0, 10.0)
    target = _pose(340.0, 20.0, -280.0, 40.0)

    chunks = compile_pick_insert(_current(), grasp, target, gripper_roll_deg=0.0, config=_config())

    phases = list(dict.fromkeys(chunk.phase for chunk in chunks))
    assert phases == list(MotionPhase)
    assert all(1 <= len(chunk.waypoints) <= 3 for chunk in chunks)
    waypoints = [waypoint for chunk in chunks for waypoint in chunk.waypoints]
    assert math.dist(_current().xyz, waypoints[0].xyz) <= 15.0 + 1e-9
    for before, after in pairwise(waypoints):
        distance = math.dist(before.xyz, after.xyz)
        yaw_step = abs(after.r - before.r)
        assert distance <= 15.0 + 1e-9
        assert yaw_step <= 10.0 + 1e-9


def test_compile_pick_insert_grips_carries_inserts_releases_and_retreats() -> None:
    """Grip state and vertical insertion/retreat direction are part of the motion contract."""
    target = _pose(340.0, 20.0, -280.0, 40.0)
    chunks = compile_pick_insert(
        _current(),
        _pose(250.0, -100.0, -300.0, 10.0),
        target,
        gripper_roll_deg=0.0,
        config=_config(),
    )
    by_phase = {
        phase: [waypoint for chunk in chunks if chunk.phase == phase for waypoint in chunk.waypoints]
        for phase in MotionPhase
    }

    for phase in (MotionPhase.APPROACH, MotionPhase.DESCEND, MotionPhase.RELEASE, MotionPhase.RETREAT):
        assert {waypoint.gripper for waypoint in by_phase[phase]} == {0.0}
    for phase in (MotionPhase.GRIP, MotionPhase.LIFT, MotionPhase.CARRY_YAW, MotionPhase.HOVER):
        assert {waypoint.gripper for waypoint in by_phase[phase]} == {1.0}
    assert {waypoint.gripper for waypoint in by_phase[MotionPhase.VERTICAL_INSERT]} == {1.0}
    insert = by_phase[MotionPhase.VERTICAL_INSERT]
    retreat = by_phase[MotionPhase.RETREAT]
    assert {(waypoint.x, waypoint.y, waypoint.r) for waypoint in insert + retreat} == {(340.0, 20.0, 40.0)}
    assert insert[-1].z == target.tcp.z
    assert retreat[-1].z == target.tcp.z + _config().retreat_mm


@pytest.mark.parametrize("which", ["grasp", "target"])
def test_compile_pick_insert_rejects_fallen_part_pose(which: str) -> None:
    """A fallen grasp or target must never be converted into robot motion."""
    grasp = _pose(250.0, -100.0, -300.0, 10.0)
    target = _pose(340.0, 20.0, -280.0, 40.0)
    fallen = replace(grasp if which == "grasp" else target, resting_state=RestingState.FALLEN)

    with pytest.raises(ValueError, match=f"{which}.*installed"):
        compile_pick_insert(
            _current(),
            fallen if which == "grasp" else grasp,
            fallen if which == "target" else target,
            gripper_roll_deg=0.0,
            config=_config(),
        )


def test_compile_pick_insert_preserves_up_face_and_allows_only_yaw_alignment() -> None:
    """Measured W/P is preserved; the compiler may only align yaw."""
    grasp = _pose(250.0, -100.0, -300.0, 10.0)
    target = _pose(340.0, 20.0, -280.0, 40.0)
    chunks = compile_pick_insert(_current(), grasp, target, gripper_roll_deg=0.0, config=_config())

    assert {waypoint.w for chunk in chunks for waypoint in chunk.waypoints} == {-180.0}
    assert {waypoint.p for chunk in chunks for waypoint in chunk.waypoints} == {0.0}
    assert chunks[-1].waypoints[-1].r == 40.0

    changed_w = replace(target, tcp=replace(target.tcp, w=-170.0))
    preserved = compile_pick_insert(_current(), grasp, changed_w, gripper_roll_deg=0.0, config=_config())
    assert {point.w for chunk in preserved for point in chunk.waypoints} == {grasp.tcp.w}
    changed_face = replace(target, up_face="installed:-z")
    with pytest.raises(ValueError, match="up-face"):
        compile_pick_insert(_current(), grasp, changed_face, gripper_roll_deg=0.0, config=_config())


def test_yaw_alignment_starts_after_lift_and_uses_shortest_wrapped_path() -> None:
    """A 170 to -170 alignment is a 20-degree carry rotation, never a 340-degree turn."""
    chunks = compile_pick_insert(
        _current(),
        _pose(250.0, -100.0, -300.0, 170.0),
        _pose(250.0, -100.0, -300.0, -170.0),
        gripper_roll_deg=0.0,
        config=_config(),
    )

    before_carry = [
        waypoint.r
        for chunk in chunks
        if chunk.phase in {MotionPhase.DESCEND, MotionPhase.GRIP, MotionPhase.LIFT}
        for waypoint in chunk.waypoints
    ]
    approach = [waypoint.r for chunk in chunks if chunk.phase is MotionPhase.APPROACH for waypoint in chunk.waypoints]
    carry_yaws = [
        waypoint.r for chunk in chunks if chunk.phase == MotionPhase.CARRY_YAW for waypoint in chunk.waypoints
    ]
    after_carry = [
        waypoint.r
        for chunk in chunks
        if chunk.phase in {MotionPhase.HOVER, MotionPhase.VERTICAL_INSERT, MotionPhase.RELEASE, MotionPhase.RETREAT}
        for waypoint in chunk.waypoints
    ]
    assert approach[-1] == 170.0
    assert set(before_carry) == {170.0}
    assert carry_yaws == [180.0, 190.0]
    assert set(after_carry) == {190.0}


def test_yaw_alignment_finishes_at_lift_before_horizontal_transfer() -> None:
    chunks = compile_pick_insert(
        _current(),
        _pose(250.0, -100.0, -300.0, 10.0),
        _pose(340.0, 20.0, -280.0, 40.0),
        gripper_roll_deg=0.0,
        config=_config(),
    )

    carry = [point for chunk in chunks if chunk.phase is MotionPhase.CARRY_YAW for point in chunk.waypoints]
    hover = [point for chunk in chunks if chunk.phase is MotionPhase.HOVER for point in chunk.waypoints]

    assert {(point.x, point.y) for point in carry} == {(250.0, -100.0)}
    assert {point.r for point in hover} == {40.0}
    assert hover[-1].xyz == (340.0, 20.0, -260.0)


def test_connector_gripper_roll_is_applied_to_grasp_and_target_yaw() -> None:
    chunks = compile_pick_insert(
        _current(),
        _pose(250.0, -100.0, -300.0, 10.0),
        _pose(340.0, 20.0, -280.0, 40.0),
        gripper_roll_deg=90.0,
        config=_config(),
    )
    by_phase = {
        phase: [point for chunk in chunks if chunk.phase is phase for point in chunk.waypoints] for phase in MotionPhase
    }

    assert by_phase[MotionPhase.APPROACH][-1].r == 100.0
    assert by_phase[MotionPhase.VERTICAL_INSERT][-1].r == 130.0


def test_provisional_insert_keeps_part_held_until_verification() -> None:
    chunks = compile_pick_to_insert(
        _current(),
        _pose(250.0, -100.0, -300.0, 10.0),
        _pose(340.0, 20.0, -280.0, 40.0),
        gripper_roll_deg=0.0,
        config=_config(),
    )

    assert chunks[-1].phase is MotionPhase.VERTICAL_INSERT
    assert chunks[-1].waypoints[-1].gripper == 1.0
    assert not {MotionPhase.RELEASE, MotionPhase.RETREAT} & {chunk.phase for chunk in chunks}


def test_reinsert_uses_only_held_part_recovery_phases() -> None:
    current = Waypoint(340.0, 20.0, -280.0, -180.0, 0.0, 20.0, 1.0)
    chunks = compile_reinsert(
        current,
        _pose(340.0, 20.0, -280.0, 40.0),
        gripper_roll_deg=0.0,
        config=_config(),
    )

    assert list(dict.fromkeys(chunk.phase for chunk in chunks)) == [
        MotionPhase.LIFT,
        MotionPhase.CARRY_YAW,
        MotionPhase.HOVER,
        MotionPhase.VERTICAL_INSERT,
    ]
    assert {point.gripper for chunk in chunks for point in chunk.waypoints} == {1.0}


def test_verified_insert_releases_then_retreats() -> None:
    target = _pose(340.0, 20.0, -280.0, 40.0)
    chunks = compile_release_retreat(
        Waypoint(340.0, 20.0, -280.0, -180.0, 0.0, 40.0, 1.0),
        target,
        gripper_roll_deg=0.0,
        config=_config(),
    )

    assert [chunk.phase for chunk in chunks] == [MotionPhase.RELEASE, MotionPhase.RETREAT]
    assert chunks[0].waypoints[-1].gripper == 0.0
    assert chunks[-1].waypoints[-1].z == target.tcp.z + _config().retreat_mm


def test_release_preserves_measured_roll_pitch_within_verified_endpoint() -> None:
    target = _pose(340.0, 20.0, -280.0, 40.0)
    chunks = compile_release_retreat(
        Waypoint(340.0, 20.0, -280.0, -179.9, 0.1, 40.0, 1.0),
        target,
        gripper_roll_deg=0.0,
        config=_config(),
    )

    points = [point for chunk in chunks for point in chunk.waypoints]
    assert {(point.w, point.p) for point in points} == {(-179.9, 0.1)}


def test_release_does_not_translate_before_opening_gripper() -> None:
    current = Waypoint(300.0, 20.0, -280.0, -180.0, 0.0, 40.0, 1.0)
    chunks = compile_release_retreat(
        current,
        _pose(400.0, 20.0, -280.0, 40.0),
        gripper_roll_deg=0.0,
        config=replace(_config(), max_linear_step_mm=10.0),
    )

    assert chunks[0].waypoints[-1].xyz == current.xyz
    assert chunks[0].waypoints[-1].gripper == 0.0


def test_release_preserves_equivalent_wrapped_yaw_without_full_turn() -> None:
    chunks = compile_release_retreat(
        Waypoint(340.0, 20.0, -280.0, -180.0, 0.0, 190.0, 1.0),
        _pose(340.0, 20.0, -280.0, -170.0),
        gripper_roll_deg=0.0,
        config=_config(),
    )

    assert {point.r for chunk in chunks for point in chunk.waypoints} == {190.0}


def test_held_reject_transfer_releases_only_at_calibrated_target() -> None:
    target = _pose(500.0, -300.0, -300.0, 0.0)
    chunks = compile_transfer_held(
        Waypoint(340.0, 20.0, -280.0, -180.0, 0.0, 40.0, 1.0),
        target,
        gripper_roll_deg=0.0,
        config=_config(),
    )

    assert chunks[-2].phase is MotionPhase.RELEASE
    assert chunks[-2].waypoints[-1].xyz == (target.tcp.x, target.tcp.y, target.tcp.z)
    assert chunks[-1].phase is MotionPhase.RETREAT


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"frame": "base"}, "UF0"),
        ({"source": "simulator_ground_truth"}, "calibrated"),
        ({"x": math.nan}, "finite"),
    ],
)
def test_compile_pick_insert_rejects_untrusted_pose(change: dict[str, object], message: str) -> None:
    """Non-UF0, simulator-truth, and nonfinite targets cannot become commands."""
    grasp = _pose(250.0, -100.0, -300.0, 10.0)
    target = _pose(340.0, 20.0, -280.0, 40.0)
    target = replace(target, tcp=replace(target.tcp, **change))

    with pytest.raises(ValueError, match=message):
        compile_pick_insert(_current(), grasp, target, gripper_roll_deg=0.0, config=_config())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"approach_mm": -1.0}, "clearance"),
        ({"max_linear_step_mm": 0.0}, "positive"),
        ({"max_yaw_step_deg": math.inf}, "finite"),
        ({"max_waypoints_per_chunk": 101}, "1..100"),
    ],
)
def test_compile_pick_insert_rejects_unsafe_config(change: dict[str, object], message: str) -> None:
    """Bad caller-supplied bounds must fail closed instead of hanging or oversizing actions."""
    config = replace(_config(), **change)

    with pytest.raises(ValueError, match=message):
        compile_pick_insert(
            _current(),
            _pose(250.0, -100.0, -300.0, 10.0),
            _pose(340.0, 20.0, -280.0, 40.0),
            gripper_roll_deg=0.0,
            config=config,
        )


@pytest.mark.parametrize("roll", [math.nan, math.inf, "90"])
def test_compile_pick_insert_rejects_invalid_gripper_roll(roll: object) -> None:
    with pytest.raises(ValueError, match="gripper_roll_deg"):
        compile_pick_insert(
            _current(),
            _pose(250.0, -100.0, -300.0, 10.0),
            _pose(340.0, 20.0, -280.0, 40.0),
            gripper_roll_deg=roll,  # type: ignore[arg-type]
            config=_config(),
        )


def test_compile_pick_insert_rejects_closed_gripper_at_new_pick() -> None:
    closed = replace(_current(), gripper=1.0)

    with pytest.raises(ValueError, match="open gripper"):
        compile_pick_insert(closed, _pose(250.0, -100.0, -300.0, 0.0), _pose(300.0, 0.0, -300.0, 0.0), gripper_roll_deg=0.0, config=_config())
