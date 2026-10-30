"""Malformed safety configuration and geometry must fail with typed errors."""

from dataclasses import replace

import pytest

from assembly.motion import MotionChunk, MotionPhase
from assembly.safety import AABB, OfficialActionGuardViolation, SafetyConfigurationViolation
from tests.test_safety import _config, _owner, _supervisor, _waypoint


def _plan() -> tuple[MotionChunk, ...]:
    return (MotionChunk(MotionPhase.APPROACH, (_waypoint(220.0, 0.0, -250.0),)),)


@pytest.mark.parametrize("dt_s", [0.0, -0.1, float("nan"), float("inf"), True])
def test_invalid_dt_is_a_typed_safety_configuration_error(dt_s: object) -> None:
    supervisor = _supervisor()
    supervisor._config = replace(_config(), dt_s=dt_s)  # type: ignore[arg-type]

    with pytest.raises(SafetyConfigurationViolation, match="dt_s"):
        supervisor.validate(
            _plan(), start=_waypoint(200.0, 0.0, -250.0), fixtures=(), parts=(), owner=_owner()
        )


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [
        (None, (1.0, 1.0, 1.0)),
        ((0.0, 0.0, 0.0), None),
        ([0.0, 0.0, 0.0], (1.0, 1.0, 1.0)),
        ((0.0, 0.0), (1.0, 1.0, 1.0)),
    ],
)
def test_malformed_aabb_bounds_fail_with_typed_error(minimum: object, maximum: object) -> None:
    obstacle = AABB("bad", minimum, maximum)  # type: ignore[arg-type]

    with pytest.raises(SafetyConfigurationViolation, match="three-axis bounds"):
        _supervisor().validate(
            _plan(),
            start=_waypoint(200.0, 0.0, -250.0),
            fixtures=(obstacle,),
            parts=(),
            owner=_owner(),
        )


def test_official_action_bounds_failure_is_a_typed_safety_violation() -> None:
    plan = (MotionChunk(MotionPhase.APPROACH, (_waypoint(700.0, 0.0, -250.0),)),)

    with pytest.raises(OfficialActionGuardViolation, match="official UF0 action guard"):
        _supervisor().validate(
            plan,
            start=_waypoint(200.0, 0.0, -250.0),
            fixtures=(),
            parts=(),
            owner=_owner(),
        )
