"""Sourced fallback hold used before the first valid cell observation."""

from dataclasses import dataclass

from assembly.motion import MotionConfigError, Waypoint
from robot.adapter import DT_LIMITS, ActionValidationError, guard_tcp_action


@dataclass(frozen=True)
class StartupHoldConfig:
    """Caller-sourced fallback TCP used before the first valid observation."""

    waypoint: Waypoint
    source: str


def validated_startup_hold(config: StartupHoldConfig) -> Waypoint:
    """Return the caller-provided startup hold after boundary validation."""
    if not isinstance(config, StartupHoldConfig) or not isinstance(config.waypoint, Waypoint):
        raise MotionConfigError("startup hold must be a StartupHoldConfig with a Waypoint")
    if not isinstance(config.source, str) or not config.source.strip():
        raise MotionConfigError("startup hold requires a recorded source")
    if config.waypoint.gripper not in {0.0, 1.0}:
        raise MotionConfigError("startup hold gripper must be exactly open or closed")
    try:
        guard_tcp_action(config.waypoint.as_action(), dt=DT_LIMITS[0])
    except ActionValidationError as error:
        raise MotionConfigError(f"startup hold is outside the official action contract: {error}") from error
    return config.waypoint
