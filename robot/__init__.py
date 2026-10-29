"""Runtime-neutral robot adapter API."""

from robot.adapter import (
    ActionBoundsError,
    ActionShapeError,
    ActionValidationError,
    ActionValueError,
    CellAdapter,
    Controller,
    JointPositionHoldController,
    MockAdapter,
    RobotAdapter,
    SimAdapter,
    guard_tcp_action,
)

__all__ = [
    "ActionBoundsError",
    "ActionShapeError",
    "ActionValidationError",
    "ActionValueError",
    "CellAdapter",
    "Controller",
    "JointPositionHoldController",
    "MockAdapter",
    "RobotAdapter",
    "SimAdapter",
    "guard_tcp_action",
]
