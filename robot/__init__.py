"""Runtime-neutral robot adapter API.
Authored by Shritej Gunda.
"""

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
    guard_joint_action,
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
    "guard_joint_action",
    "guard_tcp_action",
]
