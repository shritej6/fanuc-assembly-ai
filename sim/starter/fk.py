"""Forward kinematics for the FANUC ER-4iA.
Authored by Shritej Gunda.
"""

import numpy as np

CHAIN = [
    ((0.0, 0.0, 0.33), "z", +1),
    ((0.0, 0.0, 0.00), "y", +1),
    ((0.0, 0.0, 0.26), "y", -1),
    ((0.0, 0.0, 0.02), "x", -1),
    ((0.29, 0.0, 0.0), "y", -1),
    ((0.0, 0.0, 0.00), "x", -1),
]


def _rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(q, upto=6):
    """Flange pose in the robot base frame. Returns (R, p)."""
    R = np.eye(3)
    p = np.zeros(3)
    for i in range(upto):
        off, axis, sign = CHAIN[i]
        p = p + R @ np.array(off)
        R = R @ _rot(axis, sign * q[i])
    return R, p


def tool_pose(q, tool_offset):
    """Pose of the tool tip, given its offset along TOOL_AXIS in flange coords."""
    R, p = fk(q)
    return R, p + R @ np.array(tool_offset)


def jacobian(q, tool_offset, eps=1e-6):
    """Numerical 6xN Jacobian of the tool pose (linear rows then angular rows)."""
    R0, p0 = tool_pose(q, tool_offset)
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        dq = np.array(q, float)
        dq[i] += eps
        R1, p1 = tool_pose(dq, tool_offset)
        J[:3, i] = (p1 - p0) / eps
        dR = R1 @ R0.T
        J[3:, i] = np.array(
            [dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]
        ) / (2 * eps)
    return J
