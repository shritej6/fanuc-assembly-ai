"""Your policy. Edit this file.

act(obs) returns the arm's next target, or a list of targets to play in order. When they have been played, act is
called again with a new observation. act has no time limit: the arm holds still while it runs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import ClassVar, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from assembly.cell_runtime import CellTransport
from robot import CellAdapter, Controller
from robot.adapter import TCP_LIMITS


class _MockTcpHoldController:
    """Hold the observed mock TCP, bootstrapping from the official box midpoint."""

    def reset(self) -> None:
        """The mock-only hold has no rollout state."""

    def act(self, observation: dict) -> list[float]:
        observed_joints = observation.get("joints")
        joints = [] if observed_joints is None else list(observed_joints)
        if len(joints) != 7:
            raise ValueError("mock observation must contain J1..J6 and gripper")
        tcp_pose = observation.get("tcp_pose")
        if tcp_pose is None:
            xyz = [(lower + upper) / 2.0 for lower, upper, _unit in TCP_LIMITS[:3]]
            return [*xyz, -180.0, 0.0, 0.0, joints[6]]
        tcp = list(tcp_pose)
        if len(tcp) != 6:
            raise ValueError("mock tcp_pose must contain x, y, z, w, p, r")
        return [*tcp, joints[6]]


class Policy:
    CAMERAS: ClassVar[list[str]] = ["table", "side", "wrist"]  # the cameras you want, any subset
    IMAGE_SIZE = (640, 360)  # (width, height) of the images you get: the whole 16:9 camera frame, resized
    # (not cropped), as in the sim and the recorded datasets
    IMAGE_FORMAT = "png"  # or "jpeg": about 7x smaller, for a slow connection
    ACTION = "tcp"  # T12 GR00T emits absolute UF0 fingertip targets
    DT = 0.2  # T12 controller cadence: one guarded row per callback
    PREFETCH = False  # True: if act returns many actions at once, plan the next ones while
    # these still play, so the arm does not stop between them
    DEPTH: ClassVar[list[str]] = ["table", "wrist"]

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        controller: Controller | None = None,
        action_mode: Literal["joints", "tcp"] | None = None,
        runtime: Literal["mock", "cell"] = "cell",
        cell_transport: CellTransport = CellTransport.REAL,
    ):
        self.checkpoint = checkpoint
        self.mock_only = checkpoint == "t09-mock"
        if not isinstance(cell_transport, CellTransport):
            raise TypeError("cell_transport must be an explicit CellTransport value")
        self.cell_transport = cell_transport
        if runtime not in ("mock", "cell"):
            raise ValueError(f"unsupported runtime: {runtime!r}")
        if runtime == "cell":
            selected = self._cell_controller(controller, action_mode)
        else:
            selected = self._mock_controller(controller, action_mode)
        self.controller = selected
        self.ACTION = type(self).ACTION
        self.adapter = CellAdapter(selected, action_mode=self.ACTION, dt=lambda: self.DT)

    def _cell_controller(
        self,
        controller: Controller | None,
        action_mode: str | None,
    ) -> Controller:
        if self.mock_only:
            raise ValueError("the t09-mock checkpoint is forbidden outside the local mock runtime")
        if controller is not None or action_mode is not None:
            raise ValueError("the cell runtime does not accept injected controller overrides")
        from assembly.cell_runtime import create_cell_policy

        return create_cell_policy(
            checkpoint=self.checkpoint,
            transport=self.cell_transport,
        )

    def _mock_controller(
        self,
        controller: Controller | None,
        action_mode: str | None,
    ) -> Controller:
        if self.mock_only:
            if controller is not None or action_mode is not None:
                raise ValueError("the t09-mock checkpoint does not accept injected controller overrides")
            from robot.cell.t09_mock_runtime import T09MockController

            return T09MockController()
        if (controller is None) != (action_mode is None):
            raise ValueError("mock controller and action_mode must be injected together")
        if action_mode not in (None, "tcp"):
            raise ValueError("Policy advertises TCP actions; mock injection must also use action_mode='tcp'")
        if controller is not None:
            return controller

        return _MockTcpHoldController()

    def reset(self) -> None:
        """Called at the start of every rollout."""
        self.adapter.reset()

    def close(self) -> None:
        """Release an owned controller transport, if the runtime has one."""
        close = getattr(self.controller, "close", None)
        if callable(close):
            close()

    def act(self, obs: dict):
        """obs["images"]   {camera: (H, W, 3) uint8 RGB}
        obs["joints"]      [J1..J6 deg (the v0.5 twin's convention), gripper 0 open / 1 closed]
        obs["tcp_pose"]    the fingertip [x, y, z mm, w, p, r deg]
        obs["trial"]["instruction"]  the task in words (you may ignore it)
        obs["depth"]       only with DEPTH: {"table": (400, 640), "wrist": (480, 848)} uint16, millimetres along each
                           camera's optical axis, 0 = no depth. calibration.geometry.depth_to_base(load()["table_depth"],
                           depth) gives (N, 3) points in the robot base frame (metres); for the wrist, which moves with
                           the arm, pass joints=obs["joints"] too.

        Return the next target:
          ACTION = "joints"  [J1, J2, J3, J4, J5, J6, gripper]
          ACTION = "tcp"     [x, y, z, w, p, r, gripper]; fingers straight down is w, p, r = -180, 0, R,
                             and the table surface is at z ~ -335 mm. No IK needed.
        Example (tcp): 3 cm above the table at x 450, y 0, fingers down, gripper open
            return [450, 0, -305, -180, 0, 0, 0]
        Return where the arm should be, not a step from where it is.
        """
        return self.adapter.act(obs)
