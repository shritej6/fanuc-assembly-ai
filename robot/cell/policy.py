"""Your policy. Edit this file.

act(obs) returns the arm's next target, or a list of targets to play in order. When they have been played, act is
called again with a new observation. act has no time limit: the arm holds still while it runs.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot import CellAdapter, Controller, JointPositionHoldController


class Policy:
    CAMERAS: ClassVar[list[str]] = ["table", "side", "wrist"]  # the cameras you want, any subset
    IMAGE_SIZE = (640, 360)  # (width, height) of the images you get: the whole 16:9 camera frame, resized
    # (not cropped), as in the sim and the recorded datasets
    IMAGE_FORMAT = "png"  # or "jpeg": about 7x smaller, for a slow connection
    ACTION = "joints"  # what act returns: "joints" or "tcp" (see below)
    DT = 0.1  # seconds between absolute targets
    PREFETCH = False  # True: if act returns many actions at once, plan the next ones while
    # these still play, so the arm does not stop between them
    DEPTH: ClassVar[list[str]] = []  # ["table"] and/or ["wrist"]: also those cameras' depth, in obs["depth"]

    def __init__(
        self,
        checkpoint: str | None = None,
        *,
        controller: Controller | None = None,
        action_mode: Literal["joints", "tcp"] | None = None,
        runtime: Literal["mock", "cell"] = "cell",
        cell_runtime_factory: Callable[[], Controller] | None = None,
    ):
        self.checkpoint = checkpoint
        self.mock_only = checkpoint == "t09-mock"
        self.cell_enabled = checkpoint == "t09-cell"
        if runtime not in ("mock", "cell"):
            raise ValueError(f"unsupported runtime: {runtime!r}")
        if self.mock_only:
            controller, action_mode = self._mock_controller(runtime, controller, action_mode)
        if self.cell_enabled:
            controller, action_mode = self._cell_controller(runtime, controller, action_mode, cell_runtime_factory)
        if controller is None and action_mode is not None:
            raise ValueError("action_mode requires an explicitly injected controller")
        if controller is not None and action_mode is None:
            raise ValueError("an injected controller requires an explicit action_mode")
        selected = controller or JointPositionHoldController()
        self.controller = selected
        self.ACTION = action_mode or type(self).ACTION
        self.adapter = CellAdapter(selected, action_mode=self.ACTION, dt=lambda: self.DT)

    @staticmethod
    def _mock_controller(
        runtime: str,
        controller: Controller | None,
        action_mode: str | None,
    ) -> tuple[Controller, Literal["tcp"]]:
        if runtime != "mock":
            raise ValueError("the t09-mock checkpoint is forbidden outside the local mock runtime")
        if controller is not None or action_mode is not None:
            raise ValueError("the t09-mock checkpoint does not accept injected controller overrides")
        from robot.cell.t09_mock_runtime import T09MockController

        return T09MockController(), "tcp"

    @staticmethod
    def _cell_controller(
        runtime: str,
        controller: Controller | None,
        action_mode: str | None,
        factory: Callable[[], Controller] | None,
    ) -> tuple[Controller, Literal["tcp"]]:
        if runtime != "cell":
            raise ValueError("the t09-cell checkpoint is forbidden outside the real cell runtime")
        if controller is not None or action_mode is not None:
            raise ValueError("the t09-cell checkpoint does not accept controller overrides")
        if factory is None:
            raise RuntimeError("t09-cell requires a configured production cell runtime factory")
        return factory(), "tcp"

    def reset(self) -> None:
        """Called at the start of every rollout."""
        self.adapter.reset()

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
