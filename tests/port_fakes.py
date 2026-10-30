"""Deterministic stand-ins for the T30 port's models and calibration, so the port runs on a Mac without GPUs.

The fake world is a set of parts at known installed-local poses. Depth points are the top face of each part's CAD
bbox (0.5 mm grid), so the port's size measurement sees exactly the CAD footprint and height; tests perturb them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from perception.cell.localize import Located, Zone
from perception.port.backends import PoseFit, WristReading
from perception.port.cad import load_cad
from perception.port.dedup import RawBox
from perception.port.errors import PerceptionPortError
from perception.port.judge import CloseupThresholds
from perception.port.placement import PoseUncertainty
from perception.port.scene import ArmCorridor
from perception.port.size_gate import SizeUncertainty, Status

TABLE_Z = -335.1
DEPTH_SHAPE = (480, 848)
THRESHOLDS = CloseupThresholds(0.8, 0.6, "test fixture")
WRIST_SIZE = SizeUncertainty(0.5, 0.2, Status.MEASURED, "test fixture")
POSE_MEASURED = PoseUncertainty(0.05, 0.2, Status.MEASURED, "test fixture")
POSE_PRELIMINARY = PoseUncertainty(0.8, 0.5, Status.PRELIMINARY, "test fixture")
CORRIDOR = ArmCorridor(80.0, "test fixture")


def world_from_local(x: float, y: float, z: float, yaw_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    ct, st = math.cos(math.radians(tilt_deg)), math.sin(math.radians(tilt_deg))
    tilt = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ tilt
    matrix[:3, 3] = (x, y, z)
    return matrix


def loose_pose(object_id: str, x: float, y: float, yaw: float = 0.0) -> np.ndarray:
    """A part standing on the table on its installed face."""
    return world_from_local(x, y, TABLE_Z - load_cad().get(object_id).bbox_min_mm[2], yaw)


def top_points(object_id: str, pose: np.ndarray, top_offset_mm: float = 0.0, grow_mm: float = 0.0) -> np.ndarray:
    cad = load_cad().get(object_id)
    lo, hi = cad.bbox_min_mm, cad.bbox_max_mm
    xs = np.linspace(lo[0] - grow_mm, hi[0] + grow_mm, int(np.ceil((hi[0] - lo[0] + 2 * grow_mm) / 0.5)) + 1)
    ys = np.linspace(lo[1] - grow_mm, hi[1] + grow_mm, int(np.ceil((hi[1] - lo[1] + 2 * grow_mm) / 0.5)) + 1)
    grid = np.array([(x, y, hi[2] + top_offset_mm) for x in xs for y in ys])
    return grid @ pose[:3, :3].T + pose[:3, 3]


@dataclass
class FakeDetector:
    table: list[RawBox] = field(default_factory=list)
    side: list[RawBox] = field(default_factory=list)

    def __call__(self, image: np.ndarray) -> list[RawBox]:
        return list(self.table if int(image[0, 0, 0]) == 1 else self.side)


@dataclass
class FakeClassifier:
    reading: WristReading = field(default_factory=lambda: WristReading("M1_02", 0.97, 0.01))
    calls: int = 0

    def __call__(self, image: np.ndarray, bbox_px) -> WristReading:
        self.calls += 1
        return self.reading


@dataclass
class FakePose:
    """world_from_local per (object_id, camera); returns world_from_stl as FoundationPose would."""

    poses: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    fit: float = 0.9

    def __call__(self, object_id: str, camera: str, observation, mask: np.ndarray) -> PoseFit:
        if (object_id, camera) not in self.poses:
            raise PerceptionPortError(f"fake FoundationPose has no fit for {object_id} on {camera}")
        local_from_stl = np.array(load_cad().get(object_id).local_from_stl)
        return PoseFit(self.poses[(object_id, camera)] @ local_from_stl, self.fit)


@dataclass
class FakeGeometry:
    """Table boxes keyed by bbox; one wrist point cloud; zones by x (reject zone x > 600)."""

    located: dict[tuple, Located] = field(default_factory=dict)
    table_part_points: dict[tuple, np.ndarray] = field(default_factory=dict)
    wrist_cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    cloud: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    table_surface_z_mm: float = TABLE_Z
    work_area: tuple = ((150.0, 650.0), (-450.0, 450.0))

    def table_locate(self, observation, bbox_px) -> Located:
        if tuple(bbox_px) not in self.located:
            raise PerceptionPortError(f"no depth points inside {bbox_px}")
        return self.located[tuple(bbox_px)]

    def table_points(self, observation, bbox_px) -> np.ndarray:
        return self.table_part_points.get(tuple(bbox_px), np.zeros((0, 3)))

    def table_mask(self, observation, bbox_px) -> np.ndarray:
        return np.ones((400, 640), bool)

    def zone(self, x_mm: float, y_mm: float):
        if x_mm > 600:
            return Zone.REJECT, None
        if y_mm > 400:
            return Zone.OUTSIDE, None
        return Zone.TABLE, None

    def wrist_points(self, observation):
        n = len(self.wrist_cloud)
        pixels = np.stack([np.arange(n) % DEPTH_SHAPE[1], (np.arange(n) // DEPTH_SHAPE[1]) % DEPTH_SHAPE[0]], -1)
        return self.wrist_cloud, pixels

    def wrist_box(self, observation, points_uf0_mm):
        return (100.0, 100.0, 220.0, 210.0)

    def table_cloud(self, observation) -> np.ndarray:
        return self.cloud

    def table_box(self, observation, points_uf0_mm):
        return (10.0, 10.0, 90.0, 90.0)


def observation(t: float = 10.0, tcp=(400.0, 0.0, -168.0, -180.0, 0.0, 0.0), gripper: float = 0.0) -> dict:
    table = np.zeros((36, 64, 3), np.uint8)
    table[0, 0, 0] = 1
    side = np.zeros((36, 64, 3), np.uint8)
    side[0, 0, 0] = 2
    return {"t": t, "tcp_pose": list(tcp), "joints": [0.0, 10.0, -10.0, 0.0, -80.0, 0.0, gripper],
            "images": {"table": table, "side": side, "wrist": np.zeros((48, 85, 3), np.uint8)},
            "depth": {"table": np.zeros((400, 640), np.uint16), "wrist": np.zeros(DEPTH_SHAPE, np.uint16)}}


class MemorySink:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def __call__(self, name: str, image: np.ndarray) -> str:
        ref = f"mem://{len(self.saved)}/{name}"
        self.saved.append(ref)
        return ref
