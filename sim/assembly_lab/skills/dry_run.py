"""KINEMATIC TEST DOUBLE: no physics. It exists so the skill -> runner -> HDF5 pipeline can be exercised on any host
before the Isaac Lab env (agent 1) and the planner (agent 2) exist. Nothing it produces is a demo or a measurement:
`record.py --backend kinematic` refuses to write under ~/t12_data/demos and stamps every file KINEMATIC_DRY_RUN.

The part attaches when the fingers close within 1.5 mm of its grip TCP, follows the TCP rigidly, and on release rests
on its seat (if it is within the seat's capture tolerance) or on top of the pegs / the table. Contact force is a
spring on how far the TCP pushes a part below its support (10 N/mm), so the force guard path runs end to end.
"""

from __future__ import annotations

import math

import numpy as np

from .geometry import matrix_to_pose, pose_to_matrix

TABLE_Z = -335.0  # sdg/configs/cell_layout.yaml table_z
CAPTURE_MM, CAPTURE_DEG = 0.7, 2.5  # within this the chamfer guides the part home (dry run only)
SPRING_N_PER_MM = 10.0
HOME_TCP = [400.0, 0.0, 100.0, -180.0, 0.0, 0.0]


def obj_to_tcp(tcp_local_mm, roll_offset_deg: float) -> np.ndarray:
    """Object -> grip TCP: offset to the pad centre, fingers straight down, turned by the CAD roll offset."""
    return pose_to_matrix([*tcp_local_mm, 180.0, 0.0, roll_offset_deg])


class KinematicEnv:
    def __init__(self, objects: dict, seats: dict):
        """objects: {id: {"pose": [x,y,z,w,p,r], "t_obj_tcp": 4x4}}; seats: {id: {"pose": seated pose, "engage": mm}}"""
        self.spec = objects
        self.seats = seats
        self.reset()

    def reset(self, layout=None) -> dict:
        self.tcp = np.array(HOME_TCP, dtype=float)
        self.grip = 0.0
        self.poses = {k: np.asarray(v["pose"], dtype=float) for k, v in self.spec.items()}
        self.held: tuple[str, np.ndarray] | None = None
        self.force: dict[str, float] = {}
        return self.observe()

    def observe(self) -> dict:
        contact = {k: {"normal_N": self.force.get(k, 0.0), "lateral_N": 0.0, "torque_Nm": 0.0} for k in self.poses}
        return {"joints": [0.0] * 6 + [self.grip], "tcp_pose": list(self.tcp),
                "instruction": "", "images": {},
                "sim": {"object_poses": {k: list(v) for k, v in self.poses.items()}, "contact_peak": contact,
                        "gripper_obj": self.held[0] if self.held else None,
                        "robot_touch": {k: 1.0 if self.held and self.held[0] == k else 0.0 for k in self.poses}}}

    def _support_z(self, oid: str, pose: np.ndarray) -> float:
        seat = self.seats.get(oid)
        if seat is None:
            return pose[2] if self.held is None else TABLE_Z
        s = np.asarray(seat["pose"], dtype=float)
        dxy = math.hypot(pose[0] - s[0], pose[1] - s[1])
        dyaw = abs((pose[5] - s[5] + 180.0) % 360.0 - 180.0)
        if dxy < 25.0:  # over the base
            return s[2] if dxy <= CAPTURE_MM and dyaw <= CAPTURE_DEG else s[2] + seat["engage"]
        return TABLE_Z

    def step_tcp(self, rows, dt: float) -> dict:
        for row in np.asarray(rows, dtype=float):
            self._row(row)
        return self.observe()

    def _row(self, row: np.ndarray) -> None:
        self.tcp = row[:6].copy()
        closing, opening = row[6] > 0.5 >= self.grip, row[6] <= 0.5 < self.grip
        self.grip = float(row[6] > 0.5)
        self.force = {}
        if closing and self.held is None:
            for oid, spec in self.spec.items():
                tcp_at = matrix_to_pose(pose_to_matrix(self.poses[oid]) @ spec["t_obj_tcp"])
                if np.linalg.norm(tcp_at[:3] - self.tcp[:3]) <= 1.5:
                    self.held = (oid, np.linalg.inv(pose_to_matrix(self.tcp)) @ pose_to_matrix(self.poses[oid]))
                    break
        if self.held is not None:
            oid, t = self.held
            pose = matrix_to_pose(pose_to_matrix(self.tcp) @ t)
            floor = self._support_z(oid, pose)
            if pose[2] < floor:
                self.force[oid] = SPRING_N_PER_MM * (floor - pose[2])
                pose[2] = floor
            self.poses[oid] = pose
            if opening:
                self.held = None
                self.poses[oid][2] = max(self._support_z(oid, pose), TABLE_Z)


def demo_world(part_xy_yaw=(400.0, 150.0, 30.0), base_xy_yaw=(450.0, -60.0, 0.0)):
    """One base F01 and its first part M1_02 with the CAD numbers from cad/grasp_frames.json (dry run only)."""
    t_tcp = obj_to_tcp([0.0, 0.0, 11.4], 90.0)  # M1_02 grip: tcp_local_mm, roll_offset_deg
    px, py, pyaw = part_xy_yaw
    bx, by, byaw = base_xy_yaw
    base = np.array([bx, by, TABLE_Z, 0.0, 0.0, byaw])
    seated = matrix_to_pose(pose_to_matrix(base) @ pose_to_matrix([0.0, 0.0, 7.0, 0.0, 0.0, 0.0]))  # installed_origin
    objects = {"M1_02": {"pose": [px, py, TABLE_Z, 0.0, 0.0, pyaw], "t_obj_tcp": t_tcp},
               "F01": {"pose": list(base), "t_obj_tcp": np.eye(4)}}
    seats = {"M1_02": {"pose": seated, "engage": 2.5}}
    return objects, seats


def demo_plans(objects: dict, seats: dict):
    """Planner stand-in for the dry run: grasp at the CAD pad, seat at the CAD installed pose."""
    from .plan import FreePlacePlan, InsertPlan, PickPlan, PullOutPlan

    t = objects["M1_02"]["t_obj_tcp"]
    grasp = matrix_to_pose(pose_to_matrix(objects["M1_02"]["pose"]) @ t)
    seated_obj = np.asarray(seats["M1_02"]["pose"], dtype=float)
    seated_tcp = matrix_to_pose(pose_to_matrix(seated_obj) @ t)
    pick = PickPlan("M1_02", grasp, 7.0, grasp[2] + 60.0, grasp[2] + 60.0)
    ins = InsertPlan("M1_02", "F01", seated_tcp, seated_obj, 2.5, seated_tcp[2] + 40.0)
    pull = PullOutPlan("M1_02", seated_tcp, 7.0, 2.5, seated_tcp[2] + 40.0)
    spot = FreePlacePlan("M1_02", np.array([300.0, 200.0, TABLE_Z + 11.4, -180.0, 0.0, 0.0]), "test_bed",
                         TABLE_Z + 80.0)
    defect = FreePlacePlan("M1_02", np.array([250.0, 355.0, TABLE_Z + 11.4, -180.0, 0.0, 0.0]), "defect",
                           TABLE_Z + 80.0, drop_mm=15.0)
    return pick, ins, pull, spot, defect
