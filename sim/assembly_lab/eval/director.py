"""Oracle director that issues the planner's steps for one eval layout, one at a time.
Authored by Shritej Gunda.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from policy_learn.deploy.guards import ContactTarget

from ..mimic.catalog import encode_plan_ids
from ..mimic.instructions import instruction
from . import cad

RETREAT_MM = 10.0       # a step is done only once the fingers are open and this far above the contact
SPOT_GAP_MM = 15.0      # clear space around a remembered test-bed spot
MAX_ATTEMPTS = 2        # a step is re-issued once (fresh targets), then the episode stops as failed


@dataclass
class Task:
    """One planner step for the policy."""
    skill: str                     # pick_insert | pull_out | place_free | defect_drop
    uid: str                       # object instance in the env
    case: str                      # mimic CASES tag
    base: str | None = None        # pick_insert: base uid
    spot: tuple | None = None      # (x, y, yaw) for place_free / pull_out / defect_drop
    instruction: str = ""
    grasp_tcp: np.ndarray | None = None
    goal_tcp: np.ndarray | None = None   # seated TCP or set-down TCP
    attempts: int = 0
    started_s: float = 0.0
    grasped: bool = False
    grip_error: dict | None = None
    start_pose: np.ndarray | None = None  # object pose when the step began (for the in-hand grip check)


@dataclass
class Director:
    """Issue the planner's steps for one eval layout."""
    expect: dict
    labels: dict
    spares: dict                   # type id -> spare uid
    task_time_s: float
    plan_source: str = "eval CAD planner (cad.py; no finger-clearance check)"
    events: list = field(default_factory=list)
    queue: list = field(default_factory=list)
    current: Task | None = None
    failed: str | None = None
    spots: list = field(default_factory=list)

    @classmethod
    def for_layout(cls, ev, task_time_s: float) -> Director:
        """Build a director for an eval layout."""
        spares = {cad.type_id(s["uid"]): s["uid"] for s in ev.layout["spares"]}
        return cls(ev.expect, ev.layout["eval"]["labels"], spares, task_time_s)

    # ---------------------------------------------------------------- plan
    def reset(self, obs: dict) -> None:
        """Plan the steps for a new episode."""
        self.events, self.queue, self.current, self.failed, self.spots = [], [], None, None, []
        lab = self.labels
        if "wrong_part" in lab:  # remove only the parts above it, in reverse sheet order, then the wrong part
            prod = next(iter(self.expect["built"]))
            above = [p for p in reversed(cad.product_parts(prod)) if p > f"{prod}_02" and p in self._poses(obs)
                     and self._seated_on(obs, p, self.expect["built"][prod])]
            self.queue += [Task("pull_out", p, "fix_above") for p in above]
            self.queue += [Task("pull_out", w, "fix_above") for w in lab["wrong_part"]]
        order = sorted(self.expect["built"] | self._unbuildable(), key=lambda p: p == self.expect.get("build_order_last"))
        for prod in order:
            base = self.expect["built"].get(prod) or self._base_of(obs, prod)
            for pid in cad.product_parts(prod):
                self.queue += self._part_tasks(pid, base)

    def _unbuildable(self) -> dict:
        lab = self.labels
        bad = lab.get("damaged", []) + lab.get("fallen", []) + lab.get("missing", [])
        return {p[:2]: None for p in bad if p[:2] not in self.expect["built"]}

    def _base_of(self, obs, prod: str) -> str:
        poses = self._poses(obs)
        return next(b for b in cad.base_group(cad.base_type(prod)) if b in poses)

    def _part_tasks(self, pid: str, base: str) -> list:
        lab, out = self.labels, []
        if pid == lab.get("crowded_target"):
            out.append(Task("place_free", lab["blocker"], "crowded"))
        if pid in lab.get("damaged", []):
            out.append(Task("defect_drop", pid, "defect"))
        # a fallen part is rejected in place: the planner has no grip for it (the pad is defined upright)
        if pid in lab.get("damaged", []) + lab.get("fallen", []) + lab.get("missing", []):
            spare = self.spares.get(pid)
            out.append(Task("pick_insert", spare, "spare", base=base) if spare
                        else Task("andon", pid, "spare", base=base))
            return out
        out.append(Task("pick_insert", pid, "nominal", base=base))
        return out

    # ---------------------------------------------------------------- run
    @staticmethod
    def _poses(obs) -> dict:
        return obs["sim"]["object_poses"]

    def _seated_on(self, obs, pid: str, base: str) -> bool:
        poses = self._poses(obs)
        return cad.seat_check(pid, poses[pid], cad.seated_pose(pid, base, poses[base]))[0]

    def step(self, obs: dict, t_s: float) -> Task | None:
        """The step to run now (None: finished, or stopped by a failure or an Andon that ends the episode)."""
        if self.failed:
            return None
        if self.current is not None:
            if self._done(obs, self.current):
                self._log(t_s, "step_done", self.current)
                self.current = None
            elif t_s - self.current.started_s > self.task_time_s:
                self.current.attempts += 1
                self._log(t_s, "step_timeout", self.current)
                if self.current.skill == "pick_insert":
                    from .kpis import jam_check  # local: kpis imports this module

                    self.events[-1]["jam"] = jam_check(self.current.uid, self.current.base, self._poses(obs))
                if self.current.attempts >= MAX_ATTEMPTS:
                    self.failed = f"{self.current.skill} {self.current.uid} failed {MAX_ATTEMPTS} times"
                    return None
                self._start(obs, self.current, t_s)
        while self.current is None and self.queue:
            task = self.queue.pop(0)
            if task.skill == "andon":
                self.events.append({"t": t_s, "type": "andon", "part": task.uid, "base": task.base,
                                    "reason": "part needed, no spare"})
                self.queue = [q for q in self.queue if q.base != task.base]  # that product cannot be finished
                continue
            if task.skill == "pick_insert" and self._seated_on(obs, task.uid, task.base):
                continue  # already installed (fix scenario: parts below the removed ones)
            self._start(obs, task, t_s)
        if self.current is not None:
            self._update_grasp(obs, self.current)
        return self.current

    def _start(self, obs, task: Task, t_s: float) -> None:
        poses, tcp = self._poses(obs), np.asarray(obs["tcp_pose"], float)
        pose = np.asarray(poses[task.uid], float)
        task.start_pose = pose
        task.grasp_tcp = cad.grip_tcp(task.uid, pose, 0, float(tcp[5]))
        tid = cad.type_id(task.uid)
        if task.skill == "pick_insert":
            base_pose = np.asarray(poses[task.base], float)
            goal_obj = cad.seated_pose(tid, task.base, base_pose)
            goal = [base_pose[0], base_pose[1], base_pose[5]]
        else:
            if task.spot is None:
                task.spot = self._defect_spot() if task.skill == "defect_drop" else self._free_spot(obs, tid)
            goal_obj = cad.table_pose(tid, *task.spot)
            goal = list(task.spot)
        task.goal_tcp = cad.carry_tcp(pose, task.grasp_tcp, goal_obj)
        target = [pose[0], pose[1], pose[5], task.grasp_tcp[5], *goal]
        base_type = cad.type_id(task.base) if task.base else None
        task.instruction = instruction(encode_plan_ids(task.skill, tid, base_type, 0, task.case), target)
        task.started_s, task.grasped = t_s, False
        self.current = task
        self._log(t_s, "step_start", task)

    def contact_target(self) -> ContactTarget | None:
        """The guard's contact target: the grasp pad until the part is held, then the seat / set-down spot."""
        t = self.current
        if t is None:
            return None
        if not t.grasped:
            g = t.grasp_tcp
            return ContactTarget(float(g[0]), float(g[1]), float(g[2]), label=f"grasp {t.uid}")
        if t.skill == "defect_drop":
            return None  # dropped from above, no contact floor
        g = t.goal_tcp
        return ContactTarget(float(g[0]), float(g[1]), float(g[2]), label=f"{t.skill} {t.uid}")

    def _update_grasp(self, obs, t: Task) -> None:
        if not t.grasped and cad.held_uid(obs) == t.uid:
            t.grasped = True
            # in-hand pose (object in the TCP frame) vs the planned one: independent of how far the arm moved since
            tcp = cad.pose_to_matrix(np.asarray(obs["tcp_pose"], float))
            got = np.linalg.inv(tcp) @ cad.pose_to_matrix(self._poses(obs)[t.uid])
            want = np.linalg.inv(cad.pose_to_matrix(t.grasp_tcp)) @ cad.pose_to_matrix(t.start_pose)
            err = cad.matrix_to_pose(np.linalg.inv(want) @ got)
            t.grip_error = {"xyz_mm": float(np.linalg.norm(err[:3])), "roll_deg": abs(cad.wrap180(float(err[5])))}

    def _done(self, obs, t: Task) -> bool:
        released = float(obs["joints"][6]) < 0.5 and cad.held_uid(obs) is None
        tcp = np.asarray(obs["tcp_pose"], float)
        if not (released and t.grasped and tcp[2] >= t.goal_tcp[2] + RETREAT_MM):
            return False
        pose = np.asarray(self._poses(obs)[t.uid], float)
        if t.skill == "pick_insert":
            return self._seated_on(obs, t.uid, t.base)
        if t.skill == "defect_drop":
            return cad.in_zone(pose, cad.cell_layout()["reject_zone"])
        return upright_on_spot(t.uid, pose, t.spot)

    def _free_spot(self, obs, tid: str) -> tuple:
        area, poses = cad.cell_layout()["work_area"], self._poses(obs)
        r = float(cad.geometry()[tid]["upright"]["footprint_r"])
        used = [(p[0], p[1], 40.0) for p in poses.values() if cad.in_zone(p, area, 50.0)] + self.spots
        for x in np.arange(area["x"][0] + r, area["x"][1] - r, 10.0):
            for y in np.arange(area["y"][0] + r, area["y"][1] - r, 10.0):
                if all(math.hypot(x - ux, y - uy) >= r + ur + SPOT_GAP_MM for ux, uy, ur in used):
                    self.spots.append((float(x), float(y), r))
                    return float(x), float(y), 0.0
        raise RuntimeError(f"no empty test-bed spot for {tid}")

    def _defect_spot(self) -> tuple:
        z = cad.cell_layout()["reject_zone"]
        return (z["x"][0] + z["x"][1]) / 2.0, (z["y"][0] + z["y"][1]) / 2.0, 0.0

    def _log(self, t_s: float, kind: str, t: Task) -> None:
        self.events.append({"t": round(t_s, 2), "type": kind, "skill": t.skill, "uid": t.uid, "case": t.case,
                            "instruction": t.instruction, "attempt": t.attempts, "grip_error": t.grip_error,
                            "spot": list(t.spot) if t.spot else None})


def spot_error(uid: str, pose, spot) -> dict:
    """Set-down pose vs the remembered spot (yaw-independent tilt; yaw about world z)."""
    want = cad.table_pose(cad.type_id(uid), *spot)
    tilt, yaw = cad.world_delta(pose, want)
    return {"xy": float(math.hypot(pose[0] - want[0], pose[1] - want[1])), "yaw": abs(yaw), "tilt": tilt}


def upright_on_spot(uid: str, pose, spot) -> bool:
    """skills/gates.py spot gates: 1 mm, 2 deg yaw, 1 deg tilt."""
    e = spot_error(uid, pose, spot)
    return e["xy"] <= cad.SPOT_XY_MM and e["yaw"] <= cad.SPOT_YAW_DEG and e["tilt"] <= cad.SPOT_TILT_DEG
