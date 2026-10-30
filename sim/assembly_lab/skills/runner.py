"""Run skills closed-loop in the sim env and record every step for the HDF5 demo.

The skills are open-loop row generators; the runner adds what needs sim feedback:
- grasp check after the slow check lift (object rose with the fingers),
- seated check after the push, the single retry (lift 3 mm, re-measure, re-align), then pull out for inspection,
- the force guard on every step (abort early on the first violation: that demo is discarded anyway).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import checks, frames, gates
from .force_guard import ForceGuard
from .geometry import matrix_to_pose, pose_to_matrix, wrap_deg
from .insert import insert, release_and_retreat, retry_clear, retry_entry, retry_lift
from .motion import DT_S, FREE, MAX_TOOL_TILT_DEG, Motion, Segment, start_pose
from .pick import pick
from .place_free import place_free
from .plan import INSERT_TILT_DEG, FreePlacePlan, InsertPlan, PickPlan, PullOutPlan
from .pull_out import pull_out

SOFT_STOP_FRAC = 0.6  # stop pressing at 60 % of a force limit; the guard discards anything over 100 %
CONTACT_PHASES = ("touch", "level", "twist", "slide", "push")
MAX_LEVEL_CORRECTION_DEG = MAX_TOOL_TILT_DEG - INSERT_TILT_DEG  # leaves room for the 3 deg entry lean


class EpisodeFailed(RuntimeError):
    """A success check or the force guard failed: the episode is discarded."""


class ResetInvalid(RuntimeError):
    """The scene was bad before the robot did anything (not settled, or over a force limit before first contact):
    not kept and not counted as a robot failure (main, 2026-10-30)."""


@dataclass
class Frame:
    obs: dict  # observation before the action
    action: np.ndarray  # act_tcp row
    seg: Segment
    instruction: int


@dataclass
class Subtask:
    name: str
    skill: str
    object_ref: str
    start: int
    end: int = -1


@dataclass
class EpisodeRecorder:
    env: object
    guard: ForceGuard
    obs: dict
    frames: list[Frame] = field(default_factory=list)
    subtasks: list[Subtask] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    in_hand: dict[str, np.ndarray] = field(default_factory=dict)
    seat_tries: list[dict] = field(default_factory=list)
    level_corrections: list[dict] = field(default_factory=list)  # per insert try: in-hand tilt and the TCP fix
    marks: dict[str, int] = field(default_factory=dict)  # Mimic term signal -> first step after that subtask ended
    frame_defs: dict[str, tuple[str | None, np.ndarray]] = field(default_factory=dict)  # Mimic object_ref frames
    meta: dict = field(default_factory=dict)  # plan_ids, plan_target, skill, case, product
    allow_touch_proxy: bool = False
    measure_only: bool = False  # log the touch gate without requiring it (robot_touch not verified yet)
    gate_log: gates.GateLog = field(default_factory=gates.GateLog)
    targets: set[str] = field(default_factory=set)  # objects this episode is allowed to touch and move

    def __post_init__(self):
        self.pose = checks.tcp_pose(self.obs)
        self.gripper = float(np.asarray(self.obs["joints"])[6] > 0.5)
        self.home = start_pose(self.pose)  # the scan pose every episode returns to, inside the box
        settled = self.obs["sim"].get("settled")
        self.settled = "unknown" if settled is None else bool(settled)
        if settled is not None and not settled:
            raise ResetInvalid("the scene did not settle after reset (obs sim.settled is False)")
        self.start_poses = dict(self.obs["sim"]["object_poses"])
        self.touch = gates.TouchMonitor(self.allow_touch_proxy)

    # --- low level ---------------------------------------------------------------------------------------------
    def say(self, instruction: str) -> None:
        self.instructions.append(instruction)

    def mark(self, signal: str) -> None:
        self.marks.setdefault(signal, len(self.frames))

    def define_frame(self, name: str, object_id: str | None, tcp_pose) -> None:
        """A planner TCP frame that rides on `object_id` (None: fixed in the robot frame), as Mimic's object_ref."""
        t = pose_to_matrix(tcp_pose)
        if object_id is not None:
            t = np.linalg.inv(pose_to_matrix(checks.object_pose(self.obs, object_id))) @ t
        self.frame_defs[name] = (object_id, t)

    def frame_pose(self, obs: dict, name: str) -> np.ndarray:
        oid, t = self.frame_defs[name]
        return t if oid is None else pose_to_matrix(np.asarray(obs["sim"]["object_poses"][oid], dtype=float)) @ t

    def _step(self, row: np.ndarray, seg: Segment) -> None:
        self.frames.append(Frame(self.obs, row, seg, max(0, len(self.instructions) - 1)))
        self.obs = self.env.step_tcp(row[None, :], DT_S)
        self.guard.observe(len(self.frames) - 1, self.obs)
        self.touch.observe(self.obs, self.targets)
        if self.guard.pre_touch_violations:
            v = self.guard.pre_touch_violations[0]
            raise ResetInvalid(f"pre-touch contact: {v.object_id} {v.axis} {v.value:.2f} > {v.limit:.2f} at step {v.step}")
        if self.guard.violations:
            v = self.guard.violations[0]
            raise EpisodeFailed(f"force guard: {v.object_id} {v.axis} {v.value:.2f} > {v.limit:.2f} at step {v.step}")

    def _too_hard(self) -> bool:
        """Contact building up during an insert (part on the rim, not in the hole): stop pressing and re-check."""
        contact = self.obs["sim"].get("contact_peak", self.obs["sim"].get("contact", {}))
        for oid, c in contact.items():
            if oid in self.guard.exempt:
                continue
            for key, lim_key in (("normal_N", "axial_N"), ("lateral_N", "lateral_N")):
                lim = self.guard.limit(oid, lim_key)
                if lim is not None and abs(float(c.get(key, 0.0))) > SOFT_STOP_FRAC * lim:
                    return True
        return False

    def run(self, segments: list[Segment], stop_after: str | None = None, soft_stop: bool = False) -> list[Segment]:
        """Execute segments in order; return the ones not run yet when `stop_after` phase completes.

        soft_stop: in the insert contact phases, stop at the first step whose contact passes SOFT_STOP_FRAC of a limit
        (the rest of the attempt is skipped and the seated check decides what happens next).
        """
        for i, seg in enumerate(segments):
            self._open_subtask(seg)
            for row in seg.rows:
                self._step(row, seg)
                self.pose, self.gripper = row[:6].copy(), float(row[6])
                if soft_stop and seg.phase in CONTACT_PHASES and self._too_hard():
                    self.events.append(f"soft stop in {seg.phase} at step {len(self.frames) - 1}")
                    return []
            if stop_after is not None and seg.phase == stop_after:
                return segments[i + 1:]
        return []

    def _open_subtask(self, seg: Segment) -> None:
        if not self.subtasks or self.subtasks[-1].name != seg.subtask:
            if self.subtasks:
                self.subtasks[-1].end = len(self.frames)
            self.subtasks.append(Subtask(seg.subtask, seg.skill, seg.object_ref, len(self.frames)))

    def close(self) -> None:
        if self.subtasks:
            self.subtasks[-1].end = len(self.frames)

    # --- skills with feedback ---------------------------------------------------------------------------------
    def do_pick(self, plan: PickPlan, learn: bool = True) -> None:
        self.targets.add(plan.object_id)
        planned = self._planned_grip(plan.object_id, plan.grasp_tcp)
        segs = pick(plan, self.pose, self.gripper, learn=learn)
        rest = self.run(segs, stop_after="close")
        gates.grip_gates(self.gate_log, self.obs, plan.object_id, planned, plan.grasp_tcp[5])
        before = self.obs
        rest = self.run(rest, stop_after="lift_check")
        ok, why = checks.grasp_ok(before, self.obs, plan.object_id)
        if not ok:
            raise EpisodeFailed(f"grasp check: {why}")
        self.run(rest)
        self.in_hand[plan.object_id] = checks.in_hand(self.obs, plan.object_id)
        self.mark("grasped")

    def _offset(self, plan: InsertPlan, oid: str, extra=(0.0, 0.0, 0.0)) -> np.ndarray:
        """TCP correction (dx, dy, dyaw, w, p) so the part as held lands on its CAD seated pose: w, p level the part
        itself (a part pivoted in the fingers is squared up before it enters the bore), dx, dy include the swing.
        The level correction is capped so the tool stays within 6 deg of straight down with the 3 deg entry lean on
        top; a part pivoted further than that is a failed grasp, not something to steer around."""
        t_tcp_obj = checks.in_hand(self.obs, oid)
        need = matrix_to_pose(pose_to_matrix(plan.seated_object_pose) @ np.linalg.inv(t_tcp_obj))
        d = need - plan.seated_tcp
        held = checks.object_pose(self.obs, oid)
        level = gates.axis_tilt_deg(need, plan.seated_tcp)
        entry = {"try": len(self.level_corrections) + 1,
                 "in_hand_tilt_deg": round(gates.axis_tilt_deg(held, plan.seated_object_pose,
                                                               frames.installed_up(oid)), 4),
                 "dw_deg": round(float(wrap_deg(need[3] - plan.seated_tcp[3])), 4),
                 "dp_deg": round(float(need[4] - plan.seated_tcp[4]), 4), "level_deg": round(level, 4),
                 "dx_mm": round(float(d[0]), 4), "dy_mm": round(float(d[1]), 4),
                 "dyaw_deg": round(float(wrap_deg(d[5])), 4)}
        self.level_corrections.append(entry)
        self.events.append(f"{oid} in hand before entry: tilt {entry['in_hand_tilt_deg']:.2f} deg, TCP level "
                           f"correction {level:.2f} deg (w {entry['dw_deg']:+.2f} p {entry['dp_deg']:+.2f})")
        if level > MAX_LEVEL_CORRECTION_DEG:
            raise EpisodeFailed(f"{oid} pivoted {level:.2f} deg in the fingers: more than the "
                                f"{MAX_LEVEL_CORRECTION_DEG:.1f} deg level correction the 6 deg tool limit allows")
        return np.array([d[0] + extra[0], d[1] + extra[1], wrap_deg(d[5]) + extra[2], need[3], need[4]])

    def _seated(self, plan: InsertPlan) -> tuple[bool, dict]:
        """Seat check while still holding (decides the retry); every try's measured error is kept in seat_tries."""
        tol = checks.SeatTolerance(yaw_deg=gates.seat_yaw_limit(plan.yaw_tol_deg))
        ok, err = checks.seated_ok(checks.object_pose(self.obs, plan.object_id), plan.seated_object_pose, tol,
                                   frames.installed_up(plan.object_id))
        self.seat_tries.append({"ok": ok, **{k: round(float(v), 4) for k, v in err.items()}})
        return ok, err

    def do_insert(self, plan: InsertPlan, misalign=(0.0, 0.0, 0.0), pull_plan: PullOutPlan | None = None) -> str:
        """Returns "seated" or "needs_inspection" (pulled out after the retry failed)."""
        staged = any(misalign)  # an injected first-attempt error is staging: masked out of training
        self.run(insert(plan, self.pose, self.gripper, self._offset(plan, plan.object_id, misalign), learn=not staged),
                 soft_stop=True)
        ok, err = self._seated(plan)
        if not ok:
            self.events.append(f"{plan.object_id} not seated on try 1: {_fmt(err)}")
            self.run(retry_lift(plan, self.pose, self.gripper))
            self.run(retry_clear(plan, self.pose, self.gripper))
            self.run(retry_entry(plan, self.pose, self.gripper, self._offset(plan, plan.object_id)), soft_stop=True)
            ok, err = self._seated(plan)
        if not ok:
            self.events.append(f"{plan.object_id} not seated on try 2: {_fmt(err)}; pulling out for inspection")
            if pull_plan is None:
                raise EpisodeFailed(f"{plan.object_id} would not seat and no pull-out plan was given")
            self.run(pull_out(pull_plan, self.pose, self.gripper, holding=True))
            return "needs_inspection"
        self.mark("seated")
        self.run(release_and_retreat(plan, self.pose))
        gates.seat_gates(self.gate_log, checks.object_pose(self.obs, plan.object_id), plan.seated_object_pose,
                         plan.yaw_tol_deg, frames.installed_up(plan.object_id))
        return "seated"

    def do_pull_out(self, plan: PullOutPlan) -> None:
        self.targets.add(plan.object_id)
        planned = self._planned_grip(plan.object_id, plan.grasp_tcp)
        segs = pull_out(plan, self.pose, self.gripper)
        rest = self.run(segs, stop_after="close")
        gates.grip_gates(self.gate_log, self.obs, plan.object_id, planned, plan.grasp_tcp[5])
        self.mark("grasped")
        before = self.obs
        self.run(rest)
        ok, why = checks.grasp_ok(before, self.obs, plan.object_id)
        if not ok:
            raise EpisodeFailed(f"pull-out: {why}")
        self.in_hand[plan.object_id] = checks.in_hand(self.obs, plan.object_id)
        self.mark("pulled")

    def do_place_free(self, plan: FreePlacePlan) -> None:
        t_tcp_obj = checks.in_hand(self.obs, plan.object_id)
        expected = checks.expected_object(plan.spot_tcp, t_tcp_obj)
        segs = place_free(plan, self.pose, self.gripper)
        if plan.kind == "test_bed":
            rest = self.run(segs, stop_after="set_down")
            self.mark("placed")
        else:
            rest = self.run(segs, stop_after="release")
            self.mark("dropped")
        self.run(rest)
        actual = checks.object_pose(self.obs, plan.object_id)
        if plan.kind == "test_bed":
            gates.spot_gates(self.gate_log, actual, expected)
            ok, why = True, "ok"
        else:
            self.guard.exempt.add(plan.object_id)
            ok, why = checks.in_defect_area(actual, plan.spot_tcp)
        if not ok:
            raise EpisodeFailed(f"place_free {plan.object_id}: {why}")

    def _planned_grip(self, oid: str, grasp_tcp) -> np.ndarray:
        """Object -> planned TCP, from the part's pose when the plan was made."""
        return np.linalg.inv(pose_to_matrix(checks.object_pose(self.obs, oid))) @ pose_to_matrix(grasp_tcp)

    def finish_gates(self) -> None:
        """Episode-level gates: nothing else moved, and the robot touched only the target."""
        gates.others_moved(self.gate_log, self.start_poses, self.obs["sim"]["object_poses"], self.targets)
        self.touch.finish(self.gate_log, require=not self.measure_only)

    def go_home(self) -> None:
        m = Motion(self.pose, self.gripper, "home", "home", "world")
        m.move("home", self.home, FREE)
        self.run(m.segments)


def _fmt(err: dict) -> str:
    return ", ".join(f"{k} {v:+.2f}" for k, v in err.items())
