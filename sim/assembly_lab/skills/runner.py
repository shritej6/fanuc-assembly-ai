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
from .jam import InsertJam, InsertWatch
from .motion import DT_S, FREE, MAX_TOOL_TILT_DEG, Motion, MotionError, Segment, start_pose
from .pick import pick
from .place_free import release_and_retreat as release_from_spot
from .place_free import set_down
from .plan import INSERT_TILT_DEG, FreePlacePlan, InsertPlan, PickPlan, PullOutPlan
from .pull_out import pull_out

SOFT_STOP_FRAC = 0.6  # stop pressing at 60 % of a force limit; the guard discards anything over 100 %
TRACKING_MM = 0.1  # a kept demo needs the arm on its held commands (run9 <= 0.05 mm; 53b80ea drooped ~1 mm)
MAX_JOINT_STEP_DEG = 20.0  # obs joints (deg) per 0.1 s row; check10 measured a 352 deg J6 spin over 4 rows
OTHERS_TRACE_MM = 0.1  # log rows where a non-target object is this far from its start pose (dragged stack, nudges)
OTHERS_TRACE_MAX = 300  # entries per episode
SUBSTEP_LOG_MIN_N = 1.0  # keep the per-substep contact of insert rows loaded above this (T12_SUBSTEP_LOG=1)
SOFT_STOP_N = {"normal_N": 6.0, "lateral_N": 3.0}  # or at these, if lower: aligned parts seat at ~0.5 N (agent 2)
SEAT_PHASES = ("slide", "push", "touch_down")  # straight down onto the seat / table
SEAT_STOP_N = 2.0  # seat phases stop at this normal: check10 M1_01 seated at 1-4 N in one 0.05 mm row and hit
#                   17 N in the next (about 260 N/mm), so a 6 N stop never fires between rows
FIRST_SEAT_N = 0.5  # a seat-phase normal above this on a target is first contact with the seat or table
FINE_PRESS_MM = 0.02  # after first seat contact, press in steps this small: ~260 N/mm overshoots ~5 N per row,
#                       not the 6-11.6 N run10 measured with 0.05 mm rows (the 2 N stop still ends the press)
CONTACT_PHASES = ("touch", "level", "twist", "slide", "push", "touch_down")
# every phase: rows are kept only above SUBSTEP_LOG_MIN_N, so free-space rows cost nothing, and run10 trips came on
# untraced rows (first lift row after the close: 20-22 N; carry/set-down after a pull-out of a second-layer part)
SUBSTEP_PHASES = None
# rows where the held part is in the bore or seated, so the fingers may move on it by design (not slip)
CONSTRAINED_PHASES = (*CONTACT_PHASES, "relieve", "retry_lift", "wiggle_out")
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
    watch: InsertWatch | None = None  # jam / twist watch while an insertion attempt runs
    twist_follow_deg: list[float] = field(default_factory=list)  # per insert try: worst part-vs-TCP twist difference
    seat_contacts: list[dict | None] = field(default_factory=list)  # per insert try: offset at first base contact
    tracking_mm: float = 0.0  # worst |obs - commanded TCP| on rows whose command has held still for 3 rows
    substep_trace: list[dict] = field(default_factory=list)  # insert rows with > SUBSTEP_LOG_MIN_N in any substep
    others_trace: list[dict] = field(default_factory=list)  # rows where a non-target object had moved (see _track_others)
    holding: str | None = None  # the part whose post-grasp in-hand pose is the drift reference
    in_hand_drift_mm: dict[str, float] = field(default_factory=lambda: {"carry": 0.0, "contact": 0.0})

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
        before = np.asarray(self.obs["joints"][:6], dtype=float)
        held = len(self.frames) >= 3 and all(np.allclose(f.action[:6], row[:6], atol=1e-6) for f in self.frames[-3:])
        self.obs = self.env.step_tcp(row[None, :], DT_S)
        if held:  # the command has not moved for 3 rows: the arm must be on it
            err = float(np.linalg.norm(checks.tcp_pose(self.obs)[:3] - row[:3]))
            self.tracking_mm = max(self.tracking_mm, err)
        self._log_substeps(row, seg)
        self._track_drift(row, seg)
        self._track_others(seg)
        jump = np.abs(np.asarray(self.obs["joints"][:6], dtype=float) - before)
        if jump.max() > MAX_JOINT_STEP_DEG:  # an IK branch / wrist flip: never let it into a demo
            j = int(jump.argmax())
            raise MotionError(f"J{j + 1} moved {jump[j]:.1f} deg in one row (> {MAX_JOINT_STEP_DEG:.0f}) in {seg.phase} "
                              f"at step {len(self.frames) - 1}")
        self.guard.observe(len(self.frames) - 1, self.obs)
        self.touch.observe(self.obs, self.targets)
        if self.guard.pre_touch_violations:
            v = self.guard.pre_touch_violations[0]
            raise ResetInvalid(f"pre-touch contact: {v.object_id} {v.axis} {v.value:.2f} > {v.limit:.2f} at step {v.step}")
        if self.guard.violations:
            v = self.guard.violations[0]
            raise EpisodeFailed(f"force guard: {v.object_id} {v.axis} {v.value:.2f} > {v.limit:.2f} at step {v.step}")

    def _log_substeps(self, row: np.ndarray, seg: Segment) -> None:
        """T12_SUBSTEP_LOG=1: keep the per-substep contact of contact-phase rows loaded above SUBSTEP_LOG_MIN_N
        (logged before the guard check, so the row that trips the guard is in the trace too)."""
        sub_all = self.obs["sim"].get("substep_contact") or {}
        if not sub_all or (SUBSTEP_PHASES is not None and seg.phase not in SUBSTEP_PHASES):
            return
        for oid, sub in sub_all.items():
            if max(abs(s[0]) for s in sub) > SUBSTEP_LOG_MIN_N:
                self.substep_trace.append({"step": len(self.frames) - 1, "phase": seg.phase, "object": oid,
                                           "tcp_z": round(float(row[2]), 3), "normal_lateral_N": [s[:2] for s in sub]})

    def _track_drift(self, row: np.ndarray, seg: Segment) -> None:
        """Worst move of the held part in the TCP frame since the grasp (slip watch): free-space carry rows apart
        from contact rows, where the part is pushed in the fingers on purpose. A grip-open command ends the hold."""
        if row[6] < 0.5:
            self.holding = None
        if self.holding is None:
            return
        rel = checks.in_hand(self.obs, self.holding)[:3, 3] - self.in_hand[self.holding][:3, 3]
        key = "contact" if seg.phase in CONSTRAINED_PHASES else "carry"
        self.in_hand_drift_mm[key] = max(self.in_hand_drift_mm[key], round(float(np.linalg.norm(rel)), 4))

    def _track_others(self, seg: Segment) -> None:
        """Rows where an object the episode must not move is off its start pose by > OTHERS_TRACE_MM (main: does a
        pulled second-layer part drag the part below it up with it?). The others_moved gate still decides at the end."""
        if len(self.others_trace) >= OTHERS_TRACE_MAX:
            return
        for oid, pose in self.obs["sim"]["object_poses"].items():
            if oid in self.targets or oid not in self.start_poses:
                continue
            d = np.asarray(pose[:3], dtype=float) - np.asarray(self.start_poses[oid][:3], dtype=float)
            if np.linalg.norm(d) > OTHERS_TRACE_MM:
                self.others_trace.append({"step": len(self.frames) - 1, "phase": seg.phase, "object": oid,
                                          "d_mm": [round(float(v), 3) for v in d]})

    def _too_hard(self, phase: str = "") -> bool:
        """Contact building up during an insert (part on the rim, not in the hole): stop pressing and re-check."""
        contact = self.obs["sim"].get("contact_peak", self.obs["sim"].get("contact", {}))
        for oid, c in contact.items():
            if oid in self.guard.exempt:
                continue
            for key, lim_key in (("normal_N", "axial_N"), ("lateral_N", "lateral_N")):
                lim = self.guard.limit(oid, lim_key)
                seat = key == "normal_N" and phase in SEAT_PHASES and oid in self.targets  # the part going down
                stop = SEAT_STOP_N if seat else SOFT_STOP_N[key]
                stop = stop if lim is None else min(stop, SOFT_STOP_FRAC * lim)
                if abs(float(c.get(key, 0.0))) > stop:
                    return True
        return False

    def _watch_insert(self, row: np.ndarray, seg: Segment) -> None:
        if self.watch is None:
            return
        oid = self.watch.plan.object_id
        contact = self.obs["sim"].get("contact_peak", self.obs["sim"].get("contact", {})).get(oid, {})
        try:
            self.watch.observe(self.obs, row, seg.phase, abs(float(contact.get("normal_N", 0.0))))
        except InsertJam as e:
            self.events.append(f"{e} at step {len(self.frames) - 1}")
            raise EpisodeFailed(str(e)) from e

    def _entry_try(self, plan: InsertPlan, segs: list[Segment]) -> None:
        """One insertion attempt under the jam / twist watch; its worst twist-follow error goes into the log."""
        self.watch = InsertWatch(plan)
        try:
            self.run(segs, soft_stop=True)
        finally:
            self.twist_follow_deg.append(round(self.watch.max_follow_err_deg, 4))
            self.seat_contacts.append(self.watch.first_contact)
            self.watch = None

    def run(self, segments: list[Segment], stop_after: str | None = None, soft_stop: bool = False) -> list[Segment]:
        """Execute segments in order; return the ones not run yet when `stop_after` phase completes.

        soft_stop: in the insert contact phases, stop at the first step whose contact passes SOFT_STOP_FRAC of a limit
        (the rest of the attempt is skipped and the seated check decides what happens next).
        """
        seated_touch = False  # each run is one insert try or set-down: fine pressing starts at its first seat contact
        for i, seg in enumerate(segments):
            self._open_subtask(seg)
            for planned in seg.rows:
                fine = seated_touch and seg.phase in SEAT_PHASES
                for row in (self._fine_rows(planned) if fine else (planned,)):
                    self._step(row, seg)
                    self.pose, self.gripper = row[:6].copy(), float(row[6])
                    self._watch_insert(row, seg)
                    if soft_stop and seg.phase in CONTACT_PHASES and self._too_hard(seg.phase):
                        self.events.append(f"soft stop in {seg.phase} at step {len(self.frames) - 1}")
                        return []
                    if not seated_touch and seg.phase in SEAT_PHASES and self._seat_touched():
                        seated_touch = True
                        self.events.append(f"seat contact in {seg.phase} at step {len(self.frames) - 1}: "
                                           f"pressing at {FINE_PRESS_MM} mm/row")
            if stop_after is not None and seg.phase == stop_after:
                return segments[i + 1:]
        return []

    def _fine_rows(self, row: np.ndarray) -> list[np.ndarray]:
        """Split one planned row into steps of at most FINE_PRESS_MM from the current pose (angles wrap)."""
        d = np.asarray(row[:6], dtype=float) - self.pose
        d[3:] = wrap_deg(d[3:])
        n = max(1, int(np.ceil(np.linalg.norm(d[:3]) / FINE_PRESS_MM - 1e-9)))
        return [np.concatenate([self.pose + d * (k / n), row[6:]]) for k in range(1, n)] + [row]

    def _seat_touched(self) -> bool:
        contact = self.obs["sim"].get("contact_peak", self.obs["sim"].get("contact", {}))
        return any(abs(float(c.get("normal_N", 0.0))) > FIRST_SEAT_N for oid, c in contact.items()
                   if oid in self.targets and oid not in self.guard.exempt)

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
        self.holding = plan.object_id
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
        self._entry_try(plan, insert(plan, self.pose, self.gripper, self._offset(plan, plan.object_id, misalign),
                                     learn=not staged))
        ok, err = self._seated(plan)
        if not ok:
            self.events.append(f"{plan.object_id} not seated on try 1: {_fmt(err)}")
            self.run(retry_lift(plan, self.pose, self.gripper))
            self.run(retry_clear(plan, self.pose, self.gripper))
            self._entry_try(plan, retry_entry(plan, self.pose, self.gripper, self._offset(plan, plan.object_id)))
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
        self.holding = plan.object_id
        self.mark("pulled")

    def do_place_free(self, plan: FreePlacePlan) -> None:
        t_tcp_obj = checks.in_hand(self.obs, plan.object_id)
        expected = checks.expected_object(plan.spot_tcp, t_tcp_obj)
        self.run(set_down(plan, self.pose, self.gripper), soft_stop=True)  # soft stop: release where it touched
        if plan.kind == "test_bed":
            self.mark("placed")
            self.run(release_from_spot(plan, self.pose, self.gripper))
        else:
            rest = self.run(release_from_spot(plan, self.pose, self.gripper), stop_after="release")
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
        """Episode-level gates: the arm tracked its commands, nothing else moved, the robot touched only the target."""
        self.gate_log.check("tracking_mm", self.tracking_mm, TRACKING_MM)
        self.gate_log.require(("tracking_mm",))
        gates.others_moved(self.gate_log, self.start_poses, self.obs["sim"]["object_poses"], self.targets)
        self.touch.finish(self.gate_log, require=not self.measure_only)

    def go_home(self) -> None:
        m = Motion(self.pose, self.gripper, "home", "home", "world")
        m.move("home", self.home, FREE)
        self.run(m.segments)


def _fmt(err: dict) -> str:
    return ", ".join(f"{k} {v:+.2f}" for k, v in err.items())
