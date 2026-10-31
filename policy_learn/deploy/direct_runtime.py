"""Fallback 14:00 runtime: GR00T drives every motion, step by step through placement.yaml, guarded, hold-only.

    from policy_learn.deploy.direct_runtime import DirectGrootRuntime
    policy = DirectGrootRuntime("groot://127.0.0.1:5555")    # the official Policy interface (ACTION = "tcp")

What it does on each act(obs):
1. On the first act of a trial it picks the rollout (DIRECT_ROLLOUT, else the trial prompt matched against
   placement.yaml, else obs["trial"]["index"], else the count of resets) and plans every pick-insert step
   (direct_plan.plan_rollout): products in prompt order, parts in assembly-sheet order.
2. It gives GR00T the current step through GrootPolicy.set_plan (the training instruction format) and the guard the
   step's contact target: the grasp pad until the part is held, then the seated TCP (as sim/assembly_lab/eval/director.py).
3. It returns GrootPolicy.act(obs): the model's chunk after ChunkGuard. A refused chunk is one hold row.
4. It advances when the observed TCP and gripper show the step done. This is the sim eval's rule (director._done:
   fingers open, the part was grasped, TCP at least RETREAT_MM above the goal) with the oracle parts replaced by what
   the arm reports: "grasped" = the gripper closed within the grasp tolerance of the planned pad, and the part counts
   as released on the seat only if the gripper opened within the release tolerance of the seated TCP.

It never moves the arm itself. Every failure ends the trial with the arm held where it is (current pose, gripper as
it is), for the rest of the trial: a step over its time budget, the gripper closing away from the pad or opening away
from the seat (dropped or misplaced part), the TCP leaving the step's corridor, an empty gripper (only when the cell
reports a width), STUCK_HOLDS guard refusals in a row, or a plan the planner refuses.
"""
from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from policy_learn.deploy.direct_plan import PlacementError, Rollout, Step, load_placement, plan_rollout
from policy_learn.deploy.groot_policy import ChunkModel, GrootPolicy
from policy_learn.deploy.guards import ContactTarget, GuardLimits, hold_row

log = logging.getLogger(__name__)
RETREAT_MM = 10.0  # sim/assembly_lab/eval/director.py RETREAT_MM: done only once this far above the goal


@dataclass(frozen=True)
class Budgets:
    """PRELIMINARY engineering choices, not measured on the real cell. Tolerances cover hand placement error."""

    pick_s: float = 60.0            # from the step's first act to the grasp
    insert_s: float = 90.0          # from the grasp to fingers open and RETREAT_MM above the seat
    grasp_xy_mm: float = 15.0       # gripper closing farther than this from the planned pad: missed grasp
    grasp_z_mm: float = 10.0
    release_xy_mm: float = 10.0     # gripper opening farther than this from the seated TCP: dropped or misplaced
    release_above_mm: float = 10.0  # ... or higher than this above the seated TCP
    corridor_mm: float = 80.0       # TCP farther than this (xy) from the step's straight path: unexpected motion
    stuck_holds: int = 3            # guard refusals in a row: stuck (WIRING.md)
    empty_width_mm: float = 4.5     # plan.GAP_CLOSED_MAX_MM: fingers fully closed on nothing


@dataclass
class _Progress:
    rollout: Rollout | None = None
    steps: tuple[Step, ...] = ()
    index: int = 0
    phase: str = "pick"             # pick -> insert -> retreat -> next step
    phase_t0: float = 0.0
    start_xy: tuple[float, float] = (0.0, 0.0)
    ended: str | None = None        # why the trial is held; "done" when every step finished
    holds_in_row: int = 0
    last_rows: np.ndarray | None = None  # the last chunk sent: where the gripper changed inside it

    @property
    def step(self) -> Step | None:
        return self.steps[self.index] if self.index < len(self.steps) else None


def _xy_dist_to_segment(p, a, b) -> float:
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class DirectGrootRuntime(GrootPolicy):
    """The official Policy (CAMERAS, IMAGE_SIZE 640x360, ACTION "tcp", PREFETCH False from GrootPolicy)."""

    def __init__(self, checkpoint: str | None = None, model: ChunkModel | None = None,
                 limits: GuardLimits | None = None, placement: str | None = None, budgets: Budgets | None = None,
                 clock: Callable[[], float] = time.monotonic):
        super().__init__(checkpoint, model=model, limits=limits)
        self.placement = placement or os.environ.get("DIRECT_PLACEMENT")
        self.rollouts = load_placement(self.placement)
        self.budgets, self.clock = budgets or Budgets(), clock
        self.progress = _Progress()
        self.events: list[dict] = []  # kept across trials, for the run's report
        self.resets = 0

    # ------------------------------------------------------------------ trial setup
    def reset(self) -> None:
        super().reset()
        self.resets += 1
        self.progress = _Progress()

    def _pick_rollout(self, obs: dict) -> Rollout | None:
        forced = os.environ.get("DIRECT_ROLLOUT")
        if forced:
            k = int(forced) - 1
            return self.rollouts[k] if 0 <= k < len(self.rollouts) else None
        trial = obs.get("trial") or {}
        said = " ".join(str(trial.get("instruction") or "").split())
        matches = [r for r in self.rollouts if r.prompt and " ".join(r.prompt.split()) == said]
        if len(matches) == 1:
            return matches[0]
        k = int(trial["index"]) if isinstance(trial.get("index"), int) else self.resets - 1
        fallback = matches[0] if matches else None  # one prompt for every trial (the mock): its first rollout
        return next((r for r in (matches or self.rollouts) if r.index == k), fallback)

    def _begin_trial(self, obs: dict, tcp: np.ndarray) -> None:
        pr = self.progress
        pr.rollout = self._pick_rollout(obs)
        if pr.rollout is None:
            return self._end(f"no placement.yaml rollout for this trial (trial {obs.get('trial')})")
        try:
            pr.steps = tuple(plan_rollout(pr.rollout))
        except PlacementError as e:
            return self._end(f"plan refused: {e}")
        self._event("trial_start", rollout=pr.rollout.index + 1, products=list(pr.rollout.products),
                    steps=len(pr.steps))
        self._start_step(tcp)

    def _start_step(self, tcp: np.ndarray) -> None:
        pr, step = self.progress, self.progress.step
        pr.phase, pr.phase_t0, pr.start_xy, pr.holds_in_row = "pick", self.clock(), (tcp[0], tcp[1]), 0
        if hasattr(self.model, "reset"):
            self.model.reset()  # every training episode is one step from its own start
        g = step.grasp_tcp
        text = self.set_plan("pick_insert", step.object_id, step.base_id, 0, "nominal", list(step.plan_target),
                             ContactTarget(g[0], g[1], g[2], label=f"grasp {step.object_id}"))
        self._event("step_start", step=pr.index + 1, part=step.object_id, base=step.base_id, instruction=text)

    # ------------------------------------------------------------------ progress
    def _end(self, reason: str) -> None:
        self.progress.ended = reason
        self.set_task(None)
        (log.info if reason == "done" else log.warning)("direct runtime: HOLD for the rest of the trial: %s", reason)
        self._event("trial_end", reason=reason)

    def _event(self, kind: str, **fields) -> None:
        self.events.append({"t": round(self.clock(), 2), "rollout": self.rollout, "type": kind, **fields})

    def _update(self, obs: dict, tcp: np.ndarray, closed: bool) -> None:
        pr = self.progress
        elapsed = self.clock() - pr.phase_t0
        if pr.phase == "pick":
            self._update_pick(pr.step, tcp, closed, elapsed)
        elif pr.phase == "insert":
            self._update_insert(obs, pr.step, tcp, closed, elapsed)
        else:
            self._update_retreat(pr.step, tcp, closed, elapsed)

    def _switch_xyz(self, closed: bool, tcp: np.ndarray) -> np.ndarray:
        """Where the gripper changed: the first row of the last chunk sent with the new state (a chunk may close on
        the pad and then lift, so the observed TCP is already past the grasp); the observed TCP if none has it."""
        rows = self.progress.last_rows
        if rows is not None:
            hit = np.flatnonzero((rows[:, 6] >= 0.5) == closed)
            if hit.size:
                return rows[hit[0], :3]
        return tcp[:3]

    def _update_pick(self, step: Step, tcp: np.ndarray, closed: bool, elapsed: float) -> None:
        b, g = self.budgets, step.grasp_tcp
        if closed:
            at = self._switch_xyz(True, tcp)
            if not (np.hypot(at[0] - g[0], at[1] - g[1]) <= b.grasp_xy_mm and abs(at[2] - g[2]) <= b.grasp_z_mm):
                return self._end(f"{step.object_id}: gripper closed away from the grasp pad at "
                                 f"({at[0]:.0f}, {at[1]:.0f}, {at[2]:.0f})")
            s = step.seat_tcp
            self.guard.set_target(ContactTarget(s[0], s[1], s[2], label=step.label))
            self.progress.phase, self.progress.phase_t0 = "insert", self.clock()
            return self._event("grasped", part=step.object_id, tcp=[round(float(v), 1) for v in at])
        if elapsed > b.pick_s:
            return self._end(f"{step.object_id}: no grasp within the {b.pick_s:.0f} s pick budget")
        if _xy_dist_to_segment(tcp, self.progress.start_xy, g) > b.corridor_mm:
            self._end(f"{step.object_id}: TCP ({tcp[0]:.0f}, {tcp[1]:.0f}) left the path to the grasp pad")

    def _update_insert(self, obs: dict, step: Step, tcp: np.ndarray, closed: bool, elapsed: float) -> None:
        b, s = self.budgets, step.seat_tcp
        if not closed:
            at = self._switch_xyz(False, tcp)
            if not (np.hypot(at[0] - s[0], at[1] - s[1]) <= b.release_xy_mm and at[2] <= s[2] + b.release_above_mm):
                return self._end(f"{step.label}: gripper opened at ({at[0]:.0f}, {at[1]:.0f}, {at[2]:.0f}), "
                                 f"away from the seat: part dropped or misplaced")
            self.progress.phase = "retreat"
            self._event("released", part=step.object_id, tcp=[round(float(v), 1) for v in at])
            return self._update_retreat(step, tcp, closed, elapsed)
        width = obs.get("gripper_width_mm")
        if width is not None and float(width) <= b.empty_width_mm:
            return self._end(f"{step.label}: gripper reports closed but empty ({float(width):.1f} mm): part dropped")
        if elapsed > b.insert_s:
            return self._end(f"{step.label}: not seated and released within the {b.insert_s:.0f} s insert budget")
        if _xy_dist_to_segment(tcp, step.grasp_tcp, s) > b.corridor_mm:
            self._end(f"{step.label}: TCP ({tcp[0]:.0f}, {tcp[1]:.0f}) left the path to the base")

    def _update_retreat(self, step: Step, tcp: np.ndarray, closed: bool, elapsed: float) -> None:
        pr, b = self.progress, self.budgets
        if closed:
            return self._end(f"{step.label}: gripper closed again after the release")
        if tcp[2] >= step.seat_tcp[2] + RETREAT_MM:
            self._event("step_done", step=pr.index + 1, part=step.object_id)
            pr.index += 1
            return self._start_step(tcp) if pr.step else self._end("done")
        if elapsed > b.insert_s:
            self._end(f"{step.label}: no retreat within the {b.insert_s:.0f} s insert budget")

    # ------------------------------------------------------------------ act
    def act(self, obs: dict) -> np.ndarray:
        if obs.get("tcp_pose") is None:
            return super().act(obs)  # FK-estimate hold, the model is not asked (GrootPolicy)
        tcp = np.asarray(obs["tcp_pose"], float)[:6]
        grip = float(np.asarray(obs["joints"], float)[6])
        pr = self.progress
        if pr.rollout is None and pr.ended is None:
            self._begin_trial(obs, tcp)
        elif pr.ended is None:
            self._update(obs, tcp, grip >= 0.5)
        if pr.ended is not None:
            self.steps += 1
            return hold_row(tcp, grip, self.guard.limits)
        logged = len(self.guard_log)
        rows = super().act(obs)
        pr.last_rows = np.asarray(rows, float).reshape(-1, 7)
        held = len(self.guard_log) > logged and self.guard_log[-1]["held"]
        pr.holds_in_row = pr.holds_in_row + 1 if held else 0
        if pr.holds_in_row >= self.budgets.stuck_holds:
            self._end(f"{pr.step.label}: guard refused {pr.holds_in_row} chunks in a row (stuck)")
        return rows
