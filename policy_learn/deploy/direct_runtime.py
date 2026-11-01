"""Fallback runtime where GR00T drives every motion step by step through placement.yaml, guarded and hold-only.
Authored by Shritej Gunda."""
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
HAND_PLACE_XY_MM = 5.0      # estimate: declared hand-placement accuracy of a base; release xy vs the planned seat
SEATED_Z_MM = 1.5           # estimate: release at most this above the seated TCP (first contact is 2-13 mm above)
JAM_MM = 1.0                # estimate: reported TCP this much above the commanded row at release = blocked
EMPTY_GRIPPER_ON_CELL = False  # the cell's obs has no gripper width: see the module docstring


@dataclass(frozen=True)
class Budgets:
    """Engineering estimates for timeouts and retries, not measured on the real cell."""

    pick_s: float = 60.0            # from the step's first act to the grasp
    insert_s: float = 90.0          # from the grasp to fingers open on the seat
    retreat_s: float = 12.0         # from the release to RETREAT_MM above the seat
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
        """Return the current plan step."""
        return self.steps[self.index] if self.index < len(self.steps) else None


def _forced_rollout(n: int) -> int | None:
    """Return DIRECT_ROLLOUT as a validated 0-based index."""
    raw = os.environ.get("DIRECT_ROLLOUT", "").strip()
    if not raw:
        return None
    if not raw.isdigit() or not 1 <= int(raw) <= n:
        raise ValueError(f"DIRECT_ROLLOUT={raw!r}: must be 1..{n} (placement.yaml has {n} rollouts)")
    log.warning("direct runtime: DIRECT_ROLLOUT=%s: rollout %s runs for EVERY trial, whatever the cell's prompt", raw,
                raw)
    return int(raw) - 1


def _xy_dist_to_segment(p, a, b) -> float:
    p, a, b = (np.asarray(v, float)[:2] for v in (p, a, b))
    ab = b - a
    t = 0.0 if not ab.any() else float(np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


class DirectGrootRuntime(GrootPolicy):
    """Official policy that runs the direct plan with GR00T."""

    def __init__(self, checkpoint: str | None = None, model: ChunkModel | None = None,
                 limits: GuardLimits | None = None, placement: str | None = None, budgets: Budgets | None = None,
                 clock: Callable[[], float] = time.monotonic, exec_horizon: int | None = None):
        super().__init__(checkpoint, model=model, limits=limits, exec_horizon=exec_horizon)  # None: $GROOT_EXEC_HORIZON
        self.placement = placement or os.environ.get("DIRECT_PLACEMENT")
        self.rollouts = load_placement(self.placement)
        self.forced = _forced_rollout(len(self.rollouts))
        self.plans: dict[int, tuple[Step, ...] | str] = {}  # rollout index -> steps, or why the planner refused
        for r in self.rollouts:
            try:
                self.plans[r.index] = tuple(plan_rollout(r))
            except PlacementError as e:
                self.plans[r.index] = str(e)
                log.error("direct runtime: rollout %d cannot run (it will hold): %s", r.index + 1, e)
        self.budgets, self.clock = budgets or Budgets(), clock
        self.progress = _Progress()
        self.events: list[dict] = []  # kept across trials, for the run's report
        self.resets = 0

    # Trial setup
    def reset(self) -> None:
        """Start a new trial."""
        super().reset()
        self.resets += 1
        self.progress = _Progress()

    def _pick_rollout(self, obs: dict) -> Rollout | None:
        if self.forced is not None:
            log.warning("direct runtime: DIRECT_ROLLOUT=%d forces rollout %d for EVERY trial", self.forced + 1,
                        self.forced + 1)
            return self.rollouts[self.forced]
        trial = obs.get("trial") or {}
        said = " ".join(str(trial.get("instruction") or "").split())
        matches = [r for r in self.rollouts if r.prompt and " ".join(r.prompt.split()) == said]
        if len(matches) == 1:
            return matches[0]
        k = int(trial["index"]) if isinstance(trial.get("index"), int) else self.resets - 1
        fallback = matches[0] if matches else None  # one prompt for every trial (the mock): its first rollout
        return next((r for r in (matches or self.rollouts) if r.index == k), fallback)

    def _begin_trial(self, obs: dict, tcp: np.ndarray, closed: bool) -> None:
        pr = self.progress
        pr.rollout = self._pick_rollout(obs)
        if pr.rollout is None:
            return self._end(f"no placement.yaml rollout for this trial (trial {obs.get('trial')})")
        plan = self.plans[pr.rollout.index]
        if isinstance(plan, str):
            return self._end(f"plan refused: {plan}")
        if closed:
            return self._end("gripper closed at start: the cell must home with the gripper open")
        pr.steps = plan
        self._event("trial_start", placement_rollout=pr.rollout.index + 1, products=list(pr.rollout.products),
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

    # Progress
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

    def _switch_row(self, closed: bool) -> np.ndarray | None:
        """Return the first row of the last chunk with the new gripper state, or None."""
        rows = self.progress.last_rows
        if rows is None:
            return None
        hit = np.flatnonzero((rows[:, 6] >= 0.5) == closed)
        return rows[hit[0]] if hit.size else None

    def _update_pick(self, step: Step, tcp: np.ndarray, closed: bool, elapsed: float) -> None:
        b, g = self.budgets, step.grasp_tcp
        if closed:
            row = self._switch_row(True)
            at = tcp[:3] if row is None else row[:3]
            dxy, dz = float(np.hypot(at[0] - g[0], at[1] - g[1])), float(at[2] - g[2])
            if dxy > step.grasp_xy_tol_mm or abs(dz) > step.grasp_z_tol_mm:
                return self._end(f"{step.object_id}: gripper closed off the grasp pad at ({at[0]:.1f}, {at[1]:.1f}, "
                                 f"{at[2]:.1f}): {dxy:.1f} mm xy (limit {step.grasp_xy_tol_mm:.2f}), {dz:+.1f} mm z "
                                 f"(limit {step.grasp_z_tol_mm:.2f})")
            s = step.seat_tcp
            self.guard.set_target(ContactTarget(s[0], s[1], s[2], label=step.label))
            self.progress.phase, self.progress.phase_t0 = "insert", self.clock()
            return self._event("grasped", part=step.object_id, tcp=[round(float(v), 1) for v in at])
        if elapsed > b.pick_s:
            return self._end(f"{step.object_id}: no grasp within the {b.pick_s:.0f} s pick budget")
        if _xy_dist_to_segment(tcp, self.progress.start_xy, g) > b.corridor_mm:
            self._end(f"{step.object_id}: TCP ({tcp[0]:.0f}, {tcp[1]:.0f}) left the path to the grasp pad")

    def _release_fault(self, step: Step, tcp: np.ndarray) -> str | None:
        """Return why this release is not seated, or None."""
        s, row = step.seat_tcp, self._switch_row(False)
        at = tcp[:3] if row is None else row[:3]
        dxy, xy_tol = float(np.hypot(at[0] - s[0], at[1] - s[1])), HAND_PLACE_XY_MM
        z_max = min(s[2] + SEATED_Z_MM, step.contact_z - 0.5)
        if dxy > xy_tol or at[2] > z_max:
            return (f"gripper opened at ({at[0]:.1f}, {at[1]:.1f}, {at[2]:.1f}), {dxy:.1f} mm from the planned seat (hand-"
                    f"placement limit {xy_tol:.1f}) and {at[2] - s[2]:+.1f} mm above it (limit {z_max - s[2]:+.1f}): not seated")
        sent = self.progress.last_rows[-1] if self.progress.last_rows is not None else None
        if sent is not None and tcp[2] - sent[2] > JAM_MM:
            return (f"reported TCP z {tcp[2]:.1f} is {tcp[2] - sent[2]:.1f} mm above the commanded {sent[2]:.1f} "
                    f"(limit {JAM_MM}): blocked, part jammed above the seat")
        return None

    def _update_insert(self, obs: dict, step: Step, tcp: np.ndarray, closed: bool, elapsed: float) -> None:
        b = self.budgets
        if not closed:
            fault = self._release_fault(step, tcp)
            if fault:
                return self._end(f"{step.label}: {fault}")
            self.progress.phase, self.progress.phase_t0 = "retreat", self.clock()
            self._event("released", part=step.object_id, tcp=[round(float(v), 1) for v in tcp[:3]])
            return self._update_retreat(step, tcp, closed, 0.0)
        width = obs.get("gripper_width_mm")  # never present on the real cell today (EMPTY_GRIPPER_ON_CELL)
        if width is not None and float(width) <= b.empty_width_mm:
            return self._end(f"{step.label}: gripper reports closed but empty ({float(width):.1f} mm): part dropped")
        if elapsed > b.insert_s:
            return self._end(f"{step.label}: not seated and released within the {b.insert_s:.0f} s insert budget")
        if _xy_dist_to_segment(tcp, step.grasp_tcp, step.seat_tcp) > b.corridor_mm:
            self._end(f"{step.label}: TCP ({tcp[0]:.0f}, {tcp[1]:.0f}) left the path to the base")

    def _update_retreat(self, step: Step, tcp: np.ndarray, closed: bool, elapsed: float) -> None:
        pr, b = self.progress, self.budgets
        if closed:
            return self._end(f"{step.label}: gripper closed again after the release")
        if tcp[2] >= step.seat_tcp[2] + RETREAT_MM:
            self._event("step_done", step=pr.index + 1, part=step.object_id)
            pr.index += 1
            return self._start_step(tcp) if pr.step else self._end("done")
        if elapsed > b.retreat_s:
            self._end(f"{step.label}: no retreat within the {b.retreat_s:.0f} s retreat budget")

    # Act
    def act(self, obs: dict) -> np.ndarray:
        """Return guarded rows for one observation."""
        if obs.get("tcp_pose") is None:
            return super().act(obs)  # FK-estimate hold, the model is not asked (GrootPolicy)
        tcp = np.asarray(obs["tcp_pose"], float)[:6]
        grip = float(np.asarray(obs["joints"], float)[6])
        pr = self.progress
        if pr.rollout is None and pr.ended is None:
            self._begin_trial(obs, tcp, grip >= 0.5)
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
