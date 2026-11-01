"""Real-arm safety guard for `act_tcp` chunks (pure Python + numpy; no robot or sim imports).

Every chunk the learned policy outputs passes through `guard_chunk` before it reaches the cell. The guard never
passes a bad chunk silently:

- Hard violations REFUSE the chunk from the first refused row on. The rows before it are sent only if that prefix on
  its own passes every check below (the arm then stops at its last row and the policy is asked again from a new
  pose, instead of being re-asked forever from a pose that never changes); with no valid prefix the guard returns one
  safe HOLD row (the current pose, the current gripper state, so a held part stays held). Either way the reasons are
  reported. A malformed chunk (shape, NaN/inf, dt) is always held whole. Refused: wrong shape, NaN/inf, dt outside
  0.1-1.0 s, a target outside the
  workspace box, fingers tilted past the fingers-down limit, a target more than `floor_clamp_mm` below the contact
  floor (seated height - 1 mm) near the current insertion target, a gripper value that is neither open nor closed, gripper chatter, a
  gripper change while moving, or a step more than `refuse_factor` times the speed limit (a model glitch).
- Steps a little too fast (up to `refuse_factor` times the limit) are CLAMPED to the limit, and targets up to
  `floor_clamp_mm` below the contact floor are RAISED to it; every clamp is reported as a reason. Inside the slow zone (the last 5 mm above the contact target, or above the table floor) the
  slow limit applies.

Units: mm, degrees, seconds; frame UF0 (identical to `CellClient.act_tcp`). Fingers straight down: w, p, r = -180, 0, R.
The speed limits are PRELIMINARY engineering choices, not measured values; tune them only with the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ROW_LEN = 7  # x, y, z, w, p, r, gripper
MAX_CHUNK = 100
DT_RANGE = (0.1, 1.0)


@dataclass(frozen=True)
class GuardLimits:
    """Limits the guard enforces. Workspace and dt come from the official client (PLAN section 3.1)."""

    x_mm: tuple[float, float] = (150.0, 650.0)
    y_mm: tuple[float, float] = (-450.0, 450.0)
    z_mm: tuple[float, float] = (-334.0, 300.0)
    max_abs_wpr_deg: float = 360.0          # the official client's check on w, p, r
    max_tilt_deg: float = 6.0               # fingers-down cone: insertion tilts about 3 deg plus a +-2 deg twist margin
    max_speed_mm_s: float = 150.0           # PRELIMINARY
    slow_speed_mm_s: float = 10.0           # PRELIMINARY: the last 5 mm before contact
    max_rot_deg_s: float = 60.0             # PRELIMINARY
    slow_rot_deg_s: float = 15.0            # PRELIMINARY
    slow_zone_mm: float = 5.0               # ROBOT_BRAIN section 2: the last 5 mm is slow
    floor_margin_mm: float = 1.0            # ROBOT_BRAIN section 2: never lower than seated height - 1 mm
    floor_clamp_mm: float = 3.0             # PRELIMINARY: up to 3 mm under the floor is raised to it, deeper is refused
    refuse_factor: float = 3.0              # a step over 3x the limit is a glitch: refuse instead of clamp
    gripper_snap: float = 0.1               # 0.93 counts as closed; 0.5 is refused
    gripper_change_max_step_mm: float = 2.0  # the gripper opens/closes only while (nearly) still


DEFAULT_LIMITS = GuardLimits()


@dataclass(frozen=True)
class ContactTarget:
    """The planner's current insertion (or grasp) target, for the contact floor and the slow zone.

    z_contact_mm is the TCP z at the seated (or grasp) pose. Within radius_mm of (x_mm, y_mm), no target may go below
    z_contact_mm - floor_margin, and the last slow_zone_mm above z_contact_mm is slow.
    """

    x_mm: float
    y_mm: float
    z_contact_mm: float
    radius_mm: float = 25.0
    label: str = ""


@dataclass(frozen=True)
class GuardResult:
    rows: np.ndarray                  # (N, 7) rows safe to send
    passed: bool                      # True: the chunk went through unchanged
    held: bool                        # True: the chunk was refused and `rows` is one hold row
    reasons: tuple[str, ...] = field(default_factory=tuple)
    truncated_at: int | None = None   # set when only rows [0, truncated_at) are sent: row truncated_at was refused


def wpr_to_matrix(w: float, p: float, r: float) -> np.ndarray:
    """FANUC fixed-angle convention: R = Rz(r) @ Ry(p) @ Rx(w), degrees."""
    w, p, r = np.radians([w, p, r])
    cw, sw, cp, sp, cr, sr = np.cos(w), np.sin(w), np.cos(p), np.sin(p), np.cos(r), np.sin(r)
    rx = np.array([[1, 0, 0], [0, cw, -sw], [0, sw, cw]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_to_wpr(rot: np.ndarray) -> tuple[float, float, float]:
    p = np.degrees(np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0)))
    w = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
    r = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    return float(w), float(p), float(r)


def tilt_deg(w: float, p: float, r: float) -> float:
    """Angle between the tool z axis and straight down."""
    z_axis = wpr_to_matrix(w, p, r)[:, 2]
    return float(np.degrees(np.arccos(np.clip(-z_axis[2], -1.0, 1.0))))


def _wrap180(a: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


def hold_row(current_pose, gripper: float, limits: GuardLimits | None = None) -> np.ndarray:
    """Stay where the arm is, keep the gripper as it is. Clipped into the box so the cell accepts it."""
    limits = limits or DEFAULT_LIMITS
    pose = np.asarray(current_pose, dtype=float)[:6].copy()
    for k, (lo, hi) in enumerate((limits.x_mm, limits.y_mm, limits.z_mm)):
        pose[k] = np.clip(pose[k], lo, hi)
    grip = 1.0 if float(gripper) >= 0.5 else 0.0
    return np.concatenate([pose, [grip]])[None, :]


class ChunkGuard:
    """Checks one chunk against the current pose; stateless between calls except for the contact target."""

    def __init__(self, limits: GuardLimits | None = None):
        self.limits = limits or DEFAULT_LIMITS
        self.target: ContactTarget | None = None

    def set_target(self, target: ContactTarget | None) -> None:
        self.target = target

    def guard(self, rows, current_pose, current_gripper: float, dt: float) -> GuardResult:
        reasons = _check_shape(rows, dt)
        if reasons:
            return GuardResult(hold_row(current_pose, current_gripper, self.limits), False, True, tuple(reasons))
        raw = np.array(rows, dtype=float).reshape(-1, ROW_LEN)
        res = self._guard_chunk(raw.copy(), current_pose, current_gripper, dt)
        if not res.held:
            return res
        # Every check is per row or along the path from the start, so a prefix of a valid chunk is valid: the longest
        # valid prefix is found by shortening one row at a time, and it is re-checked on its own in full.
        for k in range(len(raw) - 1, 0, -1):
            pre = self._guard_chunk(raw[:k].copy(), current_pose, current_gripper, dt)
            if not pre.held:
                note = f"chunk truncated at row {k} of {len(raw)}: sent rows 0-{k - 1}, then the arm stops"
                return GuardResult(pre.rows, False, False, (*res.reasons, note, *pre.reasons), truncated_at=k)
        if len(raw) == 1:
            return res
        first = self._guard_chunk(raw[:1].copy(), current_pose, current_gripper, dt)  # why no prefix: row 0's own refusal
        return GuardResult(res.rows, False, True, (*res.reasons, *(f"no valid prefix: {r}" for r in first.reasons)))

    def _guard_chunk(self, chunk: np.ndarray, current_pose, current_gripper: float, dt: float) -> GuardResult:
        """All checks on one well-formed chunk (modified in place by the clamps); refused -> one hold row."""
        hold = hold_row(current_pose, current_gripper, self.limits)
        floor_reasons = self._clamp_floor(chunk)
        hard = (self._check_box(chunk) + self._check_orientation(chunk) + self._check_floor(chunk)
                + self._snap_gripper(chunk))
        if hard:
            return GuardResult(hold, False, True, tuple(hard))
        start = np.concatenate([np.asarray(current_pose, dtype=float)[:6], [1.0 if current_gripper >= 0.5 else 0.0]])
        speed_reasons, refused = self._limit_steps(chunk, start, dt)
        grip_reasons = self._check_gripper_motion(chunk, start)
        if refused or grip_reasons:
            return GuardResult(hold, False, True, tuple(speed_reasons + grip_reasons))
        if speed_reasons:  # a clamped row lies between two checked points, but recheck: the floor is not convex
            after = self._check_box(chunk) + self._check_orientation(chunk) + self._check_floor(chunk)
            if after:
                return GuardResult(hold, False, True, tuple(speed_reasons + [f"after clamp: {r}" for r in after]))
        clamped = floor_reasons + speed_reasons
        return GuardResult(chunk, not clamped, False, tuple(clamped))

    def _check_box(self, chunk: np.ndarray) -> list[str]:
        out = []
        for k, (lo, hi) in enumerate((self.limits.x_mm, self.limits.y_mm, self.limits.z_mm)):
            bad = np.flatnonzero((chunk[:, k] < lo) | (chunk[:, k] > hi))
            if bad.size:
                i = int(bad[0])
                out.append(f"row {i}: {'xyz'[k]}={chunk[i, k]:.1f} mm outside workspace [{lo}, {hi}]")
        big = np.flatnonzero(np.abs(chunk[:, 3:6]).max(axis=1) > self.limits.max_abs_wpr_deg)
        if big.size:
            out.append(f"row {int(big[0])}: |w, p, r| over {self.limits.max_abs_wpr_deg} deg")
        return out

    def _check_orientation(self, chunk: np.ndarray) -> list[str]:
        for i, (w, p, r) in enumerate(chunk[:, 3:6]):
            t = tilt_deg(w, p, r)
            if t > self.limits.max_tilt_deg:
                return [f"row {i}: fingers tilted {t:.1f} deg from straight down (limit {self.limits.max_tilt_deg})"]
        return []

    def _near_target(self, chunk: np.ndarray) -> np.ndarray:
        t = self.target
        if t is None:
            return np.zeros(len(chunk), dtype=bool)
        return np.hypot(chunk[:, 0] - t.x_mm, chunk[:, 1] - t.y_mm) <= t.radius_mm

    def _clamp_floor(self, chunk: np.ndarray) -> list[str]:
        """Raise rows a little under the contact floor up to it (the policy aims ~1-3 mm low); deeper rows stay."""
        if self.target is None:
            return []
        floor = self.target.z_contact_mm - self.limits.floor_margin_mm
        low = self._near_target(chunk) & (chunk[:, 2] < floor) & (chunk[:, 2] >= floor - self.limits.floor_clamp_mm)
        if not low.any():
            return []
        i = int(np.flatnonzero(low)[0])
        name = f" ({self.target.label})" if self.target.label else ""
        reason = f"rows {int(low.sum())} from row {i}: z={chunk[i, 2]:.2f} mm raised to the contact floor {floor:.2f} mm{name}"
        chunk[low, 2] = floor
        return [reason]

    def _check_floor(self, chunk: np.ndarray) -> list[str]:
        if self.target is None:
            return []
        floor = self.target.z_contact_mm - self.limits.floor_margin_mm
        bad = np.flatnonzero(self._near_target(chunk) & (chunk[:, 2] < floor))
        if bad.size:
            i = int(bad[0])
            name = f" ({self.target.label})" if self.target.label else ""
            return [f"row {i}: z={chunk[i, 2]:.2f} mm below the contact floor {floor:.2f} mm{name}"]
        return []

    def _snap_gripper(self, chunk: np.ndarray) -> list[str]:
        g = chunk[:, 6]
        closed, opened = np.abs(g - 1.0) <= self.limits.gripper_snap, np.abs(g) <= self.limits.gripper_snap
        bad = np.flatnonzero(~(closed | opened))
        if bad.size:
            i = int(bad[0])
            return [f"row {i}: gripper {g[i]:.2f} is neither open (0) nor closed (1)"]
        chunk[:, 6] = np.where(closed, 1.0, 0.0)
        return []

    def _slow_mask(self, chunk: np.ndarray) -> np.ndarray:
        lim = self.limits
        slow = chunk[:, 2] <= lim.z_mm[0] + lim.slow_zone_mm
        if self.target is not None:
            slow |= self._near_target(chunk) & (chunk[:, 2] <= self.target.z_contact_mm + lim.slow_zone_mm)
        return slow

    def _limit_steps(self, chunk: np.ndarray, start: np.ndarray, dt: float) -> tuple[list[str], bool]:
        lim, reasons = self.limits, []
        slow = self._slow_mask(np.vstack([start, chunk]))
        prev = start
        for i in range(len(chunk)):
            in_slow = bool(slow[i] or slow[i + 1])
            v_max = (lim.slow_speed_mm_s if in_slow else lim.max_speed_mm_s) * dt
            w_max = (lim.slow_rot_deg_s if in_slow else lim.max_rot_deg_s) * dt
            d_xyz = chunk[i, :3] - prev[:3]
            d_wpr = np.array([_wrap180(chunk[i, 3] - prev[3]), chunk[i, 4] - prev[4], chunk[i, 5] - prev[5]])
            dist, ang = float(np.linalg.norm(d_xyz)), float(np.abs(d_wpr).max())
            ratio = max(dist / v_max, ang / w_max)
            if ratio > lim.refuse_factor:
                zone = "slow zone " if in_slow else ""
                return [(f"row {i}: step {dist:.1f} mm / {ang:.1f} deg in {dt} s is {ratio:.1f}x the {zone}limit "
                         f"(refuse above {lim.refuse_factor}x)")], True
            if ratio > 1.0:
                chunk[i, :3] = prev[:3] + d_xyz / ratio
                chunk[i, 3:6] = prev[3:6] + d_wpr / ratio
                reasons.append(f"row {i}: step clamped from {dist:.1f} mm / {ang:.1f} deg to the "
                               f"{'slow ' if in_slow else ''}limit")
            prev = chunk[i]
        return reasons, False

    def _check_gripper_motion(self, chunk: np.ndarray, start: np.ndarray) -> list[str]:
        g = np.concatenate([[start[6]], chunk[:, 6]])
        changes = np.flatnonzero(np.diff(g) != 0)
        if len(changes) > 1:
            return [f"gripper changes {len(changes)} times in one chunk (chatter)"]
        if len(changes) == 1:
            i = int(changes[0])
            prev = start if i == 0 else chunk[i - 1]
            step = float(np.linalg.norm(chunk[i, :3] - prev[:3]))
            if step > self.limits.gripper_change_max_step_mm:
                return [(f"row {i}: gripper changes while moving {step:.1f} mm in one step "
                         f"(limit {self.limits.gripper_change_max_step_mm} mm)")]
        return []


def _check_shape(rows, dt) -> list[str]:
    try:
        chunk = np.asarray(rows, dtype=float)
    except (TypeError, ValueError) as e:
        return [f"chunk is not numeric: {e}"]
    if chunk.ndim == 1 and chunk.size == ROW_LEN:
        chunk = chunk[None, :]
    if chunk.ndim != 2 or chunk.shape[1] != ROW_LEN or not 1 <= len(chunk) <= MAX_CHUNK:
        return [f"chunk shape {chunk.shape} is not (N, 7) with 1 <= N <= {MAX_CHUNK}"]
    if not np.all(np.isfinite(chunk)):
        return ["chunk contains NaN or inf"]
    if not (np.isfinite(dt) and DT_RANGE[0] <= float(dt) <= DT_RANGE[1]):
        return [f"dt {dt} outside {DT_RANGE} s"]
    return []
