"""Real-arm safety guard for `act_tcp` chunks (pure Python + numpy; no robot or sim imports).

Every chunk the learned policy outputs passes through `guard_chunk` before it reaches the cell. The guard never
passes a bad chunk silently:

- Hard violations REFUSE the whole chunk and return one safe HOLD row (the current pose, the current gripper state, so
  a held part stays held) plus the reasons: wrong shape, NaN/inf, dt outside 0.1-1.0 s, a target outside the
  workspace box, fingers tilted past the fingers-down limit, a target lower than the contact floor (seated height
  - 1 mm) near the current insertion target, a gripper value that is neither open nor closed, gripper chatter, a
  gripper change while moving, or a step more than `refuse_factor` times the speed limit (a model glitch).
- Steps a little too fast (up to `refuse_factor` times the limit) are CLAMPED to the limit, and every clamp is
  reported as a reason. Inside the slow zone (the last 5 mm above the contact target, or above the table floor) the
  slow limit applies.

Units: mm, degrees, seconds; frame UF0 (identical to `CellClient.act_tcp`). Fingers straight down: w, p, r = -180, 0, R.
The speed limits are PRELIMINARY engineering choices, not measured values; tune them only with the user.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROW_LEN = 7  # x, y, z, w, p, r, gripper
MAX_CHUNK = 100
DT_RANGE = (0.1, 1.0)
_LIMIT_FIELDS = (
    "x_mm",
    "y_mm",
    "z_mm",
    "max_abs_wpr_deg",
    "max_tilt_deg",
    "max_speed_mm_s",
    "slow_speed_mm_s",
    "max_rot_deg_s",
    "slow_rot_deg_s",
    "slow_zone_mm",
    "floor_margin_mm",
    "refuse_factor",
    "gripper_snap",
    "gripper_change_max_step_mm",
)


class UnsafeHoldError(RuntimeError):
    """The observed TCP cannot be represented as a zero-motion official hold command."""


class InvalidGuardLimitsError(ValueError):
    """Guard limits are non-finite, inverted, or internally inconsistent."""


def _finite_scalar(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float, np.number)) and np.isfinite(float(value))


def _read_regular_bytes(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise InvalidGuardLimitsError(f"{label} must be one readable non-symlink file") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise InvalidGuardLimitsError(f"{label} must be one regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            return source.read()
    except OSError as error:
        raise InvalidGuardLimitsError(f"{label} could not be read") from error
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class GuardLimits:
    """Limits the guard enforces. Workspace and dt come from the official client (PLAN section 3.1)."""

    x_mm: tuple[float, float] = (150.0, 650.0)
    y_mm: tuple[float, float] = (-450.0, 450.0)
    z_mm: tuple[float, float] = (-334.0, 300.0)
    max_abs_wpr_deg: float = 360.0  # the official client's check on w, p, r
    max_tilt_deg: float = 6.0  # fingers-down cone: insertion tilts about 3 deg plus a +-2 deg twist margin
    max_speed_mm_s: float = 150.0  # PRELIMINARY
    slow_speed_mm_s: float = 10.0  # PRELIMINARY: the last 5 mm before contact
    max_rot_deg_s: float = 60.0  # PRELIMINARY
    slow_rot_deg_s: float = 15.0  # PRELIMINARY
    slow_zone_mm: float = 5.0  # ROBOT_BRAIN section 2: the last 5 mm is slow
    floor_margin_mm: float = 1.0  # ROBOT_BRAIN section 2: never lower than seated height - 1 mm
    refuse_factor: float = 3.0  # a step over 3x the limit is a glitch: refuse instead of clamp
    gripper_snap: float = 0.1  # 0.93 counts as closed; 0.5 is refused
    gripper_change_max_step_mm: float = 2.0  # the gripper opens/closes only while (nearly) still
    status: str = "PRELIMINARY"
    source: str = "T12 engineering defaults; not approved for CELL"
    _calibration_sha256: str | None = field(default=None, init=False, repr=False)
    _calibration_path: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, str) or not isinstance(self.source, str):
            raise InvalidGuardLimitsError("guard status and source must be strings")
        intervals = (("x_mm", self.x_mm), ("y_mm", self.y_mm), ("z_mm", self.z_mm))
        for label, interval in intervals:
            valid = (
                isinstance(interval, tuple)
                and len(interval) == 2
                and all(_finite_scalar(item) for item in interval)
                and interval[0] < interval[1]
            )
            if not valid:
                raise InvalidGuardLimitsError(f"{label} must be one finite increasing interval")
        positive = {
            "max_abs_wpr_deg": self.max_abs_wpr_deg,
            "max_tilt_deg": self.max_tilt_deg,
            "max_speed_mm_s": self.max_speed_mm_s,
            "slow_speed_mm_s": self.slow_speed_mm_s,
            "max_rot_deg_s": self.max_rot_deg_s,
            "slow_rot_deg_s": self.slow_rot_deg_s,
            "slow_zone_mm": self.slow_zone_mm,
            "refuse_factor": self.refuse_factor,
        }
        if any(not _finite_scalar(value) or value <= 0.0 for value in positive.values()):
            raise InvalidGuardLimitsError("guard limits must contain finite positive magnitudes")
        if self.slow_speed_mm_s > self.max_speed_mm_s or self.slow_rot_deg_s > self.max_rot_deg_s:
            raise InvalidGuardLimitsError("slow-zone limits cannot exceed free-space limits")
        if self.max_tilt_deg > 180.0 or self.refuse_factor <= 1.0:
            raise InvalidGuardLimitsError("tilt/refusal limits are outside their valid ranges")
        if not _finite_scalar(self.floor_margin_mm) or self.floor_margin_mm < 0.0:
            raise InvalidGuardLimitsError("floor margin must be finite and nonnegative")
        if not _finite_scalar(self.gripper_snap) or not 0.0 <= self.gripper_snap < 0.5:
            raise InvalidGuardLimitsError("gripper snap must be finite and below 0.5")
        if not _finite_scalar(self.gripper_change_max_step_mm) or self.gripper_change_max_step_mm < 0.0:
            raise InvalidGuardLimitsError("gripper change distance must be finite and nonnegative")

    @property
    def production_ready(self) -> bool:
        return (
            self.status == "MEASURED"
            and bool(self.source.strip())
            and self._calibration_sha256 is not None
            and self._calibration_path is not None
        )

    @property
    def calibration_sha256(self) -> str | None:
        """Digest of the report that supplied every limit, absent for direct constructors."""
        return self._calibration_sha256

    @classmethod
    def from_calibration(cls, path: str | Path, expected_sha256: str) -> GuardLimits:
        """Load every production limit from one immutable measured report."""
        report = Path(path)
        raw = _read_regular_bytes(report, "calibration report")
        measured = hashlib.sha256(raw).hexdigest()
        if measured != expected_sha256:
            raise InvalidGuardLimitsError("calibration report does not match its reviewed SHA-256")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise InvalidGuardLimitsError("calibration report is not valid JSON") from error
        if not isinstance(payload, dict) or set(payload) != {"status", "source", "limits"}:
            raise InvalidGuardLimitsError("calibration report fields must be status, source, and limits")
        values = payload["limits"]
        if not isinstance(values, dict) or set(values) != set(_LIMIT_FIELDS):
            raise InvalidGuardLimitsError("calibration report has missing or unknown limit fields")
        for field_name in ("x_mm", "y_mm", "z_mm"):
            value = values[field_name]
            if not isinstance(value, list):
                raise InvalidGuardLimitsError(f"calibration {field_name} must be a JSON array")
            values[field_name] = tuple(value)
        limits = cls(**values, status=payload["status"], source=payload["source"])
        object.__setattr__(limits, "_calibration_sha256", measured)
        object.__setattr__(limits, "_calibration_path", str(report.resolve()))
        return limits


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
    rows: np.ndarray  # (N, 7) rows safe to send
    passed: bool  # True: the chunk went through unchanged
    held: bool  # True: the chunk was refused and `rows` is one hold row
    reasons: tuple[str, ...] = field(default_factory=tuple)


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
    """Stay exactly where the arm is; never turn an invalid observation into commanded motion."""
    limits = limits or DEFAULT_LIMITS
    try:
        pose = np.asarray(current_pose, dtype=float)
    except (TypeError, ValueError) as error:
        raise UnsafeHoldError("cannot issue a TCP hold from a non-numeric pose") from error
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise UnsafeHoldError("cannot issue a TCP hold without one finite six-axis pose")
    for k, (lo, hi) in enumerate((limits.x_mm, limits.y_mm, limits.z_mm)):
        if not lo <= pose[k] <= hi:
            raise UnsafeHoldError("cannot issue a TCP hold from outside the official workspace")
    if np.abs(pose[3:6]).max() > limits.max_abs_wpr_deg:
        raise UnsafeHoldError("cannot issue a TCP hold from outside the official orientation bounds")
    grip = observed_gripper_state(gripper, limits)
    return np.concatenate([pose.copy(), [grip]])[None, :]


def observed_gripper_state(value: object, limits: GuardLimits | None = None) -> float:
    """Return a binary observed state only when the sensor value is unambiguous."""
    del limits  # command snapping is not valid authority for observed sensor state
    if isinstance(value, bool):
        raise UnsafeHoldError("observed gripper state must be numeric, not boolean")
    try:
        gripper = float(value)
    except (TypeError, ValueError) as error:
        raise UnsafeHoldError("observed gripper state must be numeric") from error
    if not np.isfinite(gripper):
        raise UnsafeHoldError("observed gripper state must be finite")
    if gripper in (0.0, 1.0):
        return gripper
    raise UnsafeHoldError("observed gripper state is neither unambiguously open nor closed")


class ChunkGuard:
    """Checks one chunk against the current pose; stateless between calls except for the contact target."""

    def __init__(self, limits: GuardLimits | None = None):
        self.limits = limits or DEFAULT_LIMITS
        self.target: ContactTarget | None = None

    def set_target(self, target: ContactTarget | None) -> None:
        self.target = target

    def guard(self, rows, current_pose, current_gripper: float, dt: float) -> GuardResult:
        hold = hold_row(current_pose, current_gripper, self.limits)
        reasons = _check_shape(rows, dt)
        if reasons:
            return GuardResult(hold, False, True, tuple(reasons))
        chunk = np.array(rows, dtype=float).reshape(-1, ROW_LEN)
        hard = (
            self._check_box(chunk)
            + self._check_orientation(chunk)
            + self._check_floor(chunk)
            + self._snap_gripper(chunk)
        )
        if hard:
            return GuardResult(hold, False, True, tuple(hard))
        start = np.concatenate(
            [np.asarray(current_pose, dtype=float)[:6], [observed_gripper_state(current_gripper, self.limits)]]
        )
        speed_reasons, refused = self._limit_steps(chunk, start, dt)
        grip_reasons = self._check_gripper_motion(chunk, start)
        if refused or grip_reasons:
            return GuardResult(hold, False, True, tuple(speed_reasons + grip_reasons))
        if speed_reasons:  # a clamped row lies between two checked points, but recheck: the floor is not convex
            after = self._check_box(chunk) + self._check_orientation(chunk) + self._check_floor(chunk)
            if after:
                return GuardResult(hold, False, True, tuple(speed_reasons + [f"after clamp: {r}" for r in after]))
        return GuardResult(chunk, not speed_reasons, False, tuple(speed_reasons))

    def static_reasons(self, rows, dt: float) -> tuple[str, ...]:
        """Check a final target without treating its distance as one control step."""
        reasons = _check_shape(rows, dt)
        if reasons:
            return tuple(reasons)
        chunk = np.array(rows, dtype=float, copy=True).reshape(-1, ROW_LEN)
        hard = (
            self._check_box(chunk)
            + self._check_orientation(chunk)
            + self._check_floor(chunk)
            + self._snap_gripper(chunk)
        )
        return tuple(hard)

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
                return [
                    (
                        f"row {i}: step {dist:.1f} mm / {ang:.1f} deg in {dt} s is {ratio:.1f}x the {zone}limit "
                        f"(refuse above {lim.refuse_factor}x)"
                    )
                ], True
            if ratio > 1.0:
                chunk[i, :3] = prev[:3] + d_xyz / ratio
                chunk[i, 3:6] = prev[3:6] + d_wpr / ratio
                reasons.append(
                    f"row {i}: step clamped from {dist:.1f} mm / {ang:.1f} deg to the {'slow ' if in_slow else ''}limit"
                )
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
                return [
                    (
                        f"row {i}: gripper changes while moving {step:.1f} mm in one step "
                        f"(limit {self.limits.gripper_change_max_step_mm} mm)"
                    )
                ]
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
