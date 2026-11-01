"""Force guard that discards any demo whose contact peak exceeds physics/limits.yaml.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

LIMITS_PATH = Path(__file__).resolve().parents[1] / "physics" / "limits.yaml"
TOUCH_N = 0.05  # robot_touch above this is robot contact (gates.FINGER_TOUCH_N, the env's TOUCH_N)
AXES = (("normal_N", "axial_N"), ("lateral_N", "lateral_N"), ("torque_Nm", "torque_Nm"), ("grip_N", "grip_N"))


class LimitsError(RuntimeError):
    """limits.yaml is missing or incomplete; recording cannot run without the force limits."""


@dataclass(frozen=True)
class Limits:
    axial_N: float
    lateral_N: float
    torque_Nm: float
    status: str
    source: str
    grip_N: float | None = None

    def get(self, key: str) -> float | None:
        return getattr(self, key)


def load_limits(path: Path = LIMITS_PATH) -> Limits:
    if not Path(path).exists():
        raise LimitsError(f"{path} not found: the force limits come from the physics study (agent 2)")
    raw = yaml.safe_load(Path(path).read_text()) or {}
    missing = [k for k in ("axial_N", "lateral_N", "torque_Nm", "status", "source") if raw.get(k) is None]
    if missing:
        raise LimitsError(f"{path} lacks {missing}")
    if raw["status"] not in ("PRELIMINARY", "SIM_MEASURED"):
        raise LimitsError(f"{path}: status must be PRELIMINARY or SIM_MEASURED, got {raw['status']!r}")
    grip = raw.get("grip_N")
    return Limits(float(raw["axial_N"]), float(raw["lateral_N"]), float(raw["torque_Nm"]), raw["status"],
                  str(raw["source"]), None if grip is None else float(grip))


@dataclass
class Violation:
    step: int
    object_id: str
    axis: str
    value: float
    limit: float


@dataclass
class ForceGuard:
    limits: Limits
    exempt: set[str] = field(default_factory=set)  # defect pieces after release into the defect area
    per_object: dict[str, dict[str, float]] = field(default_factory=dict)  # objects.yaml max_force (tighter wins)
    peaks: dict[str, dict[str, float]] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    samples: int = 0
    trace: list[list[float]] = field(default_factory=list)  # per step [normal_N, lateral_N, torque_Nm], max over parts
    touched: bool = False  # the robot has touched an object (robot_touch > TOUCH_N) at or before this step
    pre_touch_peaks: dict[str, dict[str, float]] = field(default_factory=dict)
    pre_touch_violations: list[Violation] = field(default_factory=list)  # a bad scene: the episode is reset_invalid

    def observe(self, step: int, obs: dict) -> None:
        """Log pre-contact peaks as scene checks and apply the force guard in full from first contact."""
        sim = obs.get("sim") or {}
        contact = sim.get("contact_peak", sim.get("contact"))
        if contact is None:
            raise LimitsError("observation has no sim contact forces; the guard cannot run blind")
        touch = sim.get("robot_touch")
        if touch is None or any(float(v) > TOUCH_N for v in touch.values()):
            self.touched = True  # no touch data: everything counts as robot contact (the strict side)
        self.samples += 1
        if not self.touched:
            self._pre_touch(step, contact)
            self.trace.append([0.0, 0.0, 0.0] if not contact else self._row(contact))
            return
        row = [0.0, 0.0, 0.0]
        for oid, c in contact.items():
            if oid not in self.exempt:
                for i, key in enumerate(("normal_N", "lateral_N", "torque_Nm")):
                    row[i] = max(row[i], abs(float(c.get(key, 0.0))))
            peak = self.peaks.setdefault(oid, {})
            for key, lim_key in AXES:
                if key not in c:
                    continue
                val = abs(float(c[key]))
                peak[key] = max(peak.get(key, 0.0), val)
                lim = self.limit(oid, lim_key)
                if lim is not None and val > lim and oid not in self.exempt:
                    self.violations.append(Violation(step, oid, key, val, lim))
        self.trace.append(row)

    def _row(self, contact: dict) -> list[float]:
        row = [0.0, 0.0, 0.0]
        for oid, c in contact.items():
            if oid not in self.exempt:
                for i, key in enumerate(("normal_N", "lateral_N", "torque_Nm")):
                    row[i] = max(row[i], abs(float(c.get(key, 0.0))))
        return row

    def _pre_touch(self, step: int, contact: dict) -> None:
        for oid, c in contact.items():
            peak = self.pre_touch_peaks.setdefault(oid, {})
            for key, lim_key in AXES:
                if key in c:
                    val = abs(float(c[key]))
                    peak[key] = max(peak.get(key, 0.0), val)
                    lim = self.limit(oid, lim_key)
                    if lim is not None and val > lim:
                        self.pre_touch_violations.append(Violation(step, oid, key, val, lim))

    def limit(self, oid: str, key: str) -> float | None:
        """The global limit, or the object's own (objects.yaml max_force, e.g. M5_04 axial 2.5 N) when tighter."""
        base = self.limits.get(key)
        own = self.per_object.get(oid.split("#")[0], {}).get(key)
        if own is None:
            return base
        return float(own) if base is None else min(base, float(own))

    @property
    def ok(self) -> bool:
        return self.samples > 0 and not self.violations

    def episode_peaks(self) -> dict[str, float]:
        """Max over every object (exempt ones included, so nothing is hidden) per axis."""
        out: dict[str, float] = {}
        for p in self.peaks.values():
            for k, v in p.items():
                out[k] = max(out.get(k, 0.0), v)
        return out
