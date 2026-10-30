"""Force guard: a demo whose contact peak exceeds physics/limits.yaml is discarded (never saved).

Reads `obs["sim"]["contact_peak"]` (the peak over the whole `step_tcp` chunk, which the env should report so short
spikes between control steps are not missed) and falls back to `obs["sim"]["contact"]` (the value at the end of the
step). Per object: `normal_N` is checked against `axial_N`, `lateral_N` against `lateral_N`, `torque_Nm` against
`torque_Nm`, and the optional `grip_N` (finger squeeze on the held part) against the optional `grip_N` limit, which
keeps the gripper from crushing parts. There are no built-in default limits: the guard refuses to run without the file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

LIMITS_PATH = Path(__file__).resolve().parents[1] / "physics" / "limits.yaml"
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
    peaks: dict[str, dict[str, float]] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    samples: int = 0
    trace: list[list[float]] = field(default_factory=list)  # per step [normal_N, lateral_N, torque_Nm], max over parts

    def observe(self, step: int, obs: dict) -> None:
        sim = obs.get("sim") or {}
        contact = sim.get("contact_peak", sim.get("contact"))
        if contact is None:
            raise LimitsError("observation has no sim contact forces; the guard cannot run blind")
        self.samples += 1
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
                lim = self.limits.get(lim_key)
                if lim is not None and val > lim and oid not in self.exempt:
                    self.violations.append(Violation(step, oid, key, val, lim))
        self.trace.append(row)

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
