"""Force guard for generated demos: keep a demo only if its contact force never exceeded the limits over its whole
length (docs/ROBOT_BRAIN.md section 2). Limits come from agent 2's `physics/limits.yaml`; nothing is defaulted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REQUIRED = ("axial_N", "lateral_N", "torque_Nm", "status", "source")
STATUSES = ("PRELIMINARY", "SIM_MEASURED")


class ForceLimitError(ValueError):
    """limits.yaml is missing, incomplete or not positive."""


@dataclass(frozen=True)
class ForceLimits:
    axial_N: float
    lateral_N: float
    torque_Nm: float
    status: str
    source: str

    def as_array(self) -> np.ndarray:
        return np.array([self.axial_N, self.lateral_N, self.torque_Nm], dtype=np.float64)


def load_limits(path: str | Path) -> ForceLimits:
    path = Path(path)
    if not path.exists():
        raise ForceLimitError(f"{path} not found: agent 2 must publish the force limits before demos are accepted")
    data = yaml.safe_load(path.read_text()) or {}
    missing = [k for k in REQUIRED if k not in data]
    if missing:
        raise ForceLimitError(f"{path} lacks {missing}")
    if data["status"] not in STATUSES:
        raise ForceLimitError(f"{path} status {data['status']!r} not in {STATUSES}")
    limits = ForceLimits(float(data["axial_N"]), float(data["lateral_N"]), float(data["torque_Nm"]),
                         str(data["status"]), str(data["source"]))
    if not (limits.as_array() > 0).all():
        raise ForceLimitError(f"{path}: every limit must be positive, got {limits.as_array().tolist()}")
    return limits


@dataclass(frozen=True)
class GuardResult:
    ok: bool
    peak: np.ndarray  # [normal_N, lateral_N, torque_Nm] over the episode
    first_violation_step: int | None


def check_trace(contact: np.ndarray, limits: ForceLimits) -> GuardResult:
    """contact: (T, 3) per-step peaks [normal_N, lateral_N, torque_Nm]. Non-finite values fail the guard."""
    contact = np.asarray(contact, dtype=np.float64)
    if contact.ndim != 2 or contact.shape[1] != 3 or len(contact) == 0:
        raise ValueError(f"contact trace must be (T, 3) with T > 0, got {contact.shape}")
    finite = np.isfinite(contact).all(axis=1)
    over = (np.abs(contact) > limits.as_array()).any(axis=1) | ~finite
    bad = np.flatnonzero(over)
    peak = np.where(np.isfinite(contact), np.abs(contact), np.inf).max(axis=0)
    return GuardResult(ok=bad.size == 0, peak=peak, first_violation_step=int(bad[0]) if bad.size else None)
