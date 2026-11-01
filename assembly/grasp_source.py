"""Fail-closed parsing for the pinned grasp-frame artifact.
Authored by Shritej Gunda.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Any


class GraspSourceError(ValueError):
    """Raised when a grasp frame cannot authorize one atomic turn/TCP pair."""


@dataclass(frozen=True, slots=True)
class PreferredGrasp:
    """The sole turn/TCP pair authorized for CELL execution."""

    turn_deg: float
    tcp_local_mm: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class GraspSource:
    """Immutable provenance for the grasp-frame source artifact."""

    file: str
    generator: str
    commit: str
    commit_timestamp: str
    snapshot: str
    sha256: str


_REQUIRED_FIELDS = {
    "closing_axis",
    "pad_width_mm",
    "pad_top_z_mm",
    "contact_height_mm",
    "pad_point_local_mm",
    "tcp_local_mm",
    "roll_offset_deg",
}


def _validated_roll(grip: dict[str, Any], object_id: str) -> float:
    try:
        expected = {"X": 90.0, "Y": 0.0}[grip["closing_axis"]]
        raw_roll = grip["roll_offset_deg"]
    except (KeyError, TypeError) as error:
        raise GraspSourceError(f"Malformed grasp source entry for {object_id}") from error
    if isinstance(raw_roll, bool) or not isinstance(raw_roll, (int, float)):
        raise GraspSourceError(f"T29 grasp source has invalid roll semantics for {object_id}")
    roll = float(raw_roll)
    if not isfinite(roll) or not -180.0 <= roll <= 180.0 or roll != expected:
        raise GraspSourceError(f"T29 grasp source has invalid roll semantics for {object_id}")
    return roll


def _validated_tcp(value: object, object_id: str, turn: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise GraspSourceError(f"T29 grasp source has invalid TCP for {object_id} turn {turn}")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise GraspSourceError(f"T29 grasp source TCP must contain numeric values for {object_id}")
    tcp = tuple(float(item) for item in value)
    if not all(isfinite(item) for item in tcp):
        raise GraspSourceError(f"T29 grasp source TCP must contain finite values for {object_id}")
    return tcp  # type: ignore[return-value]


def preferred_grasp(grip: object, object_id: str) -> PreferredGrasp:
    """Validate the artifact and return its source-tested turn/TCP pair."""

    if not isinstance(grip, dict) or set(grip) != _REQUIRED_FIELDS:
        raise GraspSourceError(f"Malformed grasp source fields for {object_id}")
    roll = _validated_roll(grip, object_id)
    turn_tcps = grip["tcp_local_mm"]
    if not isinstance(turn_tcps, dict):
        raise GraspSourceError(f"T29 grasp source requires turn-indexed tcp_local_mm for {object_id}")
    preferred_key = f"{roll:g}"
    if preferred_key not in turn_tcps:
        raise GraspSourceError(f"T29 grasp source is missing validated turn {preferred_key} for {object_id}")
    alternate_key = "180" if roll == 0.0 else "-90"
    if set(turn_tcps) != {preferred_key, alternate_key}:
        raise GraspSourceError(f"T29 grasp source requires canonical turn keys for {object_id}")
    parsed = {turn: _validated_tcp(tcp, object_id, turn) for turn, tcp in turn_tcps.items()}
    return PreferredGrasp(turn_deg=roll, tcp_local_mm=parsed[preferred_key])
