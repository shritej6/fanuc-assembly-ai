"""Load and validate the Mimic plan in skills.yaml without Isaac Lab.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .catalog import SKILLS

SPEC_PATH = Path(__file__).with_name("skills.yaml")
FRAMES = ("grasp_frame", "seat_frame", "spot_frame")
STRATEGIES = ("random", "nearest_neighbor_object", "nearest_neighbor_robot_distance")


class SpecError(ValueError):
    """skills.yaml breaks a rule the Mimic generator relies on."""


@dataclass(frozen=True)
class SubtaskSpec:
    """One subtask of a skill in the Mimic plan."""
    name: str
    object_ref: str
    term_signal: str | None
    term_offset: tuple[int, int]
    noise_pos_m: float
    interp_steps: int


@dataclass(frozen=True)
class SkillSpec:
    """One skill of the Mimic plan and its subtasks."""
    name: str
    seed_dir: str
    description: str
    subtasks: tuple[SubtaskSpec, ...]

    @property
    def signals(self) -> list[str]:
        """Return the subtask termination signals in order."""
        return [s.term_signal for s in self.subtasks if s.term_signal is not None]


@dataclass(frozen=True)
class MimicSpec:
    """The whole validated Mimic plan."""
    raw: dict
    skills: dict[str, SkillSpec]

    @property
    def eef_name(self) -> str:
        """Return the end effector name."""
        return self.raw["eef_name"]

    @property
    def rot_rad_per_m(self) -> float:
        """Return the rotation noise in rad per metre of position noise."""
        return math.radians(float(self.raw["rot_deg_per_mm"])) * 1000.0


def _subtask(raw: dict, noise: dict, where: str) -> SubtaskSpec:
    if raw.get("object_ref") not in FRAMES:
        raise SpecError(f"{where}: object_ref {raw.get('object_ref')!r} not in {FRAMES}")
    if raw.get("noise") not in noise:
        raise SpecError(f"{where}: noise profile {raw.get('noise')!r} not defined")
    lo, hi = (int(v) for v in raw.get("term_offset", (0, 0)))
    if not 0 <= lo <= hi:
        raise SpecError(f"{where}: term_offset must be 0 <= lo <= hi, got {(lo, hi)}")
    steps = int(raw.get("interp_steps", 5))
    if steps < 0:
        raise SpecError(f"{where}: interp_steps must be >= 0")
    return SubtaskSpec(raw["name"], raw["object_ref"], raw.get("term_signal"), (lo, hi),
                       float(noise[raw["noise"]]["pos_mm"]) / 1000.0, steps)


def _skill(name: str, raw: dict, noise: dict) -> SkillSpec:
    subs = tuple(_subtask(s, noise, f"{name}.{s.get('name')}") for s in raw["subtasks"])
    if len(subs) < 2:
        raise SpecError(f"{name}: needs at least 2 subtasks")
    if subs[-1].term_signal is not None:
        raise SpecError(f"{name}: the last subtask ends with the episode, its term_signal must be null")
    if any(s.term_signal is None for s in subs[:-1]):
        raise SpecError(f"{name}: every subtask but the last needs a term_signal")
    if len({s.term_signal for s in subs[:-1]}) != len(subs) - 1:
        raise SpecError(f"{name}: term signals must be unique")
    return SkillSpec(name, raw["seed_dir"], raw.get("description", ""), subs)


def load_spec(path: str | Path = SPEC_PATH) -> MimicSpec:
    """Load skills.yaml, validate it and return the MimicSpec."""
    raw = yaml.safe_load(Path(path).read_text())
    noise = raw["noise_profiles"]
    contact_limit_m = 0.5e-3  # 0.5 mm clearance per side (cad/products/parameters.py)
    if noise["contact"]["pos_mm"] / 1000.0 >= contact_limit_m / 2:
        raise SpecError("contact noise must stay well under the 0.5 mm insert clearance")
    if noise["contact"]["pos_mm"] * float(raw["rot_deg_per_mm"]) > 0.5:
        raise SpecError("contact rotation noise must stay under 0.5 deg (keyed parts tolerate about 1.4 deg)")
    unknown = set(raw["skills"]) - set(SKILLS)
    if unknown:
        raise SpecError(f"skills {sorted(unknown)} are not in catalog.SKILLS")
    if raw["selection"]["strategy"] not in STRATEGIES:
        raise SpecError(f"selection strategy {raw['selection']['strategy']!r} unknown")
    skills = {n: _skill(n, s, noise) for n, s in raw["skills"].items()}
    return MimicSpec(raw=raw, skills=skills)
