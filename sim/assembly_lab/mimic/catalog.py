"""Index tables that encode an episode's plan as the numeric plan_ids observation.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SKILLS = ("pick_insert", "pull_out", "place_free", "defect_drop")
CASES = ("nominal", "crowded", "defect", "fallen", "foreign", "wont_seat", "dropped", "fix_above", "spare")
NO_BASE = -1
FOREIGN = "foreign"


@lru_cache(maxsize=1)
def object_ids() -> tuple[str, ...]:
    """The 30 CAD ids (sorted) plus `foreign` for an unknown piece (ROBOT_BRAIN section 3)."""
    frames = json.loads((REPO / "cad/grasp_frames.json").read_text())
    return (*sorted(frames["objects"]), FOREIGN)


@lru_cache(maxsize=1)
def grip_axes() -> dict[str, list[str]]:
    """Per object, the closing axis of each CAD grip pad, in grasp_frames.json order (the grip index)."""
    frames = json.loads((REPO / "cad/grasp_frames.json").read_text())["objects"]
    return {oid: [g["closing_axis"] for g in obj["grips"]] for oid, obj in frames.items()}


@lru_cache(maxsize=1)
def product_bases() -> dict[int, str]:
    """Map each product to the base named by its pinned GR00T training data."""
    frames = json.loads((REPO / "cad/grasp_frames.json").read_text())["objects"]
    result: dict[int, str] = {}
    for object_id, object_data in frames.items():
        product = product_of(object_id)
        if product is not None and object_id.endswith("_01") and object_data.get("base"):
            result[product] = object_data["base"]
    return result


def product_of(object_id: str) -> int | None:
    """M4_02 -> 4, F04 -> 4, foreign -> None."""
    if object_id.startswith("M") and "_" in object_id:
        return int(object_id[1:object_id.index("_")])
    if object_id.startswith("F") and object_id[1:].isdigit():
        return int(object_id[1:])
    return None


def decode_plan_ids(ids) -> dict:
    """[skill, object, base, grip, case] -> names. Raises IndexError/ValueError on a bad index (never guesses)."""
    skill, obj, base, grip, case = (round(float(v)) for v in ids)
    oids = object_ids()
    for name, idx, table in (("skill", skill, SKILLS), ("object", obj, oids), ("case", case, CASES)):
        if not 0 <= idx < len(table):
            raise IndexError(f"plan_ids {name} index {idx} outside 0..{len(table) - 1}")
    if base != NO_BASE and not 0 <= base < len(oids):
        raise IndexError(f"plan_ids base index {base} outside 0..{len(oids) - 1}")
    object_id = oids[obj]
    axes = grip_axes().get(object_id, [])
    if object_id != FOREIGN and not 0 <= grip < len(axes):
        raise IndexError(f"{object_id} has {len(axes)} grip pads, plan_ids grip index is {grip}")
    return {
        "skill": SKILLS[skill],
        "object_id": object_id,
        "base_id": None if base == NO_BASE else oids[base],
        "grip_axis": axes[grip] if object_id != FOREIGN else None,
        "case": CASES[case],
        "product": product_of(object_id),
    }


def encode_plan_ids(skill: str, object_id: str, base_id: str | None, grip: int, case: str) -> list[int]:
    """Return the plan_ids vector for one episode's plan."""
    oids = object_ids()
    return [SKILLS.index(skill), oids.index(object_id), NO_BASE if base_id is None else oids.index(base_id),
            grip, CASES.index(case)]
