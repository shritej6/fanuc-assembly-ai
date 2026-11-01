"""Build the language instruction for one episode from the planner's target.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np

from .catalog import decode_plan_ids, product_bases

# PLAN section 3.2 declares these physical fixtures interchangeable. Language
# still names the canonical base seen in each product's training episodes.
BASE_GROUPS = (("F01", "F02"), ("F03",), ("F04", "F05", "F06"))


def _f(v: float) -> str:
    return f"{round(float(v))}"


def _named_base(plan: dict) -> str:
    """Return the trained base label without changing the physical goal pose."""
    physical = plan["base_id"]
    canonical = product_bases().get(plan["product"])
    if canonical is None or physical == canonical:
        return physical
    return canonical if any(physical in group and canonical in group for group in BASE_GROUPS) else physical


def instruction(plan_ids, plan_target) -> str:
    """plan_target = [part x, y (mm), yaw (deg), roll r (deg), base/spot x, y (mm), yaw (deg)]."""
    p = decode_plan_ids(plan_ids)
    t = np.asarray(plan_target, dtype=np.float64)
    if t.shape != (7,) or not np.isfinite(t).all():
        raise ValueError(f"plan_target must be 7 finite numbers, got {t.tolist()}")
    part = f"{p['object_id']} at ({_f(t[0])}, {_f(t[1])}, {_f(t[2])})"
    grip = f"gripping {p['grip_axis']} at roll {_f(t[3])}" if p["grip_axis"] else f"at roll {_f(t[3])}"
    goal = f"({_f(t[4])}, {_f(t[5])}, {_f(t[6])})"
    skill = p["skill"]
    if skill == "pick_insert":
        if p["base_id"] is None:
            raise ValueError("pick_insert needs a base id")
        text = f"pick {part} {grip}, insert on {_named_base(p)} at {goal}"
    elif skill == "pull_out":
        text = f"pull out {part} {grip}, set it down at {goal}"
    elif skill == "place_free":
        text = f"pick {part} {grip}, set it down at {goal}"
    else:
        text = f"pick {part} {grip}, drop it in the defect area at {goal}"
    return text if p["case"] == "nominal" else f"{text} [{p['case']}]"
