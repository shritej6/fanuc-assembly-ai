"""One recorded episode per skill, in the shape agent 4's mimic/skills.yaml expects.

Every episode starts at the scan (home) pose and ends back there:
- pick_insert : approach_grasp -> grasped -> carry_insert -> seated -> release_retreat
- pull_out    : approach_grasp -> grasped -> pull_free -> pulled -> carry_set_down -> placed -> release_retreat
- place_free  : approach_grasp -> grasped -> carry_set_down -> placed -> release_retreat
- defect_drop : approach_grasp -> grasped -> carry_drop -> dropped -> retreat
The instruction carries the planner's target (mimic/instructions.py) so GR00T follows the plan.
"""

from __future__ import annotations

import numpy as np

from ..mimic.catalog import encode_plan_ids, product_of
from ..mimic.instructions import instruction
from . import checks, correct
from .geometry import matrix_to_pose, pose_to_matrix, wrap_deg
from .plan import FreePlacePlan, InsertPlan, PickPlan, PullOutPlan
from .runner import EpisodeFailed, EpisodeRecorder


def _part_xy_yaw(rec: EpisodeRecorder, oid: str) -> list[float]:
    p = checks.object_pose(rec.obs, oid)
    return [p[0], p[1], p[5]]


def _spot_yaw(rec: EpisodeRecorder, oid: str, grasp_r: float, spot: FreePlacePlan) -> float:
    return wrap_deg(checks.object_pose(rec.obs, oid)[5] + spot.spot_tcp[5] - grasp_r)


def _begin(rec: EpisodeRecorder, skill: str, oid: str, base: str | None, grip: int, case: str,
           target: list[float]) -> None:
    ids = encode_plan_ids(skill, oid, base, grip, case)
    rec.meta = {"skill": skill, "case": case, "object_id": oid, "base_id": base or "",
                "product": product_of(oid) or 0, "plan_ids": ids, "plan_target": [float(v) for v in target]}
    rec.say(instruction(ids, target))


def _end(rec: EpisodeRecorder, required: tuple[str, ...]) -> None:
    rec.go_home()
    rec.close()
    rec.finish_gates()
    missing = [s for s in required if s not in rec.marks]
    if missing:
        raise EpisodeFailed(f"episode ended without the subtask signals {missing}")


def pick_insert(rec: EpisodeRecorder, pick: PickPlan, ins: InsertPlan, case: str = "nominal",
                misalign=(0.0, 0.0, 0.0)) -> None:
    """`misalign` (dx, dy mm, dyaw deg) stages a won't-seat first try; the retry must then seat it."""
    base_pose = checks.object_pose(rec.obs, ins.base_id)
    target = _part_xy_yaw(rec, pick.object_id) + [pick.grasp_tcp[5], base_pose[0], base_pose[1], base_pose[5]]
    _begin(rec, "pick_insert", pick.object_id, ins.base_id, pick.grip_index, case, target)
    rec.define_frame("grasp_frame", pick.object_id, pick.grasp_tcp)
    rec.define_frame("seat_frame", ins.base_id, ins.seated_tcp)
    rec.do_pick(pick)
    outcome = rec.do_insert(ins, misalign=misalign)
    if outcome != "seated":
        raise EpisodeFailed(f"{pick.object_id}: {outcome}")
    if any(misalign) and not rec.events:
        raise EpisodeFailed("staged won't-seat case seated on the first try: not a recovery demo")
    _end(rec, ("grasped", "seated"))


def align_insert(rec: EpisodeRecorder, pick: PickPlan, ins: InsertPlan, rng: np.random.Generator) -> None:
    """Correction demo: masked pick + carry to a wrong pose above the base; learned line-up + entry from there.
    Case stays nominal: the instruction must read exactly like the planner's (no "[case]" tag at deploy)."""
    base_pose = checks.object_pose(rec.obs, ins.base_id)
    target = _part_xy_yaw(rec, pick.object_id) + [pick.grasp_tcp[5], base_pose[0], base_pose[1], base_pose[5]]
    _begin(rec, "pick_insert", pick.object_id, ins.base_id, pick.grip_index, "nominal", target)
    rec.define_frame("grasp_frame", pick.object_id, pick.grasp_tcp)
    rec.define_frame("seat_frame", ins.base_id, ins.seated_tcp)
    rec.do_pick(pick, learn=False)
    rec.run(correct.stage_near_base(ins, rec.pose, rec.gripper, correct.staged_insert_pose(ins, (0.0, 0.0, 0.0), rng)))
    outcome = rec.do_insert(ins, near=True)
    if outcome != "seated":
        raise EpisodeFailed(f"{pick.object_id}: {outcome}")
    _end(rec, ("grasped", "seated"))


def align_grasp(rec: EpisodeRecorder, pick: PickPlan, ins: InsertPlan, rng: np.random.Generator) -> None:
    """Correction demo: masked move to a wrong pose above the grip pad; learned line-up + grasp + normal insert."""
    base_pose = checks.object_pose(rec.obs, ins.base_id)
    target = _part_xy_yaw(rec, pick.object_id) + [pick.grasp_tcp[5], base_pose[0], base_pose[1], base_pose[5]]
    _begin(rec, "pick_insert", pick.object_id, ins.base_id, pick.grip_index, "nominal", target)
    rec.define_frame("grasp_frame", pick.object_id, pick.grasp_tcp)
    rec.define_frame("seat_frame", ins.base_id, ins.seated_tcp)
    rec.run(correct.stage_near_pad(pick, rec.pose, correct.staged_grasp_pose(pick, rng)))
    rec.do_pick(pick, near=True)
    outcome = rec.do_insert(ins)
    if outcome != "seated":
        raise EpisodeFailed(f"{pick.object_id}: {outcome}")
    _end(rec, ("grasped", "seated"))


def pull_out(rec: EpisodeRecorder, pull: PullOutPlan, spot: FreePlacePlan, case: str = "fix_above") -> None:
    if spot.kind != "test_bed":
        raise EpisodeFailed("a pulled-out good part goes to a test-bed spot")
    target = _part_xy_yaw(rec, pull.object_id) + [pull.grasp_tcp[5], spot.spot_tcp[0], spot.spot_tcp[1],
                                                  _spot_yaw(rec, pull.object_id, pull.grasp_tcp[5], spot)]
    _begin(rec, "pull_out", pull.object_id, None, pull.grip_index, case, target)
    rec.define_frame("grasp_frame", pull.object_id, pull.grasp_tcp)
    rec.define_frame("spot_frame", None, spot.spot_tcp)
    rec.do_pull_out(pull)
    rec.do_place_free(spot)
    _end(rec, ("grasped", "pulled", "placed"))


def place_free(rec: EpisodeRecorder, pick: PickPlan, spot: FreePlacePlan, case: str = "crowded") -> None:
    if spot.kind != "test_bed":
        raise EpisodeFailed("place_free sets a good part down on a test-bed spot")
    target = _part_xy_yaw(rec, pick.object_id) + [pick.grasp_tcp[5], spot.spot_tcp[0], spot.spot_tcp[1],
                                                  _spot_yaw(rec, pick.object_id, pick.grasp_tcp[5], spot)]
    _begin(rec, "place_free", pick.object_id, None, pick.grip_index, case, target)
    rec.define_frame("grasp_frame", pick.object_id, pick.grasp_tcp)
    rec.define_frame("spot_frame", None, spot.spot_tcp)
    rec.do_pick(pick)
    rec.do_place_free(spot)
    _end(rec, ("grasped", "placed"))


def defect_drop(rec: EpisodeRecorder, pick: PickPlan, area: FreePlacePlan, case: str = "defect") -> None:
    if area.kind != "defect":
        raise EpisodeFailed("defect_drop needs the defect-area plan")
    target = _part_xy_yaw(rec, pick.object_id) + [pick.grasp_tcp[5], area.spot_tcp[0], area.spot_tcp[1], 0.0]
    _begin(rec, "defect_drop", pick.object_id, None, pick.grip_index, case, target)
    rec.define_frame("grasp_frame", pick.object_id, pick.grasp_tcp)
    rec.define_frame("spot_frame", None, area.spot_tcp)
    rec.do_pick(pick)
    rec.do_place_free(area)
    _end(rec, ("grasped", "dropped"))


def object_frame_tcp(obj_pose, t_obj_tcp: np.ndarray) -> np.ndarray:
    """TCP pose from an object pose and an object->TCP transform (used by the planner adapters)."""
    return matrix_to_pose(pose_to_matrix(obj_pose) @ t_obj_tcp)
