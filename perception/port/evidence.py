"""Contract detection and inspection dicts plus the close-up report the controller acts on.
Authored by Shritej Gunda."""

from __future__ import annotations

import math
from dataclasses import dataclass

from perception.port.frames import InstalledPose
from perception.port.judge import Judgement, NextStep
from perception.port.wrist import WristView

# A second look counts as another angle only if the tool tilt differs by at least half of the 30 deg oblique.
MIN_SECOND_LOOK_DEG = 15.0
SECOND_LOOK_STANDOFF_MM = 150.0
SECOND_LOOK_TILT_DEG = 30.0
# Best viewing side per look-alike group. Part-local sides: front = -y, back = +y, left = -x, right = +x.
SECOND_LOOK_SIDE = {
    **{f"M{p}_03": "front" for p in (4, 5, 6)}, **{f"M{p}_04": "left" for p in (4, 5, 6)},
    "M5_02": "front", "M6_02": "front", **{f"M{p}_01": "left" for p in (4, 5, 6)},
    **{f"M{p}_04": "right" for p in (1, 2, 3)}, "M1_01": "front", "M2_01": "front", "M3_01": "front",
}
_SIDE_ANGLE_DEG = {"front": -90.0, "back": 90.0, "left": 180.0, "right": 0.0}
VIEW_SOURCE = "LOOKALIKES.md s2/s3 (T12, geometry only, PRELIMINARY); not a safety-approved motion"


@dataclass(frozen=True, slots=True)
class ViewRequest:
    """Wrist camera viewpoint for the second close-up; the controller plans the motion."""

    instance_id: str
    look_at_uf0_mm: tuple[float, float, float]
    standoff_mm: float
    tilt_from_vertical_deg: float
    from_side: str
    azimuth_uf0_deg: float
    source: str


@dataclass(frozen=True, slots=True)
class CloseupReport:
    """Typed close-up report the controller acts on."""
    instance_id: str
    expected_object_id: str
    next_step: NextStep
    reason: str
    image_refs: tuple[str, ...]          # every view of this decision; ANDON shows both pictures
    second_look: ViewRequest | None


def view_request(instance_id: str, object_id: str, pose: InstalledPose, top_z_mm: float) -> ViewRequest:
    """Return the wrist viewpoint for the second close-up."""
    side = SECOND_LOOK_SIDE.get(object_id, "front")
    azimuth = (pose.yaw_deg + _SIDE_ANGLE_DEG[side] + 180.0) % 360.0 - 180.0
    return ViewRequest(instance_id, (pose.x_mm, pose.y_mm, top_z_mm), SECOND_LOOK_STANDOFF_MM, SECOND_LOOK_TILT_DEG,
                       side, azimuth, VIEW_SOURCE)


def _wrapped(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def another_angle(first: WristView, second: WristView) -> bool:
    """Return True when the second view's tilt differs enough from the first."""
    (_, _, _, w1, p1, _), (_, _, _, w2, p2, _) = first.camera_tcp, second.camera_tcp
    return max(_wrapped(w1, w2), _wrapped(p1, p2)) >= MIN_SECOND_LOOK_DEG


def detection(instance_id: str, view: WristView, object_id: str | None) -> dict:
    """Return the contract detection dict for a wrist view."""
    return {"instance_id": instance_id, "camera": "wrist", "label": view.reading.object_id, "object_id": object_id,
            "score": view.reading.p_identity, "bbox_px": [float(v) for v in view.bbox_px],
            "pose": view.pose.contract_pose(), "pose_source": "foundationpose", "pose_fit_score": view.fit_score,
            "resting_side": view.side}


def inspection(instance_id: str, view: WristView, judgement: Judgement, residual_mm: float | None) -> dict:
    """Return the contract inspection dict for a close-up."""
    residual = residual_mm if residual_mm is not None and math.isfinite(residual_mm) else None
    return {"instance_id": instance_id, "object_id": judgement.object_id, "verdict": judgement.verdict,
            "action": judgement.action, "checks_agree": judgement.checks_agree,
            "evidence": {"classifier_score": view.reading.p_identity, "patchcore_score": None,
                         "depth_residual_mm": residual, "image_ref": view.image_ref}}
