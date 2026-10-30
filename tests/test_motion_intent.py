"""Semantic motion intents keep object origins, fingertip TCPs, and release authority distinct."""

from __future__ import annotations

import pytest

from assembly.catalog import TRUSTED_GRASP_SOURCE
from assembly.motion_intent import ContactEnvelope, MotionIntent, MotionIntentError, SemanticSkill


def _intent(**changes) -> MotionIntent:
    values = {
        "skill": SemanticSkill.PICK_INSERT_HOLD,
        "instance_id": "part-4-2",
        "object_id": "M4_02",
        "base_instance_id": "base-4",
        "base_object_id": "F04",
        "grip_index": 0,
        "case": "nominal",
        "object_pose": (300.0, -50.0, 37.0),
        "object_origin_z_mm": -283.9,
        "wrist_roll_deg": 127.0,
        "goal_pose": (400.0, 50.0, 90.0),
        "grasp_tcp": (298.916732, -48.562456, -270.0, -180.0, 0.0, 127.0),
        "release_tcp": None,
        "endpoint_tcp": (401.8, 50.0, -301.2, -180.0, 0.0, 90.0),
        "final_gripper": 1.0,
        "pose_source": "T30 FoundationPose + T29 grip 0",
        "grasp_source_sha256": TRUSTED_GRASP_SOURCE.sha256,
        "contact": ContactEnvelope(
            400.0,
            50.0,
            -301.2,
            18.0,
            "M4_02 on F04",
            "cad/grasp_frames.json + measured F04 seat",
        ),
        "release_authorized": False,
    }
    values.update(changes)
    return MotionIntent(**values)


def test_plan_target_uses_object_origin_not_fingertip_tcp() -> None:
    intent = _intent()
    assert intent.plan_target == (300.0, -50.0, 37.0, 127.0, 400.0, 50.0, 90.0)
    assert intent.grasp_tcp[:3] != intent.plan_target[:3]
    assert intent.endpoint_tcp[:3] != intent.goal_pose
    assert intent.release_authorized is False


def test_only_verified_release_intent_authorizes_opening() -> None:
    release = _intent(
        skill=SemanticSkill.RELEASE_RETREAT,
        release_authorized=True,
        release_tcp=(401.8, 50.0, -301.2, -180.0, 0.0, 90.0),
        final_gripper=0.0,
    )
    assert release.release_authorized is True
    with pytest.raises(MotionIntentError, match="defect_drop"):
        _intent(skill=SemanticSkill.DEFECT_DROP)


def test_pick_insert_can_never_authorize_release_in_the_same_skill() -> None:
    with pytest.raises(MotionIntentError, match="hold"):
        _intent(release_authorized=True)


@pytest.mark.parametrize(
    "change",
    [
        {"instance_id": ""},
        {"grip_index": -1},
        {"grip_index": 999},
        {"object_pose": (1.0, 2.0, float("nan"))},
        {"object_origin_z_mm": float("nan")},
        {"grasp_tcp": (1.0, 2.0, 3.0)},
        {"grasp_tcp": (301.8, -50.0, -270.0, -180.0, 0.0, 37.0)},
        {"grasp_source_sha256": "0" * 64},
        {"release_tcp": (1.0, 2.0, 3.0)},
        {"endpoint_tcp": (1.0, 2.0, 3.0)},
        {"final_gripper": 0.5},
        {"pose_source": ""},
        {"case": ""},
        {"base_instance_id": None},
        {"base_object_id": ""},
        {"endpoint_tcp": (450.0, 50.0, -301.2, -180.0, 0.0, 90.0)},
    ],
)
def test_malformed_intent_fails_closed(change: dict) -> None:
    with pytest.raises(MotionIntentError):
        _intent(**change)
