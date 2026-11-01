"""Tests for pose interface.
Authored by Shritej Gunda.
"""

import numpy as np
import pytest

from perception.pose.interface import (
    PoseInputError,
    build_pose_request,
    compose_pose_estimate,
)


def _request(camera_to_uf0_m: np.ndarray | None = None):
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth_m = np.ones((4, 5), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 1:4] = True
    intrinsics = np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]])
    camera_to_uf0_m = np.eye(4) if camera_to_uf0_m is None else camera_to_uf0_m
    return build_pose_request(
        object_id="M1_01",
        rgb=rgb,
        depth_m=depth_m,
        object_mask=mask,
        intrinsics=intrinsics,
        camera_to_uf0_m=camera_to_uf0_m,
        known_object_ids={"M1_01"},
        min_valid_depth_fraction=0.5,
    )


def test_pose_request_requires_aligned_calibrated_metric_inputs() -> None:
    request = _request()

    assert request.object_id == "M1_01"
    assert request.rgb.shape[:2] == request.depth_m.shape == request.object_mask.shape
    assert request.valid_depth_fraction == pytest.approx(1.0)
    assert not any(
        array.flags.writeable
        for array in (
            request.rgb,
            request.depth_m,
            request.object_mask,
            request.intrinsics,
            request.camera_to_uf0_m,
        )
    )


def test_pose_request_rejects_misaligned_rgb_depth_and_mask() -> None:
    with pytest.raises(PoseInputError, match="aligned"):
        build_pose_request(
            object_id="M1_01",
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth_m=np.ones((3, 5), dtype=np.float32),
            object_mask=np.ones((4, 5), dtype=bool),
            intrinsics=np.eye(3),
            camera_to_uf0_m=np.eye(4),
            known_object_ids={"M1_01"},
            min_valid_depth_fraction=0.5,
        )


def test_pose_request_rejects_zero_depth_support() -> None:
    with pytest.raises(PoseInputError, match="valid metric depth"):
        build_pose_request(
            object_id="M1_01",
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth_m=np.zeros((4, 5), dtype=np.float32),
            object_mask=np.ones((4, 5), dtype=bool),
            intrinsics=np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]]),
            camera_to_uf0_m=np.eye(4),
            known_object_ids={"M1_01"},
            min_valid_depth_fraction=0.0,
        )


def test_pose_request_rejects_uint16_depth_without_explicit_metric_conversion() -> None:
    with pytest.raises(PoseInputError, match="floating metric depth"):
        build_pose_request(
            object_id="M1_01",
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth_m=np.full((4, 5), 500, dtype=np.uint16),
            object_mask=np.ones((4, 5), dtype=bool),
            intrinsics=np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]]),
            camera_to_uf0_m=np.eye(4),
            known_object_ids={"M1_01"},
            min_valid_depth_fraction=0.5,
        )


def test_pose_request_rejects_nonbinary_instance_mask() -> None:
    instance_mask = np.full((4, 5), 4, dtype=np.uint16)
    instance_mask[:, 3:] = 5

    with pytest.raises(PoseInputError, match="boolean target mask"):
        build_pose_request(
            object_id="M1_01",
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth_m=np.ones((4, 5), dtype=np.float32),
            object_mask=instance_mask,
            intrinsics=np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]]),
            camera_to_uf0_m=np.eye(4),
            known_object_ids={"M1_01"},
            min_valid_depth_fraction=0.5,
        )


@pytest.mark.parametrize("value", [True, "0.5", np.array([0.5]), np.nan, np.inf])
def test_pose_request_wraps_invalid_depth_support_threshold(value: object) -> None:
    with pytest.raises(PoseInputError, match="minimum valid-depth"):
        build_pose_request(
            object_id="M1_01",
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth_m=np.ones((4, 5), dtype=np.float32),
            object_mask=np.ones((4, 5), dtype=bool),
            intrinsics=np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]]),
            camera_to_uf0_m=np.eye(4),
            known_object_ids={"M1_01"},
            min_valid_depth_fraction=value,  # type: ignore[arg-type]
        )


def test_pose_estimate_composes_camera_pose_without_thresholding_score() -> None:
    camera_to_uf0_m = np.eye(4)
    camera_to_uf0_m[:3, 3] = [0.1, 0.2, 0.3]
    request = _request(camera_to_uf0_m)
    object_to_camera_m = np.eye(4)
    object_to_camera_m[:3, 3] = [0.01, 0.02, 0.03]

    estimate = compose_pose_estimate(request, object_to_camera_m, raw_fit_score=0.37)

    assert estimate.raw_fit_score == pytest.approx(0.37)
    assert estimate.object_to_uf0_m[:3, 3] == pytest.approx([0.11, 0.22, 0.33])


def test_pose_request_rejects_nonrigid_extrinsics() -> None:
    camera_to_uf0_m = np.eye(4)
    camera_to_uf0_m[0, 0] = 2.0

    with pytest.raises(PoseInputError, match="rigid"):
        build_pose_request(
            object_id="M1_01",
            rgb=np.zeros((4, 5, 3), dtype=np.uint8),
            depth_m=np.ones((4, 5), dtype=np.float32),
            object_mask=np.ones((4, 5), dtype=bool),
            intrinsics=np.array([[500.0, 0.0, 2.0], [0.0, 500.0, 1.5], [0.0, 0.0, 1.0]]),
            camera_to_uf0_m=camera_to_uf0_m,
            known_object_ids={"M1_01"},
            min_valid_depth_fraction=0.5,
        )


def test_pose_estimate_rejects_nonfinite_fit_score() -> None:
    with pytest.raises(PoseInputError, match="fit score"):
        compose_pose_estimate(_request(), np.eye(4), raw_fit_score=np.nan)
