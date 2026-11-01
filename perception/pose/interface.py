"""Calibrated, metric input and output boundary for FoundationPose inference.
Authored by Shritej Gunda."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real

import numpy as np


class PoseInputError(ValueError):
    """Raised when pose inference inputs lack safe alignment or calibration."""


@dataclass(frozen=True)
class PoseRequest:
    """Metric inputs for one pose estimate."""
    object_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    object_mask: np.ndarray
    intrinsics: np.ndarray
    camera_to_uf0_m: np.ndarray
    valid_depth_fraction: float


@dataclass(frozen=True)
class PoseEstimate:
    """One metric pose estimate with its fit score."""
    object_id: str
    object_to_camera_m: np.ndarray
    object_to_uf0_m: np.ndarray
    raw_fit_score: float


def _readonly(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    copied = np.array(array, dtype=dtype, copy=True)
    copied.flags.writeable = False
    return copied


def _rigid_transform(value: np.ndarray, name: str) -> np.ndarray:
    try:
        transform = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PoseInputError(f"{name} must be a finite rigid 4x4 transform") from exc
    valid_shape = transform.shape == (4, 4) and np.isfinite(transform).all()
    valid_tail = valid_shape and np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
    rotation = transform[:3, :3] if valid_shape else np.empty((0, 0))
    valid_rotation = valid_shape and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
    valid_rotation = valid_rotation and np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    if not (valid_shape and valid_tail and valid_rotation):
        raise PoseInputError(f"{name} must be a finite rigid 4x4 transform")
    return _readonly(transform, np.dtype(np.float64))


def _camera_intrinsics(value: np.ndarray) -> np.ndarray:
    try:
        intrinsics = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PoseInputError("intrinsics must be a calibrated finite 3x3 matrix") from exc
    valid = intrinsics.shape == (3, 3) and np.isfinite(intrinsics).all()
    valid = valid and intrinsics[0, 0] > 0 and intrinsics[1, 1] > 0
    valid = valid and np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-8)
    if not valid:
        raise PoseInputError("intrinsics must be a calibrated finite 3x3 matrix")
    return _readonly(intrinsics, np.dtype(np.float64))


def _depth_support_threshold(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise PoseInputError("minimum valid-depth fraction must be a finite real scalar")
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise PoseInputError("minimum valid-depth fraction must be between zero and one")
    return threshold


def _aligned_sensor_arrays(
    rgb: np.ndarray, depth_m: np.ndarray, object_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        rgb_array = np.asarray(rgb)
        depth_array = np.asarray(depth_m)
        mask_array = np.asarray(object_mask)
    except (TypeError, ValueError) as exc:
        raise PoseInputError("RGB, depth, and object mask must be array-like") from exc
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3 or rgb_array.dtype != np.uint8:
        raise PoseInputError("RGB must be an HxWx3 uint8 image")
    if depth_array.ndim != 2 or not np.issubdtype(depth_array.dtype, np.floating):
        raise PoseInputError("depth must be a floating metric depth array")
    if mask_array.dtype != np.bool_:
        raise PoseInputError("object mask must be a boolean target mask")
    if mask_array.shape != depth_array.shape or rgb_array.shape[:2] != depth_array.shape:
        raise PoseInputError("RGB, depth, and object mask must be pixel-aligned")
    return rgb_array, depth_array, mask_array


def build_pose_request(
    *,
    object_id: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_uf0_m: np.ndarray,
    known_object_ids: Iterable[str],
    min_valid_depth_fraction: float,
) -> PoseRequest:
    """Validate aligned sensor evidence and calibration before model inference."""
    if object_id not in set(known_object_ids):
        raise PoseInputError(f"unknown pose object ID: {object_id}")
    threshold = _depth_support_threshold(min_valid_depth_fraction)
    rgb_array, depth_array, mask_array = _aligned_sensor_arrays(rgb, depth_m, object_mask)
    if not mask_array.any():
        raise PoseInputError("object mask must contain target pixels")
    valid = mask_array & np.isfinite(depth_array) & (depth_array > 0)
    valid_fraction = float(valid.sum() / mask_array.sum())
    if not valid.any() or valid_fraction < threshold:
        raise PoseInputError("object mask lacks sufficient valid metric depth")
    return PoseRequest(
        object_id,
        _readonly(rgb_array, np.dtype(np.uint8)),
        _readonly(depth_array, np.dtype(np.float32)),
        _readonly(mask_array, np.dtype(bool)),
        _camera_intrinsics(intrinsics),
        _rigid_transform(camera_to_uf0_m, "camera_to_uf0_m"),
        valid_fraction,
    )


def compose_pose_estimate(
    request: PoseRequest, object_to_camera_m: np.ndarray, *, raw_fit_score: float
) -> PoseEstimate:
    """Compose the model result into calibrated UF0 while preserving the raw score."""
    object_to_camera = _rigid_transform(object_to_camera_m, "object_to_camera_m")
    try:
        score = float(raw_fit_score)
    except (TypeError, ValueError) as exc:
        raise PoseInputError("raw fit score must be finite") from exc
    if not np.isfinite(score):
        raise PoseInputError("raw fit score must be finite")
    object_to_uf0 = _rigid_transform(request.camera_to_uf0_m @ object_to_camera, "object_to_uf0_m")
    return PoseEstimate(request.object_id, object_to_camera, object_to_uf0, score)
