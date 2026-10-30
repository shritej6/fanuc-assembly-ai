"""Validation tests for measured perception acceptance thresholds."""

import pytest

from assembly.verification import VerificationInputError, VerificationThresholds


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_detection_score", -0.01),
        ("min_pose_fit_score", 1.01),
        ("min_classifier_score", float("nan")),
        ("max_patchcore_score", True),
        ("max_depth_residual_mm", -0.01),
    ],
)
def test_thresholds_require_measured_finite_ranges(field: str, value: object) -> None:
    values = {
        "min_detection_score": 0.9,
        "min_pose_fit_score": 0.8,
        "min_classifier_score": 0.9,
        "max_patchcore_score": 0.1,
        "max_depth_residual_mm": 1.0,
    }
    values[field] = value

    with pytest.raises(VerificationInputError, match=field):
        VerificationThresholds(**values)  # type: ignore[arg-type]
