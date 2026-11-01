"""Tests for depth residual.
Authored by Shritej Gunda.
"""

import numpy as np
import pytest

from perception.depth.residual import DepthResidualError, measure_depth_residual


def test_depth_residual_excludes_zero_returns_and_reports_raw_evidence() -> None:
    observed = np.array([[1.0, 0.0], [1.2, 1.1]], dtype=np.float32)
    expected = np.array([[0.9, 1.0], [1.0, 0.0]], dtype=np.float32)
    mask = np.ones((2, 2), dtype=bool)

    result = measure_depth_residual(observed, expected, mask, min_valid_fraction=0.5)

    assert result.status == "measured"
    assert result.valid_pixels == 2
    assert result.mask_pixels == 4
    assert result.valid_fraction == pytest.approx(0.5)
    assert result.missing_return_fraction == pytest.approx(0.25)
    assert result.mean_signed_m == pytest.approx(0.15)
    assert result.mean_absolute_m == pytest.approx(0.15)


def test_depth_residual_is_uncertain_when_valid_support_is_insufficient() -> None:
    observed = np.array([[1.0, 0.0], [1.2, 1.1]], dtype=np.float32)
    expected = np.array([[0.9, 1.0], [1.0, 0.0]], dtype=np.float32)

    result = measure_depth_residual(
        observed,
        expected,
        np.ones((2, 2), dtype=bool),
        min_valid_fraction=0.75,
    )

    assert result.status == "uncertain"
    assert result.mean_absolute_m == pytest.approx(0.15)


def test_depth_residual_is_uncertain_with_zero_comparable_pixels() -> None:
    result = measure_depth_residual(
        np.zeros((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=np.float32),
        np.ones((2, 2), dtype=bool),
        min_valid_fraction=0.0,
    )

    assert result.status == "uncertain"
    assert result.valid_pixels == 0
    assert result.mean_absolute_m is None


def test_depth_residual_rejects_misaligned_inputs() -> None:
    with pytest.raises(DepthResidualError, match="same shape"):
        measure_depth_residual(
            np.ones((2, 2), dtype=np.float32),
            np.ones((2, 3), dtype=np.float32),
            np.ones((2, 2), dtype=bool),
            min_valid_fraction=0.5,
        )
