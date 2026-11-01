"""Raw observed-versus-rendered depth evidence, with no unmeasured verdict threshold.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


class DepthResidualError(ValueError):
    """Raised when depth evidence cannot be aligned or interpreted safely."""


@dataclass(frozen=True)
class DepthResidual:
    """Measured depth residual evidence for one mask."""
    status: Literal["measured", "uncertain"]
    valid_pixels: int
    mask_pixels: int
    valid_fraction: float
    missing_return_fraction: float
    mean_signed_m: float | None
    mean_absolute_m: float | None


def measure_depth_residual(
    observed_m: np.ndarray,
    expected_m: np.ndarray,
    object_mask: np.ndarray,
    *,
    min_valid_fraction: float,
) -> DepthResidual:
    """Measure metric-depth residuals and expose insufficient support as uncertain."""
    observed = np.asarray(observed_m, dtype=np.float64)
    expected = np.asarray(expected_m, dtype=np.float64)
    mask = np.asarray(object_mask, dtype=bool)
    if observed.shape != expected.shape or observed.shape != mask.shape:
        raise DepthResidualError("observed, expected, and mask arrays must have the same shape")
    if observed.ndim != 2 or not 0.0 <= min_valid_fraction <= 1.0:
        raise DepthResidualError("depth must be 2D and min_valid_fraction must be within [0, 1]")
    mask_pixels = int(mask.sum())
    if mask_pixels == 0:
        raise DepthResidualError("object mask must contain at least one pixel")
    observed_valid = np.isfinite(observed) & (observed > 0)
    expected_valid = np.isfinite(expected) & (expected > 0)
    valid = mask & observed_valid & expected_valid
    valid_pixels = int(valid.sum())
    residual = observed[valid] - expected[valid]
    valid_fraction = valid_pixels / mask_pixels
    missing_fraction = float((mask & ~observed_valid).sum() / mask_pixels)
    status = "measured" if valid_pixels and valid_fraction >= min_valid_fraction else "uncertain"
    return DepthResidual(
        status=status,
        valid_pixels=valid_pixels,
        mask_pixels=mask_pixels,
        valid_fraction=valid_fraction,
        missing_return_fraction=missing_fraction,
        mean_signed_m=float(residual.mean()) if valid_pixels else None,
        mean_absolute_m=float(np.abs(residual).mean()) if valid_pixels else None,
    )
