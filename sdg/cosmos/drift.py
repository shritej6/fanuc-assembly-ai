"""Shape-drift check for a Cosmos-restyled image against the render it came from.

The labels of a restyled image are the render's labels, so the restyle may change the look (light, texture, noise)
but not where an edge of a part is. Measure: edge F-score inside the parts' masks. Edges are the strongest gradient
pixels of each image within that region; an edge counts as matched when the other image has one within TOL_PX.
"""

from __future__ import annotations

import numpy as np

TOL_PX, ROI_PAD_PX, EDGE_FRACTION = 2, 3, 0.15
MIN_GRADIENT = 4.0   # grey levels per pixel; below it a pixel is flat even if it ranks among the strongest


def grey(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    """Square dilation by r pixels (numpy only)."""
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def edges(g: np.ndarray, roi: np.ndarray) -> np.ndarray:
    """The EDGE_FRACTION strongest gradient pixels inside roi, ignoring flat pixels."""
    gy, gx = np.gradient(g)
    mag = np.hypot(gx, gy)
    out = np.zeros_like(roi)
    if roi.any():
        out = roi & (mag >= max(float(np.quantile(mag[roi], 1 - EDGE_FRACTION)), MIN_GRADIENT))
    return out


def edge_fscore(source: np.ndarray, restyled: np.ndarray, seg: np.ndarray) -> float:
    """F-score of restyled edges against source edges inside the dilated object masks (seg > 0)."""
    if source.shape[:2] != restyled.shape[:2] or source.shape[:2] != seg.shape:
        raise ValueError(f"sizes differ: source {source.shape}, restyled {restyled.shape}, seg {seg.shape}")
    roi = dilate(seg > 0, ROI_PAD_PX)
    a, b = edges(grey(source), roi), edges(grey(restyled), roi)
    if not a.any() or not b.any():
        return 0.0
    precision = (b & dilate(a, TOL_PX)).sum() / b.sum()
    recall = (a & dilate(b, TOL_PX)).sum() / a.sum()
    return float(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
