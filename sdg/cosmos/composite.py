"""Put the rendered parts back into a Cosmos-restyled image.

Cosmos Transfer makes the table, robot, background and light look real, but it redraws the fine features of small
parts (pins and recesses moved, merged or invented; pale discs erased) at every setting tried on 2026-10-28: default
and lowest edge threshold, vis control, guided generation with the part masks. The labels describe the rendered
parts, so the parts must stay rendered: each object's pixels (plus a feathered margin) come from the render, relit
with the light change Cosmos made to the background, and everything else comes from Cosmos.
"""

from __future__ import annotations

import numpy as np

from sdg.cosmos.drift import dilate

FEATHER_PX = 4                 # soft seam between the rendered part and the Cosmos surroundings
GAIN_RANGE = (0.5, 1.5)        # a per-channel light change outside this is not light, so it is clamped
MAX_FIT_PIXELS = 200_000


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, edges clamped (numpy only)."""
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r + 1, r)
        c = np.cumsum(np.pad(out, pad, mode="edge"), axis=axis)
        hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
        out = (hi - lo) / (2 * r + 1)
    return out


def light_change(source: np.ndarray, restyled: np.ndarray, background: np.ndarray) -> np.ndarray:
    """Per-channel gain that maps the render's background onto Cosmos's (least squares through zero). Light scales
    what a surface reflects, so a gain is the physical model; an offset fit on the near-uniform white table is not
    determined and pushed black parts to 0 in testing."""
    src = source[background].astype(np.float64)
    out = restyled[background].astype(np.float64)
    if len(src) > MAX_FIT_PIXELS:
        keep = np.random.default_rng(0).choice(len(src), MAX_FIT_PIXELS, replace=False)
        src, out = src[keep], out[keep]
    energy = (src * src).sum(axis=0)
    gain = np.where(energy > 0, (src * out).sum(axis=0) / np.maximum(energy, 1e-9), 1.0)
    return np.clip(gain, *GAIN_RANGE).astype(np.float32)


def composite(source: np.ndarray, restyled: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Cosmos image with every labelled object replaced by its relit render pixels (exact inside the mask)."""
    if source.shape != restyled.shape or source.shape[:2] != seg.shape:
        raise ValueError(f"sizes differ: source {source.shape}, restyled {restyled.shape}, seg {seg.shape}")
    parts = seg > 0
    if not parts.any():
        return restyled.copy()
    gain = light_change(source, restyled, ~dilate(parts, 2 * FEATHER_PX))
    relit = np.clip(source.astype(np.float32) * gain, 0, 255)
    alpha = np.maximum(parts, np.clip(box_blur(dilate(parts, FEATHER_PX).astype(np.float32), FEATHER_PX), 0, 1))
    blended = alpha[..., None] * relit + (1 - alpha[..., None]) * restyled.astype(np.float32)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)
