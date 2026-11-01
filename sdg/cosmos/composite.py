"""Put rendered parts back into a Cosmos-restyled image where Cosmos changed them.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np

from sdg.cosmos.drift import dilate, object_scores

FEATHER_PX = 4                 # soft seam between the rendered part and the Cosmos surroundings
KEEP_COSMOS_MIN = 0.85         # per-object edge F-score at which an object may keep its Cosmos look
MAX_COLOUR_SHIFT = 35.0        # grey levels, per channel, between Cosmos's and the relit render's mean object colour
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
    """Return the per-channel gain that maps the render background onto the Cosmos image."""
    src = source[background].astype(np.float64)
    out = restyled[background].astype(np.float64)
    if len(src) > MAX_FIT_PIXELS:
        keep = np.random.default_rng(0).choice(len(src), MAX_FIT_PIXELS, replace=False)
        src, out = src[keep], out[keep]
    energy = (src * src).sum(axis=0)
    gain = np.where(energy > 0, (src * out).sum(axis=0) / np.maximum(energy, 1e-9), 1.0)
    return np.clip(gain, *GAIN_RANGE).astype(np.float32)


def colour_shift(relit: np.ndarray, restyled: np.ndarray, mask: np.ndarray) -> float:
    """Largest per-channel difference of the object's mean colour (grey levels)."""
    return float(np.abs(restyled[mask].astype(np.float32).mean(axis=0) - relit[mask].mean(axis=0)).max())


def composite(source: np.ndarray, restyled: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return the Cosmos image with changed objects replaced by relit render pixels, and the kept ids."""
    if source.shape != restyled.shape or source.shape[:2] != seg.shape:
        raise ValueError(f"sizes differ: source {source.shape}, restyled {restyled.shape}, seg {seg.shape}")
    gain = light_change(source, restyled, ~dilate(seg > 0, 2 * FEATHER_PX))
    relit = np.clip(source.astype(np.float32) * gain, 0, 255)
    kept = sorted(uid for uid, score in object_scores(source, restyled, seg).items()
                  if score >= KEEP_COSMOS_MIN and colour_shift(relit, restyled, seg == uid) <= MAX_COLOUR_SHIFT)
    parts = (seg > 0) & ~np.isin(seg, kept)
    if not parts.any():
        return restyled.copy(), kept
    alpha = np.maximum(parts, np.clip(box_blur(dilate(parts, FEATHER_PX).astype(np.float32), FEATHER_PX), 0, 1))
    blended = alpha[..., None] * relit + (1 - alpha[..., None]) * restyled.astype(np.float32)
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8), kept
