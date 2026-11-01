"""Measure the sizes that tell look-alike objects apart from their top-down height maps.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from collections import deque

import numpy as np

STEP_MM = 0.4            # a top-surface step smaller than this is not a cue
MIN_AREA_MM2 = 2.0       # ignore slivers from tessellation
WRIST_MM_PER_PX = (0.24, 0.49)   # PLAN 3.2: wrist close-up resolution on black parts


def groups(taxonomy: dict) -> list[list[str]]:
    """Connected components of the taxonomy's look-alike lists (singletons dropped)."""
    adj = {o["object_id"]: set(o["lookalikes"]) for o in taxonomy["objects"]}
    seen, out = set(), []
    for start in sorted(adj):
        if start in seen or not adj[start]:
            continue
        comp, todo = set(), [start]
        while todo:
            n = todo.pop()
            if n not in comp:
                comp.add(n)
                todo.extend(adj[n] - comp)
        seen |= comp
        out.append(sorted(comp))
    return out


def _on_grid(hm: dict, lo: np.ndarray, shape: tuple[int, int], floor: float) -> np.ndarray:
    out = np.full(shape, floor)
    i0 = round((hm["origin"][0] - lo[0]) / hm["res"])
    j0 = round((hm["origin"][1] - lo[1]) / hm["res"])
    top = np.where(np.isnan(hm["top"]), floor, hm["top"])
    out[i0:i0 + top.shape[0], j0:j0 + top.shape[1]] = top
    return out


def _regions(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean grid, as (n, 2) index arrays."""
    label = np.zeros(mask.shape, int)
    out = []
    for seed in zip(*np.nonzero(mask)):
        if label[seed]:
            continue
        label[seed] = len(out) + 1
        cells, todo = [], deque([seed])
        while todo:
            i, j = todo.popleft()
            cells.append((i, j))
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not label[a, b]:
                    label[a, b] = label[seed]
                    todo.append((a, b))
        out.append(np.array(cells))
    return out


def compare(a: dict, b: dict, floor_a: float, floor_b: float) -> list[dict]:
    """Cues where a's top surface differs from b's, largest first."""
    res = a["res"]
    lo = np.minimum(a["origin"], b["origin"])
    hi = np.maximum(np.array(a["origin"]) + np.array(a["top"].shape) * res,
                    np.array(b["origin"]) + np.array(b["top"].shape) * res)
    shape = tuple(round(v) for v in (hi - lo) / res)
    ta, tb = _on_grid(a, lo, shape, floor_a), _on_grid(b, lo, shape, floor_b)
    cues = []
    for cells in _regions(np.abs(ta - tb) > STEP_MM):
        area = len(cells) * res * res
        if area < MIN_AREA_MM2:
            continue
        x = lo[0] + (cells[:, 0] + 0.5) * res
        y = lo[1] + (cells[:, 1] + 0.5) * res
        w, length = np.ptp(x) + res, np.ptp(y) + res
        step = float(np.median(np.abs(ta - tb)[cells[:, 0], cells[:, 1]]))
        size = min(w, length, area / max(w, length), step)
        cues.append({"region_xy_mm": [round(float(x.min()) - res / 2, 2), round(float(y.min()) - res / 2, 2),
                                      round(float(x.max()) + res / 2, 2), round(float(y.max()) + res / 2, 2)],
                     "area_mm2": round(area, 2), "top_z_here_mm": round(float(np.median(ta[cells[:, 0], cells[:, 1]])), 2),
                     "top_z_other_mm": round(float(np.median(tb[cells[:, 0], cells[:, 1]])), 2),
                     "size_mm": round(float(size), 2)})
    return sorted(cues, key=lambda c: (-c["size_mm"], -c["area_mm2"]))


def pair_report(a_id: str, b_id: str, catalog: dict, maps: dict, n_cues: int = 4) -> dict:
    sa, sb = np.array(catalog[a_id]["bbox_mm"]["size"]), np.array(catalog[b_id]["bbox_mm"]["size"])
    dbox = np.round(sa - sb, 2)
    cues = compare(maps[a_id], maps[b_id], catalog[a_id]["bbox_mm"]["min"][2], catalog[b_id]["bbox_mm"]["min"][2])
    best = max([float(np.abs(dbox).max())] + [c["size_mm"] for c in cues])
    return {"bbox_size_diff_mm": dbox.tolist(), "best_cue_mm": round(best, 2), "identical": best == 0.0,
            "cues": cues[:n_cues], "volume_diff_mm3": round(catalog[a_id]["volume_mm3"] - catalog[b_id]["volume_mm3"], 1)}


def distinguishing(oid: str, group: list[str], catalog: dict, maps: dict) -> dict:
    vs = {other: pair_report(oid, other, catalog, maps) for other in group if other != oid}
    return {"group": group, "bbox_size_mm": catalog[oid]["bbox_mm"]["size"],
            "min_measurable_difference_mm": min(v["best_cue_mm"] for v in vs.values()), "vs": vs}


def group_summary(group: list[str], catalog: dict, maps: dict) -> dict:
    pairs = {f"{a}~{b}": pair_report(a, b, catalog, maps, n_cues=1)
             for i, a in enumerate(group) for b in group[i + 1:]}
    worst = min(p["best_cue_mm"] for p in pairs.values())
    return {"members": group, "min_measurable_difference_mm": worst,
            "wrist_px_at_0.24_0.49_mm_per_px": [round(worst / r, 1) for r in WRIST_MM_PER_PX],
            "pairs": {k: {"best_cue_mm": v["best_cue_mm"], "identical": v["identical"],
                          "bbox_size_diff_mm": v["bbox_size_diff_mm"], "top_cue": v["cues"][:1]}
                      for k, v in pairs.items()}}


def size_twins(catalog: dict, maps: dict, groups_: list[list[str]] = (), tol_mm: float = 1.0) -> list[dict]:
    """Every pair whose bounding boxes agree within tol on all three axes, with the best top-surface cue."""
    ids = sorted(catalog)
    out = []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = np.abs(np.array(catalog[a]["bbox_mm"]["size"]) - np.array(catalog[b]["bbox_mm"]["size"]))
            if d.max() <= tol_mm:
                rep = pair_report(a, b, catalog, maps, n_cues=1)
                grouped = any(a in g and b in g for g in groups_)
                out.append({"a": a, "b": b, "in_taxonomy_lookalike_group": grouped,
                            "bbox_size_diff_mm": np.round(d, 2).tolist(),
                            "best_cue_mm": rep["best_cue_mm"], "top_cue": rep["cues"]})
    return out
