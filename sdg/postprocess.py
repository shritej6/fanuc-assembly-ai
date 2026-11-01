"""Post-process rendered images and assemble scene labels, using numpy only.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import numpy as np

DEPTH_MAX_MM = 65535


def uid_number(uid: str) -> int:
    """'o07' -> 8. 0 is background in the instance mask."""
    return int(uid[1:]) + 1


def uid_for_prim(prim_path: str, paths: dict[str, str]) -> str | None:
    """Map an annotator prim path (object Xform or any child of it) to the scene uid."""
    for root, uid in paths.items():
        if prim_path == root or prim_path.startswith(root + "/"):
            return uid
    return None


def instance_mask(ids: np.ndarray, id_to_prim: dict, paths: dict[str, str]) -> np.ndarray:
    """Re-key an instance-segmentation image from renderer ids to uid numbers (uint16; robot, table etc. -> 0)."""
    out = np.zeros(ids.shape, dtype=np.uint16)
    for raw_id, prim in id_to_prim.items():
        uid = uid_for_prim(str(prim), paths)
        if uid is not None:
            out[ids == int(raw_id)] = uid_number(uid)
    return out


def box_labels(boxes: np.ndarray, prim_paths: list[str], paths: dict[str, str], objects: dict[str, dict],
               min_side_px: int = 2) -> list[dict]:
    """Tight 2D boxes joined with the spec, so every label carries id, kind, variant and build state."""
    out = []
    for row, prim in zip(boxes, prim_paths):
        uid = uid_for_prim(str(prim), paths)
        if uid is None:
            continue
        x0, y0, x1, y1 = (int(row[k]) for k in ("x_min", "y_min", "x_max", "y_max"))
        if x1 - x0 < min_side_px or y1 - y0 < min_side_px:
            continue
        o = objects[uid]
        out.append({"uid": uid, "object_id": o["object_id"] or f"foreign_{o['foreign_type']}", "kind": o["kind"],
                    "state": o["state"], "variant": o["variant"], "build_state": o["build_state"],
                    "bbox_xyxy": [x0, y0, x1, y1], "occlusion": round(float(row["occlusionRatio"]), 4)})
    return out


def mask_box_labels(mask: np.ndarray, objects: dict[str, dict], min_pixels: int = 20) -> list[dict]:
    """Return tight boxes and visible pixel counts from the uid instance mask."""
    by_number = {uid_number(uid): (uid, o) for uid, o in objects.items()}
    out = []
    for number in np.unique(mask):
        if number == 0 or number not in by_number:
            continue
        ys, xs = np.nonzero(mask == number)
        if len(xs) < min_pixels:
            continue
        uid, o = by_number[int(number)]
        out.append({"uid": uid, "object_id": o["object_id"] or f"foreign_{o['foreign_type']}", "kind": o["kind"],
                    "state": o["state"], "variant": o["variant"], "build_state": o["build_state"],
                    "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "visible_px": len(xs)})
    return out


def depth_mm_u16(depth_m: np.ndarray) -> np.ndarray:
    """Metres (inf / nan where nothing was hit) -> uint16 mm, 0 = no return (same convention as the cell's depth)."""
    mm = np.nan_to_num(depth_m * 1000.0, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.rint(mm), 0, DEPTH_MAX_MM).astype(np.uint16)


def resize_nearest(img: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize (never blends depth or label values)."""
    h, w = img.shape[:2]
    ys = (np.arange(size_wh[1]) * h / size_wh[1]).astype(int)
    xs = (np.arange(size_wh[0]) * w / size_wh[0]).astype(int)
    return img[ys][:, xs]


def apply_post(rgb: np.ndarray, post: dict, seed: int) -> np.ndarray:
    """Sensor-style degradation from the spec draw: optional horizontal motion blur, then Gaussian noise."""
    rng = np.random.default_rng(seed)
    img = rgb.astype(np.float32) / 255.0
    if post["motion_blur"]:
        k = int(rng.integers(3, 8))
        img = sum(np.roll(img, s, axis=1) for s in range(-(k // 2), k - k // 2)) / k
    if post["gaussian_noise_sigma"] > 0:
        img = img + rng.normal(0.0, post["gaussian_noise_sigma"], img.shape)
    return np.clip(np.rint(img * 255.0), 0, 255).astype(np.uint8)
