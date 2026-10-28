"""Quality gates for rendered chunks (docs/SDG_PLAN.md section 7). Runs on any host with the chunk folders.

    uv run python -m sdg.qa $DATA_ROOT/synthetic/A/chunk_0000 [more chunks...] --sheet qa_A0.png
    uv run python -m sdg.qa <chunk> --sample 40          # random 40 scenes per chunk (the watchdog's check)

Checks every scene: all files present, boxes inside the image and equal to the mask extent, depth sane.
Exit code 1 if any check fails. The overlay sheet (boxes + ids on 12 images) is for the by-eye gate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

DEPTH_RANGE_MM = {"table": (400, 2000), "wrist": (60, 900)}   # plausible median depth per camera
COLOURS = {"base": (40, 140, 255), "part": (40, 220, 90), "trap": (255, 60, 60), "spare": (250, 200, 0),
           "reject": (220, 60, 220), "foreign": (255, 140, 0)}


def check_scene(chunk: Path, label_file: Path) -> list[str]:
    labels = json.loads(label_file.read_text())
    idx, errors = label_file.stem, []
    for cam, boxes in labels["cameras"].items():
        rgb_path, seg_path = chunk / "rgb" / cam / f"{idx}.jpg", chunk / "seg" / cam / f"{idx}.png"
        if not rgb_path.exists() or not seg_path.exists():
            errors.append(f"{idx}/{cam}: missing rgb or seg")
            continue
        seg = np.array(Image.open(seg_path))
        h, w = seg.shape
        if Image.open(rgb_path).size != (w, h):
            errors.append(f"{idx}/{cam}: rgb and seg sizes differ")
        if not boxes:
            errors.append(f"{idx}/{cam}: no labelled objects in view")
        for b in boxes:
            errors += _check_box(idx, cam, b, seg)
        errors += _check_depth(chunk, idx, cam, labels["spec"])
    return errors


def _check_box(idx: str, cam: str, b: dict, seg: np.ndarray) -> list[str]:
    x0, y0, x1, y1 = b["bbox_xyxy"]
    h, w = seg.shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        return [f"{idx}/{cam}/{b['uid']}: box {b['bbox_xyxy']} outside {w}x{h}"]
    ys, xs = np.nonzero(seg == int(b["uid"][1:]) + 1)
    if len(xs) == 0 or [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] != [x0, y0, x1, y1]:
        return [f"{idx}/{cam}/{b['uid']}: box does not match its mask"]
    return []


def _check_depth(chunk: Path, idx: str, cam: str, spec: dict) -> list[str]:
    path = chunk / "depth" / cam / f"{idx}.png"
    if cam not in DEPTH_RANGE_MM or "depth" not in spec["outputs"]:
        return []
    if not path.exists():
        return [f"{idx}/{cam}: depth missing"]
    d = np.array(Image.open(path))
    valid = d[d > 0]
    lo, hi = DEPTH_RANGE_MM[cam]
    if valid.size < 0.5 * d.size or not lo <= float(np.median(valid)) <= hi:
        median = float(np.median(valid)) if valid.size else 0.0
        return [f"{idx}/{cam}: depth implausible (valid {valid.size / d.size:.0%}, median {median:.0f} mm)"]
    return []


def overlay_sheet(chunks: list[Path], out: Path, n: int = 12, width: int = 640) -> None:
    files = [(c, f) for c in chunks for f in sorted((c / "labels").glob("*.json"))]
    picks = [files[i] for i in np.linspace(0, len(files) - 1, min(n, len(files))).astype(int)]
    tiles = []
    for chunk, f in picks:
        labels = json.loads(f.read_text())
        cam, boxes = next(iter(labels["cameras"].items()))
        img = Image.open(chunk / "rgb" / cam / f"{f.stem}.jpg").convert("RGB")
        draw = ImageDraw.Draw(img)
        for b in boxes:
            colour = COLOURS.get(b["kind"], (255, 255, 255))
            draw.rectangle(b["bbox_xyxy"], outline=colour, width=3)
            tag = b["object_id"] + ("" if b["variant"] == "ok" else f" [{b['variant']}]")
            draw.text((b["bbox_xyxy"][0] + 3, b["bbox_xyxy"][1] + 2), tag, fill=colour)
        draw.text((8, 8), f"{chunk.parent.name}/{chunk.name}/{f.stem} {cam}", fill=(255, 0, 0))
        tiles.append(img.resize((width, int(width * img.height / img.width))))
    cols = 3
    th = max(t.height for t in tiles)
    sheet = Image.new("RGB", (cols * width, th * -(-len(tiles) // cols)), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * width, (i // cols) * th))
    sheet.save(out)


def check_chunks(chunks: list[Path], sample: int = 0, seed: int = 0) -> dict:
    """Check every scene, or a random `sample` per chunk (fixed seed, so a re-check looks at the same scenes)."""
    scenes, errors = 0, []
    for chunk in chunks:
        files = sorted((chunk / "labels").glob("*.json"))
        if sample and len(files) > sample:
            files = sorted(random.Random(seed).sample(files, sample))
        for f in files:
            scenes += 1
            errors += check_scene(chunk, f)
    return {"scenes": scenes, "errors": len(errors), "first_errors": errors[:20]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chunks", nargs="+", type=Path)
    ap.add_argument("--sheet", type=Path)
    ap.add_argument("--sample", type=int, default=0, help="check N random scenes per chunk instead of all")
    args = ap.parse_args()
    result = check_chunks(args.chunks, args.sample)
    print(json.dumps(result, indent=1))
    if args.sheet:
        overlay_sheet(args.chunks, args.sheet)
        print(f"overlay sheet: {args.sheet}")
    sys.exit(1 if result["errors"] or result["scenes"] == 0 else 0)


if __name__ == "__main__":
    main()
