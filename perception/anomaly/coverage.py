"""Export PatchCore crops for all 24 movable parts with in-domain defects to measure against.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from perception.anomaly.testbed import square_crop, wrist_target
from perception.manifest_types import split_for_scene

SPLIT_SEED = 1337
EVAL_SETS = ("U", "T")
MASK_FILL = 128


def masked(image: Image.Image, seg_path: Path, uid: str) -> Image.Image:
    """Return the image with everything outside instance `uid` set to mid-grey."""
    with Image.open(seg_path) as seg:
        keep = np.asarray(seg) == int(uid[1:]) + 1
    if not keep.any():
        raise ValueError(f"{seg_path}: no pixels for {uid}")
    pixels = np.asarray(image).copy()
    pixels[~keep] = MASK_FILL
    return Image.fromarray(pixels)


def eval_split(set_name: str, index: int) -> str:
    """Return the split for one evaluation scene: half set the threshold, half are reported."""
    token = f"{SPLIT_SEED}:eval:{set_name}:{index}".encode()
    return "val" if hashlib.sha256(token).digest()[0] % 2 == 0 else "test"


def bucket(set_name: str, index: int, obj: dict) -> tuple[str, str] | None:
    """Return (split, folder) for one wrist target, or None when it is not used."""
    if obj["kind"] != "part" or not obj["object_id"].startswith("M") or obj.get("rest") != "installed":
        return None
    good = obj.get("variant") == "ok"
    if set_name == "O":
        if not good:
            return None
        split = split_for_scene("O", index, SPLIT_SEED)
        return ("train", "train_good") if split == "train" else ({"validation": "val"}.get(split, "test"), "good")
    if set_name in EVAL_SETS:
        return eval_split(set_name, index), "good" if good else "bad"
    return None


def crop_path(out: Path, part: str, split: str, folder: str, name: str) -> Path:
    """Return the output path of one crop."""
    return out / part / ("train_good" if split == "train" else f"{split}/{folder}") / name


def export_chunk(set_name: str, chunk: Path, out: Path, counts: Counter, index_rows: dict, mask: bool) -> None:
    """Write the wrist-target crops of one chunk and record them in the index."""
    for label_path in sorted((chunk / "labels").glob("*.json")):
        label = json.loads(label_path.read_text())
        found = wrist_target(label)
        if found is None:
            continue
        obj, box = found
        where = bucket(set_name, int(label["spec"]["index"]), obj)
        if where is None:
            continue
        split, folder = where
        part, name = obj["object_id"], f"{set_name}_{chunk.name}_{label_path.stem}.png"
        with Image.open(chunk / "rgb" / "wrist" / f"{label_path.stem}.jpg") as image:
            rgb = image.convert("RGB")
        if mask:
            rgb = masked(rgb, chunk / "seg" / "wrist" / f"{label_path.stem}.png", obj["uid"])
        crop = square_crop(rgb, box["bbox_xyxy"])
        target = crop_path(out, part, split, folder, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        crop.save(target)
        counts[(part, folder if split == "train" else f"{split}_{folder}")] += 1
        if split != "train":
            index_rows.setdefault(part, []).append(
                {"file": f"{split}/{folder}/{name}", "split": split, "set": set_name, "variant": obj["variant"]})


def export(root: Path, out: Path, sets: tuple[str, ...], mask: bool = False) -> Counter:
    """Export crops for every chunk of the given sets and return the crop counts."""
    counts: Counter = Counter()
    index_rows: dict[str, list[dict]] = {}
    for set_name in sets:
        chunks = sorted((root / set_name).glob("chunk_*"))
        if not chunks:
            raise FileNotFoundError(f"{root / set_name}: no chunk_* directories")
        for chunk in chunks:
            if not (chunk / "_DONE.json").is_file():
                raise FileNotFoundError(f"{chunk}: chunk is not complete (_DONE.json missing)")
            export_chunk(set_name, chunk, out, counts, index_rows, mask)
    for part, rows in index_rows.items():
        (out / part / "index.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return counts


def main() -> None:
    """Parse arguments, export the crops and print the counts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True, help="synthetic/ holding the test-bed sets O, U and T")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sets", default="O,U,T")
    ap.add_argument("--mask", action="store_true", help="grey out everything but the target's instance mask")
    args = ap.parse_args()
    counts = export(args.root, args.out, tuple(args.sets.split(",")), args.mask)
    table: dict[str, dict[str, int]] = {}
    for (part, folder), n in sorted(counts.items()):
        table.setdefault(part, {})[folder] = n
    (args.out / "counts.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(json.dumps(table, sort_keys=True))


if __name__ == "__main__":
    main()
