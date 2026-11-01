"""Export per-part PatchCore crops: train on good O wrist close-ups, test on C and I.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from perception.manifest_types import split_for_scene

CONTEXT = 1.3
SPLIT_SEED = 1337


def wrist_target(label: dict) -> tuple[dict, dict] | None:
    """Return the wrist target's scene object and wrist box, or None when it is not visible."""
    spec = label["spec"]
    uid = spec.get("wrist_target_uid")
    obj = next((o for o in spec["objects"] if o["uid"] == uid), None)
    box = next((b for b in label["cameras"].get("wrist", []) if b["uid"] == uid), None)
    if obj is None or box is None or not box.get("visible_px"):
        return None
    return obj, box


def square_crop(image: Image.Image, bbox: list[int]) -> Image.Image:
    """Return a square crop around the target, clipped to the image."""
    x0, y0, x1, y1 = bbox
    side = max(x1 - x0, y1 - y0) * CONTEXT
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left, top = max(0, round(cx - side / 2)), max(0, round(cy - side / 2))
    return image.crop((left, top, min(image.width, round(cx + side / 2)), min(image.height, round(cy + side / 2))))


def bucket(set_name: str, index: int, obj: dict, rest: dict | None) -> str | None:
    """Which folder a wrist target belongs in, or None when it is not used."""
    if obj["kind"] != "part" or obj["object_id"].startswith("F"):
        return None
    if set_name == "O":
        if obj.get("variant") != "ok":
            return None
        split = split_for_scene("O", index, SPLIT_SEED)
        return {"train": "train_good", "test": "o_test_good"}.get(split)
    if rest is None or rest.get(obj["uid"]) != "installed":
        return None                    # old sets: only parts lying as on the real bed are a fair test
    return "test_good" if obj.get("variant") == "ok" else "test_bad"


def export(new_root: Path, old_root: Path, out: Path) -> Counter:
    """Export crops from the O, C and I sets and return the crop counts."""
    counts: Counter = Counter()
    sources = [("O", new_root / "O")] + [(s, old_root / s) for s in ("C", "I")]
    for set_name, set_dir in sources:
        for chunk in sorted(set_dir.glob("chunk_*")):
            sidecar = chunk / "_REST.json"
            rests = json.loads(sidecar.read_text())["images"] if sidecar.exists() else {}
            for label_path in sorted((chunk / "labels").glob("*.json")):
                found = wrist_target(json.loads(label_path.read_text()))
                if found is None:
                    continue
                obj, box = found
                rest = rests.get(label_path.stem) if set_name != "O" else None
                folder = bucket(set_name, int(label_path.stem), obj, rest)
                if folder is None:
                    continue
                with Image.open(chunk / "rgb" / "wrist" / f"{label_path.stem}.jpg") as image:
                    crop = square_crop(image.convert("RGB"), box["bbox_xyxy"])
                target = out / obj["object_id"] / folder / f"{set_name}_{chunk.name}_{label_path.stem}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                crop.save(target)
                counts[(obj["object_id"], folder)] += 1
    return counts


def main() -> None:
    """Parse arguments, export the crops and print the counts."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new-root", type=Path, required=True, help="synthetic/ holding the test-bed set O")
    ap.add_argument("--old-root", type=Path, required=True, help="synthetic/ holding C and I with _REST.json")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    counts = export(args.new_root, args.old_root, args.out)
    table = {}
    for (part, folder), n in sorted(counts.items()):
        table.setdefault(part, {})[folder] = n
    (args.out / "counts.json").write_text(json.dumps(table, indent=2, sort_keys=True) + "\n")
    print(json.dumps(table, sort_keys=True))


if __name__ == "__main__":
    main()
