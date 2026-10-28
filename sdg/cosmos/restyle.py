"""Cosmos Transfer restyling of rendered images: pick scenes, export a bundle, import the checked results.

    python -m sdg.cosmos.restyle export --sets A B C --fraction 0.15 --out bundle/        # on Rosie
    python -m sdg.cosmos.restyle export --sets A B C --count 4 --out pilot/                # a fixed number per set
    (bundle/ is restyled on an H100 by sdg/cosmos/run_transfer.sh, which writes bundle/out/<name>.jpg)
    python -m sdg.cosmos.restyle import bundle/ --min-fscore <measured> --min-object <measured>   # on Rosie

Cosmos Transfer 2.5 image-to-image with edge + vis control keeps every part edge where the render put it, so a
restyled image keeps the render's labels. Accepted images go to <chunk>/rgb_cosmos/<cam>/<idx>.jpg at the render's
size, the check result of every image to <chunk>/_COSMOS.json. Training reads rgb_cosmos as extra images with the same labels.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from sdg.cosmos.drift import edge_fscore, worst_object

DATA_ROOT = Path.home() / "fanuc-assembly-data"
# Edge control pins every shape; vis control (the blurred input) keeps every colour: the yellow robot and tape, the
# black parts, the coloured foreign objects. The prompt only adds real-camera look, so Cosmos changes nothing else.
# (Edge control alone took colours from the prompt and painted the yellow robot black: pilot 2026-10-28.)
SCENE = ("A real photograph of a FANUC robot cell. Every object keeps exactly the shape, position and colour it has in "
         "the input: matte 3D-printed plastic parts with fine visible layer lines, a white table, real materials.")
LIGHTS = ["Cool fluorescent ceiling light, soft shadows.", "Warm LED work light from one side, firm shadows.",
          "Mixed daylight from a window and ceiling light.", "Dim overhead light, slight sensor noise and grain."]
CONTROLS = {"edge": {"control_weight": 1.0}, "vis": {"control_weight": 0.5}}
CAMERAS = {"table": "Shot from an overhead industrial camera.", "side": "Shot from a fixed side camera.",
           "wrist": "Close-up from a camera on the robot wrist, shallow depth of field."}


def name_of(set_name: str, chunk: str, cam: str, idx: str) -> str:
    return f"{set_name}__{chunk}__{cam}__{idx}"


def parse_name(name: str) -> tuple[str, str, str, str]:
    set_name, chunk, cam, idx = name.split("__")
    return set_name, chunk, cam, idx


def pick(data_root: Path, set_name: str, amount: float, seed: int) -> list[tuple[str, str, str, str]]:
    """A fixed random share of the set's scenes (amount < 1: fraction, else count), one camera per scene in turn."""
    picks = []
    for chunk in sorted((data_root / "synthetic" / set_name).glob("chunk_*")):
        for label in sorted((chunk / "labels").glob("*.json")):
            cams = sorted(json.loads(label.read_text())["cameras"])
            picks.append((set_name, chunk.name, cams, label.stem))
    rng = random.Random(f"{set_name}:{seed}")
    n = round(len(picks) * amount) if amount < 1 else min(int(amount), len(picks))
    chosen = sorted(rng.sample(picks, n), key=lambda p: (p[1], p[3]))
    return [(s, c, cams[i % len(cams)], idx) for i, (s, c, cams, idx) in enumerate(chosen)]


def spec_for(name: str, cam: str, seed: int) -> dict:
    light = LIGHTS[random.Random(f"{name}:{seed}").randrange(len(LIGHTS))]
    return {"name": name, "prompt": f"{SCENE} {CAMERAS[cam]} {light}", "video_path": f"../inputs/{name}.jpg",
            "max_frames": 1, "num_video_frames_per_chunk": 1, "seed": seed, **CONTROLS}


def export(data_root: Path, sets: list[str], amount: float, seed: int, out: Path) -> int:
    (out / "inputs").mkdir(parents=True, exist_ok=True)
    (out / "specs").mkdir(exist_ok=True)
    n = 0
    for set_name in sets:
        for s, chunk, cam, idx in pick(data_root, set_name, amount, seed):
            name = name_of(s, chunk, cam, idx)
            shutil.copyfile(data_root / "synthetic" / s / chunk / "rgb" / cam / f"{idx}.jpg", out / "inputs" / f"{name}.jpg")
            (out / "specs" / f"{name}.json").write_text(json.dumps(spec_for(name, cam, seed), indent=1))
            n += 1
    return n


def check_one(data_root: Path, restyled_path: Path) -> tuple[Path, str, str, dict, Image.Image]:
    """Restyled image resized to the render's size, and its edge scores against the render (whole image, worst object)."""
    s, chunk, cam, idx = parse_name(restyled_path.stem)
    chunk_dir = data_root / "synthetic" / s / chunk
    source = Image.open(chunk_dir / "rgb" / cam / f"{idx}.jpg").convert("RGB")
    restyled = Image.open(restyled_path).convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    seg = np.array(Image.open(chunk_dir / "seg" / cam / f"{idx}.png"))
    src, out = np.asarray(source), np.asarray(restyled)
    worst, uid = worst_object(src, out, seg)
    scores = {"edge_fscore": round(edge_fscore(src, out, seg), 4), "worst_object_fscore": round(worst, 4),
              "worst_object_seg_id": uid}
    return chunk_dir, cam, idx, scores, restyled


def import_results(data_root: Path, bundle: Path, min_fscore: float, min_object: float) -> dict:
    """Keep restyled images whose edges match the render, as a whole and for every object; record every result."""
    records: dict[Path, dict] = {}
    for path in sorted((bundle / "out").glob("*.jpg")):
        chunk_dir, cam, idx, scores, restyled = check_one(data_root, path)
        ok = scores["edge_fscore"] >= min_fscore and scores["worst_object_fscore"] >= min_object
        if ok:
            target = chunk_dir / "rgb_cosmos" / cam / f"{idx}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            restyled.save(target, quality=95)
        records.setdefault(chunk_dir, {})[f"{cam}/{idx}"] = {**scores, "accepted": ok}
    for chunk_dir, rec in records.items():
        path = chunk_dir / "_COSMOS.json"
        old = json.loads(path.read_text()) if path.exists() else {}
        path.write_text(json.dumps({**old, **rec}, indent=1, sort_keys=True))
    scores = [r["accepted"] for rec in records.values() for r in rec.values()]
    return {"checked": len(scores), "accepted": sum(scores), "min_fscore": min_fscore, "min_object": min_object}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="copy picked renders and write Cosmos specs into a bundle")
    ex.add_argument("--sets", nargs="+", required=True)
    share = ex.add_mutually_exclusive_group(required=True)
    share.add_argument("--fraction", type=float, help="share of each set's scenes (0-1)")
    share.add_argument("--count", type=int, help="scenes per set")
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--out", type=Path, required=True)
    im = sub.add_parser("import", help="shape-drift check the restyled images and keep the good ones")
    im.add_argument("bundle", type=Path)
    im.add_argument("--min-fscore", type=float, required=True, help="whole-image edge F-score")
    im.add_argument("--min-object", type=float, required=True, help="edge F-score of the worst object")
    for p in (ex, im):
        p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    args = ap.parse_args()
    if args.cmd == "export":
        amount = args.fraction if args.fraction is not None else args.count
        if not (0 < args.fraction < 1 if args.fraction is not None else args.count >= 1):
            ap.error("--fraction must be in (0, 1), --count at least 1")
        print(f"{export(args.data_root, args.sets, amount, args.seed, args.out)} images exported")
    else:
        print(json.dumps(import_results(args.data_root, args.bundle, args.min_fscore, args.min_object)))


if __name__ == "__main__":
    main()
