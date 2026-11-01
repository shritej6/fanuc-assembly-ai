"""After a sampler fix: mark rendered scenes whose spec changed so the generator renders them again.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdg.scene_sampler import Inputs, sample_scene

RENDER_FIELDS = ("render_dz_mm",)   # written into the stored spec by the renderer (sdg/isaac/stage.py), not the sampler


def canonical(spec: dict) -> str:
    spec = json.loads(json.dumps(spec))
    for obj in spec["objects"]:
        for field in RENDER_FIELDS:
            obj.pop(field, None)
    return json.dumps(spec, sort_keys=True)


def supersede(chunk: Path, inputs: Inputs) -> int:
    set_name = chunk.parent.name
    old = chunk / "superseded"
    changed = []
    for label in sorted((chunk / "labels").glob("*.json")):
        spec = json.loads(label.read_text())["spec"]
        if canonical(spec) != canonical(sample_scene(inputs, set_name, spec["index"])):
            changed.append(label)
    if changed:
        old.mkdir(exist_ok=True)
        for f in [*changed, chunk / "_DONE.json", chunk / "_QA.json"]:
            if f.exists():
                f.rename(old / f.name)
    return len(changed)


def restore(chunk: Path, inputs: Inputs) -> int:
    old, labels = chunk / "superseded", chunk / "labels"
    back = 0
    for label in sorted(old.glob("[0-9]*.json")):
        spec = json.loads(label.read_text())["spec"]
        if not (labels / label.name).exists() and \
                canonical(spec) == canonical(sample_scene(inputs, chunk.parent.name, spec["index"])):
            label.rename(labels / label.name)
            back += 1
    specs = chunk.parents[2] / "specs" / chunk.parent.name / f"{chunk.name}.jsonl"
    expected = sum(1 for line in specs.read_text().splitlines() if line.strip())
    if len(list(labels.glob("*.json"))) == expected:
        for name in ("_DONE.json", "_QA.json"):
            if (old / name).exists() and not (chunk / name).exists():
                (old / name).rename(chunk / name)
    return back


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--restore", action="store_true", help="move back labels that match the current spec")
    ap.add_argument("chunks", nargs="+", type=Path)
    args = ap.parse_args()
    inputs = Inputs.load()
    for chunk in args.chunks:
        if args.restore:
            print(f"{chunk.parent.name}/{chunk.name}: {restore(chunk, inputs)} labels restored")
        else:
            print(f"{chunk.parent.name}/{chunk.name}: {supersede(chunk, inputs)} scenes to re-render")


if __name__ == "__main__":
    main()
