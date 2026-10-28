"""After a sampler fix: mark rendered scenes whose spec changed so the generator renders them again.

    python -m sdg.tools.supersede_changed ~/fanuc-assembly-data/synthetic/D/chunk_*

Each label stores the spec it was rendered from. A scene whose current spec differs has its label moved to
<chunk>/superseded/ (the generator then re-renders it and overwrites its images); the chunk's _DONE.json and
_QA.json move there too, so the chunk is rendered and QA'd again. Unchanged scenes are left alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sdg.scene_sampler import Inputs, sample_scene


def canonical(spec: dict) -> str:
    return json.dumps(json.loads(json.dumps(spec)), sort_keys=True)


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("chunks", nargs="+", type=Path)
    args = ap.parse_args()
    inputs = Inputs.load()
    for chunk in args.chunks:
        print(f"{chunk.parent.name}/{chunk.name}: {supersede(chunk, inputs)} scenes to re-render")


if __name__ == "__main__":
    main()
