"""Stage agent 3's closed seed demos for Mimic generation (stdlib only; runs on the gather host, fa-mimic).

    python3 -m sim.assembly_lab.mimic.stage_seeds --demos ~/t12_data/demos --out ~/t12_data/a7/seed_stage

Copies every file listed in <demos>/run10_closed.txt (closed and annotated by agent 3; files not listed may still be
written) with its .meta.jsonl into <out>/<skill>/, and writes <file>.drop.json with the meta lines that
<demos>/run10_drop_tags.jsonl tags as bad sources. mimicgen.load_sources honours the drop list and also skips sources
whose carry in-hand drift is over 0.5 mm. Prints kept episodes per skill and product.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def closed_files(demos: Path, listing: str) -> list[Path]:
    out = []
    for line in (demos / listing).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line).expanduser()
        if not p.is_absolute():
            p = demos.parent / p if (demos.parent / p).exists() else demos / p
        if p.suffix == ".hdf5" and p.exists():
            out.append(p)
    return out


def drop_tags(demos: Path, name: str) -> dict[str, list[int]]:
    tags: dict[str, list[int]] = {}
    path = demos / name
    if not path.exists():
        return tags
    for line in path.read_text().splitlines():
        if line.strip():
            t = json.loads(line)
            tags.setdefault(Path(t["file"]).name, []).append(int(t["meta_line"]))
    return tags


def stage(demos: Path, out: Path, listing: str, tags_name: str) -> dict:
    tags = drop_tags(demos, tags_name)
    counts: Counter = Counter()
    for f in closed_files(demos, listing):
        meta = Path(f"{f}.meta.jsonl")
        if not meta.exists():
            continue
        skill = f.parent.name
        dst = out / skill / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.stat().st_size != f.stat().st_size:
            shutil.copy2(f, dst)
        shutil.copy2(meta, Path(f"{dst}.meta.jsonl"))
        Path(f"{dst}.drop.json").write_text(json.dumps(sorted(tags.get(f.name, []))))
        for i, line in enumerate(meta.read_text().splitlines()):
            m = json.loads(line)
            if m.get("kept") and i not in tags.get(f.name, []):
                counts[f"{skill}:{m.get('product')}:{m.get('case')}"] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", type=Path, default=Path("~/t12_data/demos"))
    ap.add_argument("--out", type=Path, default=Path("~/t12_data/a7/seed_stage"))
    ap.add_argument("--closed", default="run10_closed.txt")
    ap.add_argument("--drop-tags", default="run10_drop_tags.jsonl")
    a = ap.parse_args()
    print(json.dumps(stage(a.demos.expanduser(), a.out.expanduser(), a.closed, a.drop_tags), indent=1))


if __name__ == "__main__":
    main()
