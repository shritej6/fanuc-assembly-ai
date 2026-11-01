"""Write and check completion markers for rendered chunks.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

DONE = "_DONE.json"


def write_json_atomic(path: Path, obj: object) -> None:
    """A reader sees either the old file or the whole new one, never a truncated write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def specs_digest(specs: Path) -> str:
    return hashlib.sha256(specs.read_bytes()).hexdigest()


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def scene_files(spec: dict, depth_cams: set[str]) -> list[str]:
    idx = f"{spec['index']:06d}"
    files = []
    for cam in spec["cameras"]:
        files += [f"rgb/{cam}/{idx}.jpg", f"seg/{cam}/{idx}.png"]
        if cam in depth_cams and "depth" in spec["outputs"]:
            files.append(f"depth/{cam}/{idx}.png")
    return files


def scene_complete(out: Path, spec: dict, depth_cams: set[str]) -> bool:
    label = _read_json(out / "labels" / f"{spec['index']:06d}.json")
    if not isinstance(label, dict) or label.get("spec") != spec or set(label.get("cameras", {})) != set(spec["cameras"]):
        return False
    return all((out / f).is_file() and (out / f).stat().st_size > 0 for f in scene_files(spec, depth_cams))


def chunk_done(out: Path, specs: Path) -> dict | None:
    """The chunk's _DONE.json when it is valid and was rendered from exactly these specs, else None."""
    done = _read_json(out / DONE)
    if isinstance(done, dict) and done.get("specs_sha256") == specs_digest(specs):
        return done
    return None


def check(out: Path, specs: Path) -> list[str]:
    """Problems that keep the chunk from counting as done (empty when it is complete)."""
    done = chunk_done(out, specs)
    if done is None:
        return [f"{out / DONE} missing, unreadable or made from other specs"]
    lines = [line for line in specs.read_text().splitlines() if line.strip()]
    if done.get("scenes") != len(lines):
        return [f"{DONE} counts {done.get('scenes')} scenes, the specs have {len(lines)}"]
    depth_cams = set(done.get("depth_cams", []))
    bad = [json.loads(line)["index"] for line in lines if not scene_complete(out, json.loads(line), depth_cams)]
    return [f"{len(bad)} scenes incomplete, first {bad[:5]}"] if bad else []


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "check":
        sys.exit("usage: python3 -m sdg.chunk_marks check <chunk dir> <specs.jsonl>")
    problems = check(Path(sys.argv[2]), Path(sys.argv[3]))
    for problem in problems:
        print(f"chunk {sys.argv[2]}: {problem}", file=sys.stderr)
    sys.exit(1 if problems else 0)
