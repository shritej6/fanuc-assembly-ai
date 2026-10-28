"""Unattended watchdog for a render host: QA every finished chunk on a sample, retry failed chunks, catch hangs.

    python -m sdg.watchdog --host spark --plan C:0-15 D:0-11 A:0-31 --loop 600
    python -m sdg.watchdog --host rosie --loop 600

Each pass appends one status line to $DATA_ROOT/watchdog/log.txt and rewrites watchdog/status.json.
Per finished chunk: `_QA.json` (random 40 scenes checked + scene count) and watchdog/sheets/<set>_<chunk>.png.
A failed chunk (`_FAILED.txt`) is moved to `_FAILED_retry<n>.txt` and re-rendered, at most MAX_RETRIES times; the
generator skips scenes that already have labels, so a retry only renders what is missing.
spark (one GPU, serial): the watchdog also starts the next unfinished chunks of `--plan` whenever nothing renders,
and kills a render whose chunk wrote no label for HANG_MIN minutes (recorded as a failure, so it is retried).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

from sdg.qa import check_chunks, overlay_sheet
from sdg.scene_sampler import Inputs, n_chunks

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path.home() / "fanuc-assembly-data"))
SAMPLE, MAX_RETRIES, HANG_MIN = 40, 3, 30


def chunk_dirs() -> list[Path]:
    return sorted(DATA_ROOT.glob("synthetic/*/chunk_*"))


def qa_finished(chunk: Path) -> dict | None:
    """Sample-check a chunk once, when it is done. Returns the new result (None if nothing to do)."""
    if not (chunk / "_DONE.json").exists() or (chunk / "_QA.json").exists():
        return None
    result = check_chunks([chunk], sample=SAMPLE)
    specs = DATA_ROOT / "specs" / chunk.parent.name / f"{chunk.name}.jsonl"
    expected = sum(1 for line in specs.read_text().splitlines() if line.strip()) if specs.exists() else None
    result["labelled_scenes"] = len(list((chunk / "labels").glob("*.json")))
    result["expected_scenes"] = expected
    result["ok"] = result["errors"] == 0 and result["labelled_scenes"] == expected
    (chunk / "_QA.json").write_text(json.dumps(result, indent=1))
    sheets = DATA_ROOT / "watchdog" / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    overlay_sheet([chunk], sheets / f"{chunk.parent.name}_{chunk.name}.png")
    return result


def retries(chunk: Path) -> int:
    return len(list(chunk.glob("_FAILED_retry*.txt")))


def take_failure(chunk: Path) -> bool:
    """Move `_FAILED.txt` aside so the chunk can run again. False once it has failed MAX_RETRIES times."""
    n = retries(chunk)
    if n >= MAX_RETRIES:
        return False
    (chunk / "_FAILED.txt").rename(chunk / f"_FAILED_retry{n + 1}.txt")
    return True


def parse_plan(plan: list[str]) -> list[tuple[str, int]]:
    out = []
    for item in plan:
        name, rng = item.split(":")
        lo, hi = (int(v) for v in rng.split("-"))
        out += [(name, c) for c in range(lo, hi + 1)]
    return out


def chunk_path(name: str, c: int) -> Path:
    return DATA_ROOT / "synthetic" / name / f"chunk_{c:04d}"


def run(cmd: list[str] | str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, **kw)


def rosie_retry(chunk: Path, events: list[str]) -> None:
    if not take_failure(chunk):
        events.append(f"GAVE UP {chunk.parent.name}/{chunk.name} after {MAX_RETRIES} retries")
        return
    c = int(chunk.name.split("_")[1])
    out = run([str(REPO / "sdg/rosie/submit.sh"), chunk.parent.name, f"{c}-{c}"], env={**os.environ})
    events.append(f"retry {chunk.parent.name}/{chunk.name}: {out.stdout.strip().splitlines()[-1:] or out.stderr[-200:]}")


def spark_rendering() -> bool:
    return run(["pgrep", "-f", "sdg/spark/render_chunk.sh"]).returncode == 0


def spark_hang_check(events: list[str]) -> None:
    """Kill a render whose newest label is older than HANG_MIN minutes; the chunk is then retried."""
    active = [c for c in chunk_dirs() if not (c / "_DONE.json").exists() and not (c / "_FAILED.txt").exists()]
    stamps = [(max((f.stat().st_mtime for f in (c / "labels").glob("*.json")), default=c.stat().st_mtime), c)
              for c in active]
    if not stamps:
        return
    newest, chunk = max(stamps)
    if time.time() - newest > HANG_MIN * 60:
        image = os.environ.get("ISAAC_IMAGE", "nvcr.io/nvidia/isaac-sim:6.0.0")
        ids = run(["docker", "ps", "-q", "--filter", f"ancestor={image}"]).stdout.split()
        if ids:
            run(["docker", "kill", *ids])
            (chunk / "_FAILED.txt").write_text(f"watchdog: no new label for {HANG_MIN} min, render killed\n")
            events.append(f"HANG {chunk.parent.name}/{chunk.name}: killed render, will retry")


def spark_schedule(plan: list[tuple[str, int]], events: list[str]) -> None:
    """Nothing renders: start every unfinished chunk of the first set in the plan that still has work."""
    todo: dict[str, list[int]] = {}
    for name, c in plan:
        path = chunk_path(name, c)
        if (path / "_DONE.json").exists():
            continue
        if (path / "_FAILED.txt").exists() and not take_failure(path):
            events.append(f"GAVE UP {name}/{path.name} after {MAX_RETRIES} retries")
            continue
        todo.setdefault(name, []).append(c)
    if not todo:
        return
    name, chunks = next(iter(todo.items()))
    log = DATA_ROOT / "logs" / "production_spark.out"
    cmd = f"setsid nohup {REPO}/sdg/spark/render_chunk.sh {name} {' '.join(map(str, chunks))} >> {log} 2>&1 < /dev/null &"
    subprocess.Popen(["bash", "-c", cmd], cwd=REPO)
    events.append(f"started {name} chunks {chunks[0]}..{chunks[-1]} ({len(chunks)})")


def summary() -> dict:
    sets: dict[str, dict] = {}
    for c in chunk_dirs():
        s = sets.setdefault(c.parent.name, {"chunks_done": 0, "scenes": 0, "failed": [], "qa_bad": []})
        s["scenes"] += len(list((c / "labels").glob("*.json")))
        s["chunks_done"] += (c / "_DONE.json").exists()
        if (c / "_FAILED.txt").exists():
            s["failed"].append(c.name)
        qa = c / "_QA.json"
        if qa.exists() and not json.loads(qa.read_text())["ok"]:
            s["qa_bad"].append(c.name)
        s["qa_passed"] = s.get("qa_passed", 0) + (qa.exists() and json.loads(qa.read_text())["ok"])
    inputs = Inputs.load()
    for name, s in sets.items():
        s["chunks_total"] = n_chunks(inputs, name) if name in inputs.sets["sets"] else None
        s["ready_for_training"] = s["qa_passed"] == s["chunks_total"]   # every chunk done and sample-checked
    return sets


def one_pass(host: str, plan: list[tuple[str, int]]) -> None:
    events = []
    for chunk in chunk_dirs():
        result = qa_finished(chunk)
        if result is not None and not result["ok"]:
            events.append(f"QA FAIL {chunk.parent.name}/{chunk.name}: {result['errors']} errors, "
                          f"{result['labelled_scenes']}/{result['expected_scenes']} scenes {result['first_errors'][:2]}")
        if host == "rosie" and (chunk / "_FAILED.txt").exists():
            rosie_retry(chunk, events)
    if host == "spark":
        if spark_rendering():
            spark_hang_check(events)
        else:
            spark_schedule(plan, events)
    status = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "host": host, "sets": summary(), "events": events}
    out = DATA_ROOT / "watchdog"
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    with (out / "log.txt").open("a") as f:
        f.write(json.dumps(status) + "\n")
    print(json.dumps(status), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", required=True, choices=["rosie", "spark"])
    ap.add_argument("--plan", nargs="*", default=[], help="spark: SET:first-last in render order")
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes (0 = one pass)")
    args = ap.parse_args()
    plan = parse_plan(args.plan)
    while True:
        if not args.loop:
            one_pass(args.host, plan)
            return
        try:
            one_pass(args.host, plan)
        except Exception:  # noqa: BLE001 - unattended loop: log the failure with its trace and keep watching
            with (DATA_ROOT / "watchdog" / "log.txt").open("a") as f:
                f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "watchdog_error":
                                    traceback.format_exc()}) + "\n")
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
