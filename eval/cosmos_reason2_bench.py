"""Benchmark the live Cosmos Reason 2 8B build-check endpoint on rendered set D scenes (sim only).

    uv run --with pillow python -m eval.cosmos_reason2_bench prepare --set-dir <D> --out <dir> --per-state 10
    uv run python -m eval.cosmos_reason2_bench run --out <dir> [--base-url http://127.0.0.1:8000/v1]

``prepare`` picks scenes whose labelled product (the one with a ``build_state`` on its top installed part) has only
``ok`` parts, crops that product from the camera where it is most visible and writes ``captures/<id>.jpg`` plus
``manifest.jsonl``. Ground truth: ``seated`` should pass; ``high``, ``tilted`` and ``wrong_part`` should fail.
``run`` sends every capture through ``CosmosReason2Endpoint.check`` (the production port) and writes
``results.jsonl`` and ``summary.json``: latency p50/p95, how many answers parsed, and the verdict confusion table.
Real-cell images may be benchmarked the same way for evaluation only, never for training.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.cosmos_reason import CosmosReasonUnavailable, FinalBuildRequest, InstalledInstanceLineage
from agent.cosmos_reason2_endpoint import CosmosReason2Endpoint, DirectoryImageSource, EndpointConfig
from assembly.catalog import Catalog, load_catalog

STATES = ("seated", "high", "tilted", "wrong_part")
MARGIN = 0.35
MIN_SIDE = 448


def _focus(spec: dict) -> tuple[dict, list[dict]] | None:
    objects = spec["objects"]
    tops = [o for o in objects if o["state"] == "installed" and o.get("build_state") in STATES]
    if len(tops) != 1:
        return None
    top = tops[0]
    group = [o for o in objects if o.get("product") == top["product"] and o["state"] in ("installed", "on_table")]
    group = [o for o in group if o["kind"] == "base" or o["state"] == "installed"]
    if any(o.get("variant") != "ok" for o in group):
        return None
    return top, group


def _crop_box(camera: list[dict], uids: set[str], top_uid: str) -> tuple[tuple[int, int, int, int], int] | None:
    boxes = [c for c in camera if c["uid"] in uids and c.get("bbox_xyxy")]
    top = [c for c in boxes if c["uid"] == top_uid]
    if not top or top[0]["visible_px"] < 400:
        return None
    x0, y0 = min(c["bbox_xyxy"][0] for c in boxes), min(c["bbox_xyxy"][1] for c in boxes)
    x1, y1 = max(c["bbox_xyxy"][2] for c in boxes), max(c["bbox_xyxy"][3] for c in boxes)
    side = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    return box, top[0]["visible_px"]


def _sample(label_path: Path) -> dict | None:
    data = json.loads(label_path.read_text())
    found = _focus(data["spec"])
    if found is None:
        return None
    top, group = found
    uids = {o["uid"] for o in group}
    views = [(cam, _crop_box(data["cameras"][cam], uids, top["uid"])) for cam in data["cameras"]]
    views = [(cam, view) for cam, view in views if view is not None]
    if not views:
        return None
    camera, (box, _) = max(views, key=lambda item: item[1][1])
    step = data["spec"]["assembly_step"][str(top["product"])]
    return {"label": str(label_path), "camera": camera, "box": box, "product": top["product"], "step": step,
            "state": top["build_state"], "top_object": top["object_id"]}


def _write_capture(sample: dict, rgb_root: Path, out: Path) -> dict:
    from PIL import Image

    label = Path(sample["label"])
    image = Image.open(rgb_root / sample["camera"] / f"{label.stem}.jpg").convert("RGB")
    crop = image.crop(sample["box"])
    if crop.width < MIN_SIDE:
        crop = crop.resize((MIN_SIDE, MIN_SIDE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG", quality=95)
    data = buffer.getvalue()
    capture_id = f"D-{label.parent.parent.name}-{label.stem}-{sample['camera']}"
    (out / "captures" / f"{capture_id}.jpg").write_bytes(data)
    return {**sample, "capture_id": capture_id, "sha256": hashlib.sha256(data).hexdigest()}


def prepare(set_dir: Path, out: Path, per_state: int) -> None:
    (out / "captures").mkdir(parents=True, exist_ok=True)
    taken: Counter[str] = Counter()
    rows = []
    for label_path in sorted(set_dir.glob("chunk_*/labels/*.json")):
        sample = _sample(label_path)
        if sample is None or taken[sample["state"]] >= per_state:
            continue
        rows.append(_write_capture(sample, label_path.parent.parent / "rgb", out))
        taken[sample["state"]] += 1
    (out / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(json.dumps({"captures": len(rows), "per_state": dict(taken)}))


def _request(row: dict, index: int, catalog: Catalog) -> FinalBuildRequest:
    steps = catalog.product(row["product"]).steps[: row["step"]]
    now = datetime.now(UTC)
    return FinalBuildRequest(
        "bench", 0, f"task-{index}", f"request-{index}", row["product"],
        tuple(InstalledInstanceLineage(f"sim-{index}-{s.step}", s.object_id) for s in steps),
        row["capture_id"], row["sha256"], now, now + timedelta(minutes=5), "sim-set-d",
    )


def _one(endpoint: CosmosReason2Endpoint, row: dict, index: int, catalog: Catalog) -> dict:
    started = time.perf_counter()
    try:
        result = endpoint.check(_request(row, index, catalog))
        outcome = {"parsed": True, "passed": result.passed, "reason": result.explanation,
                   "reasoning_chars": len(endpoint.last_verdict.reasoning) if endpoint.last_verdict else 0}
    except CosmosReasonUnavailable as error:
        outcome = {"parsed": False, "error": str(error)}
    return {**row, **outcome, "latency_s": round(time.perf_counter() - started, 3)}


def _summary(results: list[dict]) -> dict:
    latencies = sorted(r["latency_s"] for r in results)
    parsed = [r for r in results if r["parsed"]]
    confusion = Counter(f"{r['state']}->{'pass' if r['passed'] else 'fail'}" for r in parsed)
    correct = sum((r["state"] == "seated") == r["passed"] for r in parsed)
    return {
        "n": len(results), "parsed": len(parsed), "unavailable": len(results) - len(parsed),
        "latency_p50_s": round(statistics.median(latencies), 3),
        "latency_p95_s": round(latencies[max(0, round(0.95 * len(latencies)) - 1)], 3),
        "latency_max_s": latencies[-1], "verdict_correct": correct, "confusion": dict(sorted(confusion.items())),
    }


def run(out: Path, base_url: str) -> None:
    catalog = load_catalog()
    endpoint = CosmosReason2Endpoint(EndpointConfig(base_url=base_url), DirectoryImageSource(out / "captures"), catalog)
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    results = []
    with (out / "results.jsonl").open("w") as sink:
        for index, row in enumerate(rows):
            results.append(_one(endpoint, row, index, catalog))
            sink.write(json.dumps(results[-1]) + "\n")
            sink.flush()
    summary = _summary(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--set-dir", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--per-state", type=int, default=10)
    bench = sub.add_parser("run")
    bench.add_argument("--out", type=Path, required=True)
    bench.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.set_dir, args.out, args.per_state)
    else:
        run(args.out, args.base_url)


if __name__ == "__main__":
    main()
