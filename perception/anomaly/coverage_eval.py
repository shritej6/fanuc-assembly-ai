"""Fit one PatchCore per part on good O close-ups and measure it on held-out crops.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

MIN_EVAL = 10           # fewer good or bad crops than this in a split leave the part unmeasured
MAX_VAL_FPR = 0.02
REPORTED_FPRS = (0.01, 0.02, 0.05)   # extra operating points, each threshold chosen on val, read on test
SEED = 1337


def auroc(good: np.ndarray, bad: np.ndarray) -> float:
    """Return the Mann-Whitney AUROC: the chance a defective crop scores above a good one."""
    diff = bad[:, None] - good[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def threshold_for_fpr(good: np.ndarray, max_fpr: float) -> float:
    """Return the lowest threshold whose false alarm share is at most max_fpr."""
    ordered = np.sort(good)[::-1]
    allowed = math.floor(max_fpr * len(ordered))
    return float(ordered[allowed]) if allowed < len(ordered) else float(ordered[-1])


def rates(good: np.ndarray, bad: np.ndarray, threshold: float) -> dict:
    """Return false alarm and catch rates at one threshold."""
    return {"fpr": float((good > threshold).mean()), "tpr": float((bad > threshold).mean()),
            "false_alarms": int((good > threshold).sum()), "good": len(good),
            "caught": int((bad > threshold).sum()), "bad": len(bad)}


def summarise(rows: list[dict], max_val_fpr: float) -> dict:
    """Pick thresholds on val and return AUROC and rates on test, overall, per set and per defect."""
    def scores(split: str, good: bool, **match: str) -> np.ndarray:
        """Return the scores of the rows that match a split, label and filters."""
        return np.array([r["score"] for r in rows if r["split"] == split and (r["variant"] == "ok") == good
                         and all(r[k] == v for k, v in match.items())], dtype=np.float64)

    counts = {f"{s}_{k}": len(scores(s, k == "good")) for s in ("val", "test") for k in ("good", "bad")}
    if min(counts.values()) < MIN_EVAL:
        return {"status": "insufficient_evaluation_support", "counts": counts, "min_eval": MIN_EVAL}
    threshold = threshold_for_fpr(scores("val", True), max_val_fpr)
    test_good, test_bad = scores("test", True), scores("test", False)
    per_type = {}
    for variant in sorted({r["variant"] for r in rows if r["variant"] != "ok"}):
        bad = scores("test", False, variant=variant)
        if len(bad):
            per_type[variant] = {"tpr": float((bad > threshold).mean()), "bad": len(bad),
                                 "auroc": auroc(test_good, bad)}
    per_set_fpr = {}
    for set_name in sorted({r["set"] for r in rows}):
        good = scores("test", True, set=set_name)
        if len(good):
            per_set_fpr[set_name] = {"fpr": float((good > threshold).mean()), "good": len(good)}
    points = {}
    for fpr in REPORTED_FPRS:
        t = threshold_for_fpr(scores("val", True), fpr)
        points[f"{fpr:.2f}"] = {"threshold": t, **rates(test_good, test_bad, t)}
    return {"status": "measured", "counts": counts, "threshold": threshold, "max_val_fpr": max_val_fpr,
            "val_auroc": auroc(scores("val", True), scores("val", False)), "test_operating_points": points,
            "val": rates(scores("val", True), scores("val", False), threshold),
            "test": {"auroc": auroc(test_good, test_bad), **rates(test_good, test_bad, threshold)},
            "test_per_defect": per_type, "test_fpr_per_set": per_set_fpr}


DEFAULT_CONFIG = {"image_size": 256, "layers": ("layer2", "layer3"), "coreset": 0.1}


def fit(part_dir: Path, out: Path, workers: int, config: dict):
    """Fit PatchCore on the good training crops only and return the model."""
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    datamodule = Folder(name=part_dir.name, root=part_dir, normal_dir="train_good", train_batch_size=32,
                        eval_batch_size=32, num_workers=workers, test_split_mode="none", val_split_mode="none",
                        seed=SEED)
    size = (config["image_size"], config["image_size"])
    model = Patchcore(backbone="wide_resnet50_2", layers=tuple(config["layers"]),
                      coreset_sampling_ratio=config["coreset"],
                      pre_processor=Patchcore.configure_pre_processor(image_size=size))
    engine = Engine(default_root_dir=out / part_dir.name, logger=False, devices=1, enable_progress_bar=False)
    datamodule.setup()
    # train loader only: with no validation loop, anomalib fits the memory bank at the end of the epoch
    engine.fit(model=model, train_dataloaders=datamodule.train_dataloader())
    if not bool(model._is_fitted):
        raise RuntimeError(f"{part_dir.name}: PatchCore memory bank was not fitted")
    engine.trainer.save_checkpoint(out / part_dir.name / "model.ckpt")
    return model



def raw_scores(model, paths: list[Path], batch: int = 32) -> list[float]:
    """Return PatchCore's raw image score for each crop."""
    import torch
    from PIL import Image
    from torchvision.transforms.functional import to_tensor

    from perception.cell.models import patchcore_raw_scores

    # Lightning hands the model back on the CPU after fit, and in training mode PatchCore returns embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out: list[float] = []
    for start in range(0, len(paths), batch):
        images = [to_tensor(Image.open(p).convert("RGB")) for p in paths[start:start + batch]]
        with torch.inference_mode():
            out.extend(patchcore_raw_scores(model, images, device))
    return out


def measure(part_dir: Path, out: Path, workers: int, max_val_fpr: float, config: dict) -> dict:
    """Fit one part, score its crops and return the summary."""
    index = [json.loads(line) for line in (part_dir / "index.jsonl").read_text().splitlines()]
    train = len(list((part_dir / "train_good").glob("*.png")))
    if train == 0:
        return {"part": part_dir.name, "status": "no_training_crops"}
    model = fit(part_dir, out, workers, config)
    for row, score in zip(index, raw_scores(model, [part_dir / r["file"] for r in index]), strict=True):
        row["score"] = score
    (out / part_dir.name / "scores.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in index))
    return {"part": part_dir.name, "train_good": train, "config": config, **summarise(index, max_val_fpr)}


def resummarise(run_dir: Path, max_val_fpr: float) -> dict:
    """Recompute a finished part's summary from its scores.jsonl."""
    previous = json.loads((run_dir / "metrics.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "scores.jsonl").read_text().splitlines()]
    keep = {k: previous[k] for k in ("part", "train_good", "config") if k in previous}
    return {**keep, **summarise(rows, max_val_fpr)}


def main() -> None:
    """Parse arguments, measure one part and write its results."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crops", type=Path, required=True, help="output of perception.anomaly.coverage")
    ap.add_argument("--part", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-val-fpr", type=float, default=MAX_VAL_FPR)
    ap.add_argument("--image-size", type=int, default=DEFAULT_CONFIG["image_size"])
    ap.add_argument("--layers", default=",".join(DEFAULT_CONFIG["layers"]))
    ap.add_argument("--coreset", type=float, default=DEFAULT_CONFIG["coreset"])
    ap.add_argument("--resummarise", action="store_true", help="recompute metrics.json from scores.jsonl only")
    args = ap.parse_args()
    if args.resummarise:
        result = resummarise(args.out / args.part, args.max_val_fpr)
    else:
        config = {"image_size": args.image_size, "layers": args.layers.split(","), "coreset": args.coreset}
        result = measure(args.crops / args.part, args.out, args.workers, args.max_val_fpr, config)
    target = args.out / args.part / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
