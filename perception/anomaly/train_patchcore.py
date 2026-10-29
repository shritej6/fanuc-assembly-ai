"""Fit one PatchCore model for one part and measure it (anomalib, single GPU per part, as the user asked).

Reads the folders written by `perception.anomaly.testbed`. Writes `<out>/<part>/metrics.json`:
  image AUROC and F1 on C / I good vs defective close-ups (installed orientation), and the share of good O test
  close-ups flagged as anomalous at the model's own threshold (false alarms on the real-bed view).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

MIN_TEST = 10   # fewer good or bad crops than this leave under 5 per half after the validation split


def _count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.is_dir() else 0


def false_alarm_share(engine: Engine, model: Patchcore, folder: Path) -> float | None:
    if _count(folder) == 0:
        return None
    predictions = engine.predict(model=model, dataset=PredictDataset(path=folder), return_predictions=True) or []
    flags = [bool(label) for batch in predictions for label in torch.as_tensor(batch.pred_label).reshape(-1)]
    return sum(flags) / len(flags) if flags else None


def fit_and_measure(part_dir: Path, out: Path, workers: int) -> dict:
    counts = {name: _count(part_dir / name) for name in ("train_good", "test_good", "test_bad", "o_test_good")}
    result: dict = {"part": part_dir.name, "counts": counts}
    if counts["train_good"] == 0:
        return {**result, "status": "no_training_crops"}
    datamodule = Folder(
        name=part_dir.name,
        root=part_dir,
        normal_dir="train_good",
        normal_test_dir="test_good" if counts["test_good"] else None,
        abnormal_dir="test_bad" if counts["test_bad"] else None,
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=workers,
        val_split_mode="from_test",   # half the C / I test crops set the threshold, the other half measure it
        val_split_ratio=0.5,
        seed=1337,
    )
    model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), coreset_sampling_ratio=0.1)
    engine = Engine(default_root_dir=out / part_dir.name, logger=False, devices=1, enable_progress_bar=False)
    engine.fit(model=model, datamodule=datamodule)
    if min(counts["test_good"], counts["test_bad"]) >= MIN_TEST:
        metrics = engine.test(model=model, datamodule=datamodule, verbose=False)[0]
        result["test"] = {key: float(value) for key, value in metrics.items()}
        result["status"] = "measured"
    else:
        result["status"] = "fitted_too_few_test_crops"
    result["o_test_false_alarm_share"] = false_alarm_share(engine, model, part_dir / "o_test_good")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--crops", type=Path, required=True, help="output of perception.anomaly.testbed")
    ap.add_argument("--part", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    result = fit_and_measure(args.crops / args.part, args.out, args.workers)
    target = args.out / args.part / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
