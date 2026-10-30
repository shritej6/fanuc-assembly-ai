"""Fit one PatchCore model per part and measure only when test support is sufficient.

``train/good`` contains only good Set O crops. ``test/good`` and ``test/bad`` contain only installed C/I targets and
installed Set T rear guards for evaluation. C/I/T never enter PatchCore's memory bank. ``o_test_good`` measures
false alarms on the held-out Set O test split.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import torch
from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore

from perception.anomaly.patchcore_contract import (
    MODEL_CONFIG,
    MODEL_FORMAT,
    REQUIRED_IMAGE_METRICS,
)

MIN_TEST = 10  # leaves at least five good and five bad crops after anomalib's 50/50 validation split
FOLDER_LAYOUT = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_bad": "test/bad",
    "o_test_good": "o_test_good",
}


def _count(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.is_dir() else 0


def false_alarm_share(engine: Engine, model: Patchcore, folder: Path, checkpoint: Path) -> float:
    expected = _count(folder)
    if expected == 0:
        raise RuntimeError("the approved Set O test split must not be empty")
    predictions = (
        engine.predict(
            model=model,
            dataset=PredictDataset(path=folder),
            return_predictions=True,
            ckpt_path=str(checkpoint),
        )
        or []
    )
    flags = [bool(label) for batch in predictions for label in torch.as_tensor(batch.pred_label).reshape(-1)]
    if len(flags) != expected:
        raise RuntimeError("Set O false-alarm prediction count disagrees with its input crops")
    return sum(flags) / len(flags)


def _finite_metrics(raw: dict) -> dict[str, float]:
    required = REQUIRED_IMAGE_METRICS
    if any(key not in raw for key in required):
        raise RuntimeError("PatchCore test metrics lack image AUROC or F1")
    if any(isinstance(raw[key], bool) for key in required):
        raise RuntimeError("PatchCore image AUROC and F1 must be numeric, not boolean")
    try:
        metrics = {key: float(raw[key]) for key in required}
    except (TypeError, ValueError) as exc:
        raise RuntimeError("PatchCore test metrics must be numeric") from exc
    required_values = (metrics[key] for key in required)
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in required_values):
        raise RuntimeError("PatchCore test metrics must be non-empty and finite")
    return metrics


def _preserve_checkpoint(engine: Engine, part_root: Path) -> Path:
    best_model_path = engine.best_model_path
    if not isinstance(best_model_path, (str, Path)) or not str(best_model_path):
        raise RuntimeError("anomalib did not report a best checkpoint")
    source = Path(best_model_path)
    invalid = source.is_symlink() or not source.is_file() or source.stat().st_size <= 0
    if invalid or not source.resolve().is_relative_to(part_root.resolve()):
        raise RuntimeError("anomalib best checkpoint is missing or outside its part output")
    target = part_root / "model.ckpt"
    if target.exists():
        raise RuntimeError("stable PatchCore checkpoint path must be fresh")
    shutil.copyfile(source, target)
    if target.stat().st_size <= 0:
        raise RuntimeError("stable PatchCore checkpoint is empty")
    return target


def _preflight(part: str, counts: dict[str, int]) -> dict | None:
    result: dict = {"part": part, "counts": counts}
    evaluation_support = {
        "minimum_per_class": MIN_TEST,
        "observed": {"test_good": counts["test_good"], "test_bad": counts["test_bad"]},
    }
    if min(evaluation_support["observed"].values()) < MIN_TEST:
        return {
            **result,
            "evaluation_support": evaluation_support,
            "status": "insufficient_evaluation_support",
        }
    if counts["train_good"] == 0:
        return {**result, "status": "no_training_crops"}
    return None


def _runtime(part_dir: Path, out: Path, workers: int) -> tuple[Folder, Patchcore, Engine]:
    datamodule = Folder(
        name=part_dir.name,
        root=part_dir,
        normal_dir=FOLDER_LAYOUT["train_good"],
        normal_test_dir=FOLDER_LAYOUT["test_good"],
        abnormal_dir=FOLDER_LAYOUT["test_bad"],
        train_batch_size=32,
        eval_batch_size=32,
        num_workers=workers,
        val_split_mode="from_test",
        val_split_ratio=0.5,
        seed=1337,
    )
    model = Patchcore(
        backbone=MODEL_CONFIG["backbone"],
        layers=tuple(MODEL_CONFIG["layers"]),
        coreset_sampling_ratio=MODEL_CONFIG["coreset_sampling_ratio"],
        num_neighbors=MODEL_CONFIG["num_neighbors"],
    )
    engine = Engine(default_root_dir=out / part_dir.name, logger=False, devices=1, enable_progress_bar=False)
    return datamodule, model, engine


def _measure(part_dir: Path, out: Path, datamodule: Folder, model: Patchcore, engine: Engine) -> dict:
    engine.fit(model=model, datamodule=datamodule)
    checkpoint = _preserve_checkpoint(engine, out / part_dir.name)
    measured = engine.test(model=model, datamodule=datamodule, verbose=False, ckpt_path=str(checkpoint))
    if len(measured) != 1 or not isinstance(measured[0], dict):
        raise RuntimeError("PatchCore test must return exactly one metric mapping")
    result = {
        "test": _finite_metrics(measured[0]),
        "o_test_false_alarm_share": false_alarm_share(
            engine, model, part_dir / FOLDER_LAYOUT["o_test_good"], checkpoint
        ),
        "status": "measured",
    }
    model_info = {
        "path": checkpoint.relative_to(out / part_dir.name).as_posix(),
        "format": MODEL_FORMAT,
        "size_bytes": checkpoint.stat().st_size,
        "config": MODEL_CONFIG,
    }
    return {**result, "model": model_info}


def fit_and_measure(part_dir: Path, out: Path, workers: int) -> dict:
    counts = {name: _count(part_dir / relative) for name, relative in FOLDER_LAYOUT.items()}
    result: dict = {"part": part_dir.name, "counts": counts}
    rejected = _preflight(part_dir.name, counts)
    if rejected is not None:
        return rejected
    datamodule, model, engine = _runtime(part_dir, out, workers)
    measured = _measure(part_dir, out, datamodule, model, engine)
    result.update(measured)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crops", type=Path, required=True)
    parser.add_argument("--part", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = fit_and_measure(args.crops / args.part, args.out, args.workers)
    target = args.out / args.part / "metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
