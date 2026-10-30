from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image

from perception.anomaly.dataset import PatchCoreSample, prepare_patchcore_folders

FOLDER_LAYOUT = {
    "train_good": "train/good",
    "test_good": "test/good",
    "test_bad": "test/bad",
    "o_test_good": "o_test_good",
}


def _runtime_types(reject_runtime: bool):
    class RuntimeObject:
        def __init__(self, *args, **kwargs) -> None:
            if reject_runtime:
                raise AssertionError("anomalib runtime was constructed before evaluation preflight")

    class Engine(RuntimeObject):
        fitted = False

        def fit(self, **kwargs) -> None:
            self.fitted = True

        def test(self, **kwargs) -> list[dict[str, float]]:
            if not self.fitted:
                raise AssertionError("PatchCore was tested before fitting")
            return [{"image_AUROC": 0.75, "image_F1Score": 0.5}]

        def predict(self, **kwargs) -> list:
            return []

    return RuntimeObject, Engine


@pytest.fixture
def train_module_loader(monkeypatch: pytest.MonkeyPatch):
    def load(*, reject_runtime: bool):
        RuntimeObject, Engine = _runtime_types(reject_runtime)
        torch = ModuleType("torch")
        torch.as_tensor = lambda value: SimpleNamespace(reshape=lambda _: value)
        anomalib = ModuleType("anomalib")
        data = ModuleType("anomalib.data")
        data.Folder = RuntimeObject
        data.PredictDataset = RuntimeObject
        engine = ModuleType("anomalib.engine")
        engine.Engine = Engine
        models = ModuleType("anomalib.models")
        models.Patchcore = RuntimeObject
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "anomalib", anomalib)
        monkeypatch.setitem(sys.modules, "anomalib.data", data)
        monkeypatch.setitem(sys.modules, "anomalib.engine", engine)
        monkeypatch.setitem(sys.modules, "anomalib.models", models)
        sys.modules.pop("perception.anomaly.train_patchcore", None)
        return importlib.import_module("perception.anomaly.train_patchcore")

    yield load
    sys.modules.pop("perception.anomaly.train_patchcore", None)


def _write_crops(part_dir: Path, split: str, count: int) -> None:
    folder = part_dir / FOLDER_LAYOUT[split]
    folder.mkdir(parents=True)
    for index in range(count):
        (folder / f"{index:03d}.png").write_bytes(b"crop")


def _prepared_sample(root: Path, index: int, variant: str) -> PatchCoreSample:
    rgb = root / f"source-{index:03d}.jpg"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(rgb)
    return PatchCoreSample(
        set_name="C",
        sample_id=f"C/chunk_0000/{index:06d}",
        object_id="M1_04",
        variant=variant,
        stable_pose=0,
        eligibility="train_eligible" if variant == "ok" else "test_only",
        bbox_xyxy=(8, 8, 56, 56),
        segmentation_value=4,
        rgb_path=rgb,
        depth_path=root / f"depth-{index:03d}.png",
        segmentation_path=root / f"seg-{index:03d}.png",
        rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
    )


def test_patchcore_consumes_the_prepared_anomalib_layout(tmp_path: Path, train_module_loader) -> None:
    samples = tuple(_prepared_sample(tmp_path, index, "ok") for index in range(20))
    samples += tuple(_prepared_sample(tmp_path, index, "warp") for index in range(20, 30))
    crops = tmp_path / "crops"
    summary = prepare_patchcore_folders(
        samples,
        crops,
        expected_object_ids=("M1_04",),
        normal_test_modulus=2,
    )
    train = train_module_loader(reject_runtime=False)

    result = train.fit_and_measure(crops / "M1_04", tmp_path / "runs", workers=0)

    assert (summary.train_good, summary.test_good, summary.test_bad) == (10, 10, 10)
    assert result["counts"] == {"train_good": 10, "test_good": 10, "test_bad": 10, "o_test_good": 0}
    assert result["status"] == "measured"


@pytest.mark.parametrize(("test_good", "test_bad"), [(0, 10), (10, 0), (9, 10), (10, 9)])
def test_insufficient_evaluation_support_is_reported_before_patchcore_fit(
    tmp_path: Path,
    train_module_loader,
    test_good: int,
    test_bad: int,
) -> None:
    train = train_module_loader(reject_runtime=True)
    part_dir = tmp_path / "M1_04"
    _write_crops(part_dir, "train_good", 20)
    _write_crops(part_dir, "test_good", test_good)
    _write_crops(part_dir, "test_bad", test_bad)

    result = train.fit_and_measure(part_dir, tmp_path / "runs", workers=0)

    assert result == {
        "part": "M1_04",
        "counts": {
            "train_good": 20,
            "test_good": test_good,
            "test_bad": test_bad,
            "o_test_good": 0,
        },
        "evaluation_support": {
            "minimum_per_class": 10,
            "observed": {"test_good": test_good, "test_bad": test_bad},
        },
        "status": "insufficient_evaluation_support",
    }


def test_missing_training_and_evaluation_data_reports_evaluation_support_first(
    tmp_path: Path, train_module_loader
) -> None:
    train = train_module_loader(reject_runtime=True)

    result = train.fit_and_measure(tmp_path / "M2_04", tmp_path / "runs", workers=0)

    assert result["status"] == "insufficient_evaluation_support"
    assert result["evaluation_support"] == {
        "minimum_per_class": 10,
        "observed": {"test_good": 0, "test_bad": 0},
    }


def test_missing_evaluation_folder_is_insufficient_support(tmp_path: Path, train_module_loader) -> None:
    train = train_module_loader(reject_runtime=True)
    part_dir = tmp_path / "M3_04"
    _write_crops(part_dir, "train_good", 20)
    _write_crops(part_dir, "test_bad", 10)

    result = train.fit_and_measure(part_dir, tmp_path / "runs", workers=0)

    assert result["status"] == "insufficient_evaluation_support"
    assert result["evaluation_support"]["observed"] == {"test_good": 0, "test_bad": 10}


def test_complete_evaluation_support_runs_patchcore_and_reports_metrics(tmp_path: Path, train_module_loader) -> None:
    train = train_module_loader(reject_runtime=False)
    part_dir = tmp_path / "M1_01"
    for split, count in (("train_good", 20), ("test_good", 10), ("test_bad", 10)):
        _write_crops(part_dir, split, count)

    result = train.fit_and_measure(part_dir, tmp_path / "runs", workers=0)

    assert result["status"] == "measured"
    assert result["test"] == {"image_AUROC": 0.75, "image_F1Score": 0.5}
    assert result["o_test_false_alarm_share"] is None
