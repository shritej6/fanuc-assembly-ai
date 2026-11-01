from __future__ import annotations

import hashlib
import importlib
import json
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
            self.args = args
            self.kwargs = kwargs

    class Engine(RuntimeObject):
        fitted = False

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.root = Path(kwargs["default_root_dir"])
            self.best_model_path = ""

        def fit(self, **kwargs) -> None:
            self.fitted = True
            checkpoint = self.root / "weights/lightning/model.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"measured-patchcore-checkpoint")
            self.best_model_path = str(checkpoint)

        def test(self, **kwargs) -> list[dict[str, float]]:
            if not self.fitted:
                raise AssertionError("PatchCore was tested before fitting")
            if Path(kwargs["ckpt_path"]) != self.root / "model.ckpt":
                raise AssertionError("PatchCore test did not use the preserved checkpoint")
            return [{"image_AUROC": 0.75, "image_F1Score": 0.5}]

        def predict(self, **kwargs) -> list:
            if Path(kwargs["ckpt_path"]) != self.root / "model.ckpt":
                raise AssertionError("PatchCore prediction did not use the preserved checkpoint")
            folder = Path(kwargs["dataset"].kwargs["path"])
            labels = [index % 2 for index, _ in enumerate(folder.glob("*.png"))]
            return [SimpleNamespace(pred_label=labels)]

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


def _o_indices(split: str, count: int) -> list[int]:
    found = []
    for index in range(10_000):
        token = f"1337:O:{index}".encode()
        bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10_000
        assigned = "train" if bucket < 8_000 else "validation" if bucket < 9_000 else "test"
        if assigned == split:
            found.append(index)
        if len(found) == count:
            return found
    raise AssertionError(f"not enough O indices for {split}")


def _source_evidence(root: Path, set_name: str, index: int, rest: str | None) -> tuple[Path, ...]:
    depth = root / f"depth-{set_name}-{index:03d}.png"
    segmentation = root / f"seg-{set_name}-{index:03d}.png"
    label = root / f"label-{set_name}-{index:03d}.json"
    depth.write_bytes(b"depth")
    segmentation.write_bytes(b"segmentation")
    label.write_text("{}")
    if rest is None:
        return depth, segmentation, label
    sidecar = root / f"rest-{set_name}-{index:03d}.json"
    sidecar.write_text(json.dumps({"rest": rest}))
    return depth, segmentation, label, sidecar


def _prepared_sample(
    root: Path,
    set_name: str,
    index: int,
    variant: str,
    eligibility: str,
    assigned_split: str,
    rest: str | None,
) -> PatchCoreSample:
    rgb = root / f"source-{set_name}-{index:03d}.jpg"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(rgb)
    evidence = _source_evidence(root, set_name, index, rest)
    depth, segmentation, label = evidence[:3]
    sidecar = evidence[3] if len(evidence) == 4 else None
    return PatchCoreSample(
        set_name=set_name,
        sample_id=f"{set_name}/chunk_0000/{index:06d}",
        object_id="M1_04",
        variant=variant,
        stable_pose=0,
        eligibility=eligibility,
        bbox_xyxy=(8, 8, 56, 56),
        segmentation_value=4,
        rgb_path=rgb,
        depth_path=depth,
        segmentation_path=segmentation,
        label_path=label,
        rgb_sha256=hashlib.sha256(rgb.read_bytes()).hexdigest(),
        depth_sha256=hashlib.sha256(depth.read_bytes()).hexdigest(),
        segmentation_sha256=hashlib.sha256(segmentation.read_bytes()).hexdigest(),
        label_sha256=hashlib.sha256(label.read_bytes()).hexdigest(),
        rest_sidecar_path=sidecar,
        rest_sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest() if sidecar else None,
        rest=rest,
        assigned_split=assigned_split,
    )


def _complete_samples(root: Path) -> tuple[PatchCoreSample, ...]:
    samples = tuple(
        _prepared_sample(root, "O", index, "ok", "train_eligible", "train/good", None)
        for index in _o_indices("train", 10)
    )
    samples += (
        _prepared_sample(
            root,
            "O",
            _o_indices("test", 1)[0],
            "ok",
            "false_alarm_only",
            "o_test_good",
            None,
        ),
    )
    samples += tuple(
        _prepared_sample(root, "C", index, "ok", "evaluation_only", "test/good", "installed")
        for index in range(100, 110)
    )
    samples += tuple(
        _prepared_sample(root, "I", index, "warp", "evaluation_only", "test/bad", "installed")
        for index in range(200, 210)
    )
    return samples


def test_patchcore_consumes_the_prepared_anomalib_layout(tmp_path: Path, train_module_loader) -> None:
    samples = _complete_samples(tmp_path)
    crops = tmp_path / "crops"
    summary = prepare_patchcore_folders(
        samples,
        crops,
        expected_object_ids=("M1_04",),
    )
    train = train_module_loader(reject_runtime=False)

    result = train.fit_and_measure(crops / "M1_04", tmp_path / "runs", workers=0)

    assert "only good Set O crops" in train.__doc__
    assert (summary.train_good, summary.test_good, summary.test_bad) == (10, 10, 10)
    assert result["counts"] == {"train_good": 10, "test_good": 10, "test_bad": 10, "o_test_good": 1}
    assert result["status"] == "measured"
    assert result["model"] == {
        "path": "model.ckpt",
        "format": "anomalib-lightning-ckpt",
        "size_bytes": 29,
        "config": {
            "backbone": "wide_resnet50_2",
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": 0.1,
            "num_neighbors": 9,
        },
    }
    assert (tmp_path / "runs/M1_04/model.ckpt").is_file()


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
    _write_crops(part_dir, "o_test_good", 2)

    result = train.fit_and_measure(part_dir, tmp_path / "runs", workers=0)

    assert result["status"] == "measured"
    assert result["test"] == {"image_AUROC": 0.75, "image_F1Score": 0.5}
    assert result["o_test_false_alarm_share"] == 0.5
    assert result["model"]["path"] == "model.ckpt"


@pytest.mark.parametrize("best_model_path", ["", None])
def test_missing_best_checkpoint_is_rejected(tmp_path: Path, train_module_loader, best_model_path) -> None:
    train = train_module_loader(reject_runtime=False)
    engine = SimpleNamespace(best_model_path=best_model_path)

    with pytest.raises(RuntimeError, match="best checkpoint"):
        train._preserve_checkpoint(engine, tmp_path)
