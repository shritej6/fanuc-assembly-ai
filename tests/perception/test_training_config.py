import tomllib
from pathlib import Path

import pytest
import yaml

from perception.training_config import ConfigError, apply_smoke_step_override, load_training_config

ROOT = Path(__file__).resolve().parents[2]
DETECTOR = ROOT / "perception" / "configs" / "rtdetr_t4.yaml"
CLASSIFIER = ROOT / "perception" / "configs" / "classifier_t4.yaml"
H100_DETECTOR = ROOT / "perception" / "configs" / "rtdetr_h100.yaml"
H100_CLASSIFIER = ROOT / "perception" / "configs" / "classifier_h100.yaml"


def test_ml_group_pins_security_compatible_torch_pair() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())

    dependencies = set(project["dependency-groups"]["ml"])

    assert "torch==2.14.1" in dependencies
    assert "torchvision==0.29.1" in dependencies
    assert "transformers==5.21.0" in dependencies


def test_detector_config_pins_rtdetr_and_t4_safe_trainer_flags() -> None:
    config = load_training_config(DETECTOR)

    assert config["task"] == "detector"
    assert config["hardware_profile"] == "t4"
    assert config["model"] == {
        "name": "PekingU/rtdetr_r50vd",
        "revision": "df939e661d8c52e80608d1ec566561aabd25a4e7",
        "image_size": 640,
    }
    assert config["training"]["fp16"] is True
    assert config["training"]["bf16"] is False
    assert config["training"]["tf32"] is False
    assert config["training"]["fp16_initial_scale"] == 128
    assert config["training"]["remove_unused_columns"] is False
    assert config["training"]["eval_do_concat_batches"] is False
    assert config["training"]["metric_for_best_model"] == "eval_map"
    assert config["training"]["greater_is_better"] is True


def test_smoke_overlay_is_bounded_and_keeps_the_production_identity() -> None:
    production = load_training_config(DETECTOR)
    smoke = load_training_config(DETECTOR, smoke=True)

    assert smoke["model"] == production["model"]
    assert smoke["training"]["max_train_samples"] == 16
    assert smoke["training"]["max_eval_samples"] == 8
    assert smoke["training"]["max_steps"] == 2


def test_smoke_step_override_allows_resume_probe_but_never_changes_production() -> None:
    smoke = load_training_config(DETECTOR, smoke=True)

    updated = apply_smoke_step_override(smoke, smoke=True, max_steps=3)

    assert updated["training"]["max_steps"] == 3
    assert smoke["training"]["max_steps"] == 2
    with pytest.raises(ConfigError, match="smoke"):
        apply_smoke_step_override(smoke, smoke=False, max_steps=3)
    with pytest.raises(ConfigError, match=r"\[1, 5\]"):
        apply_smoke_step_override(smoke, smoke=True, max_steps=6)
    assert smoke["training"]["eval_strategy"] == "steps"
    assert smoke["training"]["save_strategy"] == "steps"


def test_classifier_config_requires_pretrained_resnet18_and_30_outputs() -> None:
    config = load_training_config(CLASSIFIER)

    assert config["task"] == "classifier"
    assert config["hardware_profile"] == "t4"
    assert config["model"] == {
        "architecture": "resnet18",
        "weights": "DEFAULT",
        "num_labels": 30,
        "image_size": 224,
    }
    assert config["training"]["fp16"] is True
    assert config["training"]["bf16"] is False
    assert config["training"]["tf32"] is False
    assert config["training"]["fp16_initial_scale"] == 128


@pytest.mark.parametrize(
    ("path", "task", "image_size", "batch_size", "workers"),
    [
        (H100_DETECTOR, "detector", 1280, 20, 8),
        (H100_CLASSIFIER, "classifier", 224, 64, 4),
    ],
)
def test_h100_configs_use_bf16_without_an_fp16_scaler(
    path: Path,
    task: str,
    image_size: int,
    batch_size: int,
    workers: int,
) -> None:
    config = load_training_config(path)

    assert config["task"] == task
    assert config["hardware_profile"] == "h100"
    assert config["model"]["image_size"] == image_size
    assert config["training"]["fp16"] is False
    assert config["training"]["bf16"] is True
    assert config["training"]["tf32"] is False
    assert "fp16_initial_scale" not in config["training"]
    assert config["training"]["gradient_accumulation_steps"] == 1
    assert config["training"]["per_device_train_batch_size"] == batch_size
    assert config["training"]["dataloader_num_workers"] == workers


@pytest.mark.parametrize(
    ("base", "mutation", "message"),
    [
        (DETECTOR, ("training", "bf16", True), "BF16"),
        (DETECTOR, ("training", "tf32", True), "TF32"),
        (H100_DETECTOR, ("training", "fp16", True), "H100.*FP16"),
        (H100_DETECTOR, ("model", "image_size", 640), "1280"),
        (CLASSIFIER, ("model", "num_labels", 29), "30 labels"),
        (CLASSIFIER, ("training", "fp16_initial_scale", 65_536), "fp16_initial_scale"),
    ],
)
def test_unsafe_or_wrong_label_configs_are_rejected(
    tmp_path: Path,
    base: Path,
    mutation: tuple[str, str, object],
    message: str,
) -> None:
    payload = yaml.safe_load(base.read_text())
    section, key, value = mutation
    payload[section][key] = value
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ConfigError, match=message):
        load_training_config(path)
