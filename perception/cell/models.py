"""Load the trained models once and run them on live cell images.
Authored by Shritej Gunda.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from perception.anomaly.testbed import square_crop
from perception.crops import _crop_box

CLASSIFIER_CONTEXT = 1.3


class ModelLoadError(RuntimeError):
    """A model directory is missing or does not match what the cell runtime expects."""


@dataclass(frozen=True, slots=True)
class Box:
    """One detection: label, score and pixel box."""
    label: str
    score: float
    bbox_px: tuple[float, float, float, float]


def classifier_crop(image: Image.Image, bbox_px: tuple[float, float, float, float]) -> Image.Image:
    """Return the classifier crop around a box."""
    return image.crop(_crop_box([*bbox_px], image.size, CLASSIFIER_CONTEXT))


def patchcore_crop(image: Image.Image, bbox_px: tuple[float, float, float, float]) -> Image.Image:
    """Return the square PatchCore crop around a box."""
    return square_crop(image, [round(v) for v in bbox_px])


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
    """RT-DETR detector loaded from its run directory."""
    def __init__(self, run_dir: Path, min_score: float) -> None:
        from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

        if not (run_dir / "config.json").is_file():
            raise ModelLoadError(f"{run_dir}: no saved detector (config.json)")
        self.device = _device()
        self.processor = RTDetrImageProcessor.from_pretrained(run_dir)
        self.model = RTDetrForObjectDetection.from_pretrained(run_dir).to(self.device).eval()
        self.min_score = min_score

    def __call__(self, image: Image.Image) -> list[Box]:
        import torch

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        result = self.processor.post_process_object_detection(
            outputs, threshold=self.min_score, target_sizes=[(image.height, image.width)])[0]
        labels = self.model.config.id2label
        return [Box(labels[int(k)], float(s), tuple(float(v) for v in b))
                for s, k, b in zip(result["scores"], result["labels"], result["boxes"], strict=True)]


class Classifier:
    """Wrist ResNet classifier loaded from its run directory."""
    def __init__(self, run_dir: Path, labels: tuple[str, ...], image_size: int) -> None:
        from safetensors.torch import load_file

        from perception.classifier.train import _build_model, _build_transforms

        weights = run_dir / "model.safetensors"
        if not weights.is_file():
            raise ModelLoadError(f"{run_dir}: no saved classifier (model.safetensors)")
        self.device = _device()
        self.model = _build_model(len(labels))
        self.model.load_state_dict(load_file(weights))
        self.model.to(self.device).eval()
        self.labels = labels
        self.transform = _build_transforms(image_size)[1]

    def __call__(self, crop: Image.Image) -> tuple[str, float]:
        import torch

        batch = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs = torch.softmax(self.model(pixel_values=batch).logits[0], dim=0)
        best = int(probs.argmax())
        return self.labels[best], float(probs[best])


def patchcore_raw_scores(model: Any, images: list[Any], device: Any) -> list[float]:
    """Return raw PatchCore image scores, pre-processing each crop on its own."""
    import torch

    batch = torch.cat([model.pre_processor(image.unsqueeze(0)) for image in images]).to(device)
    return [float(v) for v in model.model(batch).pred_score.reshape(-1)]


class PatchCoreBank:
    """One fitted PatchCore per part; a part without a checkpoint is a load error, not a pass."""

    def __init__(self, root: Path, parts: tuple[str, ...]) -> None:
        from anomalib.models import Patchcore

        self.device = _device()
        self.models: dict[str, Any] = {}
        for part in parts:
            ckpt = root / "runs" / part / "model.ckpt"
            if not ckpt.is_file():
                raise ModelLoadError(f"{root}: no PatchCore checkpoint for {part}")
            self.models[part] = Patchcore.load_from_checkpoint(ckpt).to(self.device).eval()

    def __call__(self, part: str, crop: Image.Image) -> float:
        import torch
        from torchvision.transforms.functional import to_tensor

        with torch.inference_mode():
            return patchcore_raw_scores(self.models[part], [to_tensor(crop.convert("RGB"))], self.device)[0]
