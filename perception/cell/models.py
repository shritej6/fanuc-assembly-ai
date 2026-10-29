"""The trained T27 models, loaded once and run on live cell images (GPU when present).

- detector: the RT-DETR run directory written by perception/detector/train.py (`trainer.save_model()`: config with
  id2label, weights, preprocessor), run at the size it was trained at;
- classifier: the wrist ResNet-18 run directory written by perception/classifier/train.py; its labels are the
  ontology's classifier labels and its crops are the training crops (target box x 1.3, perception/crops.py), resized
  and centre-cropped as the evaluation transform does;
- PatchCore: one anomalib checkpoint per part (perception/brev/h100_t14.sh); its crops are the training crops (square,
  context 1.3, perception/anomaly/testbed.py) and its score is anomalib's normalised score, whose decision threshold
  (0.5) anomalib set from that part's measured validation split.
Torch, transformers and anomalib load lazily, so the decision logic around them stays importable and testable.
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
    label: str
    score: float
    bbox_px: tuple[float, float, float, float]


def classifier_crop(image: Image.Image, bbox_px: tuple[float, float, float, float]) -> Image.Image:
    return image.crop(_crop_box([*bbox_px], image.size, CLASSIFIER_CONTEXT))


def patchcore_crop(image: Image.Image, bbox_px: tuple[float, float, float, float]) -> Image.Image:
    return square_crop(image, [round(v) for v in bbox_px])


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class Detector:
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


class PatchCoreBank:
    """One fitted PatchCore per part; a part without a checkpoint is a load error, not a pass."""

    def __init__(self, root: Path, parts: tuple[str, ...]) -> None:
        from anomalib.models import Patchcore

        self.models: dict[str, Any] = {}
        for part in parts:
            ckpts = sorted((root / "runs" / part).rglob("*.ckpt"))
            if not ckpts:
                raise ModelLoadError(f"{root}: no PatchCore checkpoint for {part}")
            self.models[part] = Patchcore.load_from_checkpoint(ckpts[-1]).eval()

    def __call__(self, part: str, crop: Image.Image) -> float:
        import torch
        from torchvision.transforms.functional import to_tensor

        with torch.inference_mode():
            result = self.models[part](to_tensor(crop.convert("RGB")).unsqueeze(0))
        return float(result.pred_score.reshape(-1)[0])
