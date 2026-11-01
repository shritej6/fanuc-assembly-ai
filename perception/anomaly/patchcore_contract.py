"""Dependency-light immutable contract for T14 PatchCore evidence."""

from __future__ import annotations

MODEL_FORMAT = "anomalib-lightning-ckpt"
MODEL_CONFIG = {
    "backbone": "wide_resnet50_2",
    "layers": ["layer2", "layer3"],
    "coreset_sampling_ratio": 0.1,
    "num_neighbors": 9,
}
REQUIRED_IMAGE_METRICS = ("image_AUROC", "image_F1Score")
