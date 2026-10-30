"""What the port needs from the trained models and the calibration, as typed protocols plus the real adapters.

Tests inject deterministic fakes; the cell runtime injects the adapters below (torch and transformers load lazily, on
the GPU host). Every reading is validated here, so a malformed model output is a typed error, never a silent pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from perception.cell.localize import Located, Zone
from perception.cell.models import ModelLoadError, classifier_crop
from perception.port.dedup import RawBox
from perception.port.errors import ModelOutputError
from robot.adapter import Observation


@dataclass(frozen=True, slots=True)
class WristReading:
    """Wrist classifier output for one crop: top identity, its softmax probability, and the defect head's p_defect."""

    object_id: str
    p_identity: float
    p_defect: float

    def __post_init__(self) -> None:
        for name in ("p_identity", "p_defect"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ModelOutputError(f"wrist {name} must be a probability, got {value!r}")


@dataclass(frozen=True, slots=True)
class PoseFit:
    """FoundationPose result: the STL mesh pose in UF0 (mm) and its fit score in [0, 1]."""

    world_from_stl: np.ndarray
    fit_score: float

    def __post_init__(self) -> None:
        score = self.fit_score
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) \
                or not 0.0 <= score <= 1.0:
            raise ModelOutputError(f"FoundationPose fit score must be in [0, 1], got {score!r}")


class Detector(Protocol):
    def __call__(self, image: np.ndarray) -> list[RawBox]: ...


class WristClassifier(Protocol):
    def __call__(self, image: np.ndarray, bbox_px: tuple[float, float, float, float]) -> WristReading: ...


class PoseEstimator(Protocol):
    """FoundationPose against one STL, on one camera's RGB-D with a part mask (depth-image pixels)."""

    def __call__(self, object_id: str, camera: str, observation: Observation, mask: np.ndarray) -> PoseFit: ...


class CellGeometry(Protocol):
    """Calibration-backed geometry of the real cell (CalibratedCellGeometry wraps robot/cell/calibration + T28)."""

    table_surface_z_mm: float
    work_area: tuple[tuple[float, float], tuple[float, float]]   # UF0 x and y bounds (sdg/configs/cell_layout.yaml)

    def table_locate(self, observation: Observation, bbox_px: tuple[float, float, float, float]) -> Located: ...

    def table_mask(self, observation: Observation, bbox_px: tuple[float, float, float, float]) -> np.ndarray: ...

    def table_points(self, observation: Observation, bbox_px: tuple[float, float, float, float]) -> np.ndarray:
        """UF0 points (N, 3) mm of the part inside a table box: the same depth group table_locate uses."""
        ...

    def zone(self, x_mm: float, y_mm: float) -> tuple[Zone, str | None]: ...

    def table_cloud(self, observation: Observation) -> np.ndarray:
        """All table-depth UF0 points (N, 3) mm above the table-noise threshold."""
        ...

    def table_box(self, observation: Observation, points_uf0_mm: np.ndarray) -> tuple[float, float, float, float]:
        """Table colour-image box (observation pixels) around UF0 points, e.g. a posed CAD bbox."""
        ...

    def wrist_points(self, observation: Observation) -> tuple[np.ndarray, np.ndarray]:
        """UF0 points (N, 3) mm from the wrist depth map at the observation's joints, and their depth pixels (N, 2)."""
        ...

    def wrist_box(self, observation: Observation, points_uf0_mm: np.ndarray) -> tuple[float, float, float, float]: ...


class RtDetrDetector:
    """perception/cell/models.Detector (RT-DETR r50vd run dir, T27/T31 weights) on an observation image."""

    def __init__(self, run_dir: Path, min_score: float) -> None:
        from perception.cell.models import Detector as CellDetector

        self._model = CellDetector(run_dir, min_score)

    def __call__(self, image: np.ndarray) -> list[RawBox]:
        boxes = self._model(Image.fromarray(np.asarray(image, dtype=np.uint8)))
        return [RawBox(b.label, b.score, b.bbox_px) for b in boxes]


class DefectHeadClassifier:
    """T27 wrist ResNet-18 with the defect head (32 logits: 30 identity, then ok/defect), classifier_defect_v100."""

    def __init__(self, run_dir: Path, labels: tuple[str, ...], image_size: int) -> None:
        import json

        from perception.classifier import train

        if not hasattr(train, "_build_defect_model"):
            raise ModelLoadError("defect-head model code is on feature/T27-newset-training d869bad; merge it first")
        from safetensors.torch import load_file

        _build_defect_model, _build_transforms = train._build_defect_model, train._build_transforms
        weights = json.loads((run_dir / "defect_loss_weights.json").read_text())
        self._model = _build_defect_model(len(labels), (weights["ok"], weights["defect"]))
        self._model.load_state_dict(load_file(str(run_dir / "model.safetensors")))
        self._model.eval()
        self._labels = labels
        self._transform = _build_transforms(image_size)[1]

    def __call__(self, image: np.ndarray, bbox_px: tuple[float, float, float, float]) -> WristReading:
        import torch

        crop = classifier_crop(Image.fromarray(np.asarray(image, dtype=np.uint8)), bbox_px)
        batch = self._transform(crop.convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(pixel_values=batch).logits[0].float()
        identity = torch.softmax(logits[: len(self._labels)], dim=0)
        defect = torch.softmax(logits[len(self._labels):], dim=0)
        best = int(identity.argmax())
        return WristReading(self._labels[best], float(identity[best]), float(defect[1]))
