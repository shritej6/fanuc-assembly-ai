"""Live crops match the crops each wrist model was trained on.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import inspect

from PIL import Image

from perception.anomaly.testbed import CONTEXT
from perception.cell.models import CLASSIFIER_CONTEXT, classifier_crop, patchcore_crop
from perception.crops import export_crops


def test_classifier_crop_uses_the_training_context():
    assert inspect.signature(export_crops).parameters["context_scale"].default == CLASSIFIER_CONTEXT
    crop = classifier_crop(Image.new("RGB", (1280, 720)), (600.0, 300.0, 700.0, 340.0))
    assert crop.size == (130, 52)


def test_patchcore_crop_is_the_square_training_crop():
    assert CONTEXT == 1.3
    crop = patchcore_crop(Image.new("RGB", (1280, 720)), (600.0, 300.0, 700.0, 340.0))
    assert crop.size == (130, 130)


def test_crops_clip_at_the_image_edge():
    crop = classifier_crop(Image.new("RGB", (1280, 720)), (0.0, 0.0, 100.0, 40.0))
    assert crop.size == (115, 46)
