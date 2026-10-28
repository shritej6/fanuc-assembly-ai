import json
from pathlib import Path

import pytest
from PIL import Image

from perception.coco import ConversionError, export_coco
from perception.crops import export_crops


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _detector_record(image_path: Path, camera: str, bbox: list[int]) -> dict:
    return {
        "schema_version": 1,
        "task": "detector",
        "scene_id": "A:000001",
        "set": "A",
        "index": 1,
        "camera": camera,
        "split": "train",
        "variant": "native",
        "image_path": str(image_path),
        "label_path": "/labels/000001.json",
        "width": 64,
        "height": 48,
        "objects": [
            {
                "uid": "o00",
                "object_id": "F01",
                "class_id": 0,
                "bbox_xyxy": bbox,
                "kind": "base",
                "state": "on_table",
                "variant": "ok",
                "build_state": None,
            }
        ],
    }


def _classifier_record(image_path: Path, index: int, bbox: list[int]) -> dict:
    return {
        "schema_version": 1,
        "task": "classifier",
        "scene_id": f"B:{index:06d}",
        "set": "B",
        "index": index,
        "camera": "wrist",
        "split": "train",
        "variant": "native",
        "image_path": str(image_path),
        "label_path": f"/labels/{index:06d}.json",
        "width": 20,
        "height": 10,
        "target": {
            "uid": "o00",
            "object_id": "F01",
            "class_id": 0,
            "bbox_xyxy": bbox,
            "kind": "base",
            "state": "on_table",
            "variant": "ok",
            "build_state": None,
        },
    }


def test_coco_export_uses_xywh_area_and_unique_camera_image_ids(tmp_path: Path) -> None:
    table = tmp_path / "table.jpg"
    side = tmp_path / "side.jpg"
    Image.new("RGB", (64, 48), "white").save(table)
    Image.new("RGB", (64, 48), "white").save(side)
    manifest = _write_jsonl(
        tmp_path / "detector.jsonl",
        [
            _detector_record(table, "table", [10, 5, 30, 25]),
            _detector_record(side, "side", [1, 2, 11, 12]),
        ],
    )

    outputs = export_coco(manifest, tmp_path / "coco", labels=("F01", "foreign_screw_m3"))

    payload = json.loads(outputs["train"].read_text())
    assert [image["id"] for image in payload["images"]] == [1, 2]
    assert [image["camera"] for image in payload["images"]] == ["table", "side"]
    assert payload["annotations"][0] == {
        "id": 1,
        "image_id": 1,
        "category_id": 0,
        "bbox": [10, 5, 20, 20],
        "area": 400,
        "iscrowd": 0,
    }
    assert payload["categories"] == [
        {"id": 0, "name": "F01"},
        {"id": 1, "name": "foreign_screw_m3"},
    ]


def test_coco_export_refuses_zero_area_boxes(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    Image.new("RGB", (64, 48), "white").save(image)
    manifest = _write_jsonl(tmp_path / "detector.jsonl", [_detector_record(image, "table", [10, 5, 10, 25])])

    with pytest.raises(ConversionError, match="positive area"):
        export_coco(manifest, tmp_path / "coco", labels=("F01",))


def test_crop_export_clips_context_at_image_edges_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (20, 10), (10, 20, 30)).save(source)
    manifest = _write_jsonl(
        tmp_path / "classifier.jsonl",
        [
            _classifier_record(source, 1, [0, 0, 4, 4]),
            _classifier_record(source, 2, [16, 6, 20, 10]),
        ],
    )

    metadata_path = export_crops(manifest, tmp_path / "crops", context_scale=2.0)

    rows = [json.loads(line) for line in metadata_path.read_text().splitlines()]
    assert [row["crop_bbox_xyxy"] for row in rows] == [[0, 0, 6, 6], [14, 4, 20, 10]]
    assert [row["crop_path"] for row in rows] == [
        str((tmp_path / "crops/train/F01/B__000001__wrist__native.jpg").resolve()),
        str((tmp_path / "crops/train/F01/B__000002__wrist__native.jpg").resolve()),
    ]
    assert [Image.open(row["crop_path"]).size for row in rows] == [(6, 6), (6, 6)]

    second = export_crops(manifest, tmp_path / "crops_again", context_scale=2.0)
    second_rows = [json.loads(line) for line in second.read_text().splitlines()]
    for row in second_rows:
        row["crop_path"] = row["crop_path"].replace("crops_again", "crops")
    assert second_rows == rows
