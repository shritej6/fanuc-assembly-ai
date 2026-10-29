import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from perception.coco import ConversionError, export_coco
from perception.crops import export_crops


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


def _detector_record(image_path: Path, camera: str, bbox: list[int]) -> dict:
    return {
        "schema_version": 2,
        "task": "detector",
        "scene_id": "A:000001",
        "set": "A",
        "index": 1,
        "camera": camera,
        "split": "train",
        "variant": "native",
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
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
        "schema_version": 2,
        "task": "classifier",
        "scene_id": f"B:{index:06d}",
        "set": "B",
        "index": index,
        "camera": "wrist",
        "split": "train",
        "variant": "native",
        "image_path": str(image_path),
        "image_sha256": _sha256(image_path),
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
    assert [image["image_sha256"] for image in payload["images"]] == [_sha256(table), _sha256(side)]
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


def test_coco_export_preserves_detector_scenario_dimensions(tmp_path: Path) -> None:
    image = tmp_path / "scenario.jpg"
    Image.new("RGB", (64, 48), "white").save(image)
    record = _detector_record(image, "table", [1, 2, 11, 12])
    record["objects"].extend(
        [
            {
                "uid": "o01",
                "object_id": "M1_01",
                "class_id": 1,
                "bbox_xyxy": [12, 4, 24, 20],
                "kind": "part",
                "state": "on_table",
                "variant": "crack",
                "build_state": "high",
            },
            {
                "uid": "o02",
                "object_id": "M2_01",
                "class_id": 2,
                "bbox_xyxy": [30, 5, 42, 21],
                "kind": "trap",
                "state": "on_table",
                "variant": "ok",
                "build_state": None,
            },
        ]
    )
    manifest = _write_jsonl(tmp_path / "detector.jsonl", [record])

    outputs = export_coco(manifest, tmp_path / "coco", labels=("F01", "M1_01", "M2_01"))

    exported = json.loads(outputs["train"].read_text())["images"][0]
    assert exported["products"] == [1, 2]
    assert exported["defect_types"] == ["crack"]
    assert exported["build_states"] == ["high"]
    assert exported["has_trap"] is True


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
    assert all(row["schema_version"] == 2 for row in rows)
    assert all(row["crop_sha256"] == _sha256(Path(row["crop_path"])) for row in rows)

    second = export_crops(manifest, tmp_path / "crops_again", context_scale=2.0)
    second_rows = [json.loads(line) for line in second.read_text().splitlines()]
    for row in second_rows:
        row["crop_path"] = row["crop_path"].replace("crops_again", "crops")
    assert second_rows == rows


def test_crop_export_rejects_source_changed_after_manifest(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    Image.new("RGB", (20, 10), "white").save(source)
    record = _classifier_record(source, 1, [2, 2, 10, 8])
    manifest = _write_jsonl(tmp_path / "classifier.jsonl", [record])
    Image.new("RGB", (20, 10), "black").save(source)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        export_crops(manifest, tmp_path / "crops")


@pytest.mark.parametrize("task", ["detector", "classifier"])
def test_converters_reject_records_without_image_digest(tmp_path: Path, task: str) -> None:
    image = tmp_path / "source.jpg"
    Image.new("RGB", (64, 48), "white").save(image)
    record = (
        _detector_record(image, "table", [1, 2, 11, 12])
        if task == "detector"
        else _classifier_record(image, 1, [1, 2, 11, 9])
    )
    record.pop("image_sha256")
    manifest = _write_jsonl(tmp_path / f"{task}.jsonl", [record])

    with pytest.raises(ValueError, match="image_sha256"):
        if task == "detector":
            export_coco(manifest, tmp_path / "coco", labels=("F01",))
        else:
            export_crops(manifest, tmp_path / "crops")
