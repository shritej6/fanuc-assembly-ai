"""Tests for anomaly source contract.
Authored by Shritej Gunda.
"""

import json
from pathlib import Path

import pytest

from perception.anomaly import source_contract
from perception.anomaly.source_contract import DatasetBoundaryError

PRODUCTION_O_TARGETS = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F06",
    "M1_01",
    "M1_02",
    "M1_03",
    "M1_04",
    "M2_01",
    "M2_02",
    "M2_03",
    "M2_04",
    "M3_01",
    "M3_02",
    "M3_03",
    "M3_04",
    "M4_01",
    "M4_02",
    "M4_03",
    "M4_04",
    "M5_01",
    "M5_02",
    "M5_03",
    "M5_04",
    "M6_01",
    "M6_02",
    "M6_03",
    "M6_04",
)


def _write_inventory(
    new_root: Path,
    old_root: Path,
    chunks: dict[str, int],
    labels_per_chunk: int,
    o_targets: tuple[str, ...],
) -> None:
    o_cursor = 0
    for set_name, chunk_count in chunks.items():
        root = new_root if set_name in ("O", "T") else old_root
        for chunk_index in range(chunk_count):
            labels = root / set_name / f"chunk_{chunk_index:04d}" / "labels"
            labels.mkdir(parents=True)
            first = chunk_index * labels_per_chunk
            for index in range(first, first + labels_per_chunk):
                payload = {}
                if set_name == "O":
                    payload = _o_label(index, o_targets[o_cursor])
                    o_cursor += 1
                (labels / f"{index:06d}.json").write_text(json.dumps(payload))


def _o_label(index: int, object_id: str) -> dict:
    return {
        "spec": {
            "set": "O",
            "index": index,
            "wrist_target_uid": "o00",
            "objects": [{"uid": "o00", "object_id": object_id}],
        }
    }


def _configure(monkeypatch) -> tuple[dict[str, int], dict[str, int]]:
    chunks = {"O": 2, "T": 1, "C": 1, "I": 1}
    o_counts = {"F01": 2, "M1_01": 2}
    monkeypatch.setattr(source_contract, "SOURCE_CHUNK_COUNTS", chunks)
    monkeypatch.setattr(source_contract, "SOURCE_LABELS_PER_CHUNK", 2)
    monkeypatch.setattr(source_contract, "SOURCE_O_TARGET_COUNTS", o_counts)
    return chunks, o_counts


def _canonical_excluded_base() -> tuple[dict, dict]:
    target = {
        "uid": "o00",
        "object_id": "F01",
        "kind": "base",
        "state": "on_table",
        "variant": "ok",
        "build_state": None,
    }
    return target, {**target, "bbox_xyxy": [1, 2, 10, 20]}


def _assert_invalid_exclusion(set_name: str, target: dict, box: dict) -> None:
    with pytest.raises(DatasetBoundaryError, match="invalid excluded O base target"):
        source_contract.exclude_o_base_target(set_name, target, box, Path("label.json"))


def test_production_o_target_counts_are_exact_and_unpatched() -> None:
    expected = {target: 800 for target in PRODUCTION_O_TARGETS}

    assert source_contract.SOURCE_O_TARGET_COUNTS == expected
    assert source_contract.O_BASE_TARGETS == frozenset(PRODUCTION_O_TARGETS[:6])


def test_canonical_o_base_target_is_excluded() -> None:
    target, box = _canonical_excluded_base()

    assert source_contract.exclude_o_base_target("O", target, box, Path("label.json"))


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("uid", None),
        ("uid", "fixture00"),
        ("uid", "o"),
        ("uid", "oA"),
        ("object_id", "F07"),
        ("state", "installed"),
        ("variant", "warp"),
        ("build_state", "complete"),
    ],
)
def test_excluded_o_base_rejects_noncanonical_target_fields(field: str, invalid_value: object) -> None:
    target, box = _canonical_excluded_base()
    target[field] = invalid_value
    box[field] = invalid_value

    _assert_invalid_exclusion("O", target, box)


@pytest.mark.parametrize("set_name", ["T", "C", "I"])
def test_excluded_o_base_rejects_wrong_set(set_name: str) -> None:
    target, box = _canonical_excluded_base()

    _assert_invalid_exclusion(set_name, target, box)


@pytest.mark.parametrize("kind", ["part", "trap", "fixture"])
def test_exclusion_predicate_ignores_non_base_kind(kind: str) -> None:
    target, box = _canonical_excluded_base()
    target["kind"] = kind
    box["kind"] = kind

    assert not source_contract.exclude_o_base_target("O", target, box, Path("label.json"))


@pytest.mark.parametrize(("forbidden_field", "value"), [("stable_pose", 0), ("rest", "installed")])
def test_excluded_o_base_rejects_forbidden_pose_fields(forbidden_field: str, value: object) -> None:
    target, box = _canonical_excluded_base()
    target[forbidden_field] = value

    _assert_invalid_exclusion("O", target, box)


@pytest.mark.parametrize(
    "bbox",
    [
        (1, 2, 10, 20),
        [1, 2, 10],
        [True, 2, 10, 20],
        [-1, 2, 10, 20],
        [1, 2, 1, 20],
        [1, 2, 10, 2],
    ],
)
def test_excluded_o_base_rejects_noncanonical_bbox(bbox: object) -> None:
    target, box = _canonical_excluded_base()
    box["bbox_xyxy"] = bbox

    _assert_invalid_exclusion("O", target, box)


@pytest.mark.parametrize("field", ["uid", "object_id", "kind", "state", "variant", "build_state"])
def test_excluded_o_base_requires_target_box_field_agreement(field: str) -> None:
    target, box = _canonical_excluded_base()
    box[field] = "mismatch"

    _assert_invalid_exclusion("O", target, box)


def test_source_inventory_records_exact_reviewed_o_targets(tmp_path: Path, monkeypatch) -> None:
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    chunks, o_counts = _configure(monkeypatch)
    _write_inventory(new_root, old_root, chunks, 2, ("F01", "F01", "M1_01", "M1_01"))

    inventory = source_contract.validate_patchcore_source_inventory(new_root, old_root)

    assert inventory == {
        "O": {"chunks": 2, "labels": 4, "target_counts": o_counts},
        "T": {"chunks": 1, "labels": 2},
        "C": {"chunks": 1, "labels": 2},
        "I": {"chunks": 1, "labels": 2},
    }


@pytest.mark.parametrize("missing", ["chunk", "label"])
def test_source_inventory_rejects_incomplete_copy(tmp_path: Path, monkeypatch, missing: str) -> None:
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    chunks, _ = _configure(monkeypatch)
    _write_inventory(new_root, old_root, chunks, 2, ("F01", "F01", "M1_01", "M1_01"))
    if missing == "chunk":
        (new_root / "O/chunk_0001").rename(new_root / "O/missing_chunk")
    else:
        (old_root / "C/chunk_0000/labels/000001.json").unlink()

    with pytest.raises(DatasetBoundaryError, match="inventory is incomplete"):
        source_contract.validate_patchcore_source_inventory(new_root, old_root)


def test_source_inventory_rejects_changed_o_target_counts(tmp_path: Path, monkeypatch) -> None:
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    chunks, _ = _configure(monkeypatch)
    _write_inventory(new_root, old_root, chunks, 2, ("F01", "F01", "F01", "M1_01"))

    with pytest.raises(DatasetBoundaryError, match="target inventory"):
        source_contract.validate_patchcore_source_inventory(new_root, old_root)
