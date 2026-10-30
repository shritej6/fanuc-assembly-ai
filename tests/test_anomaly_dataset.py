import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from perception.anomaly import source_contract
from perception.anomaly.dataset import (
    DatasetBoundaryError,
    discover_patchcore_samples,
    prepare_patchcore_folders,
)


def _o_index(split: str) -> int:
    def assigned(index: int) -> str:
        token = f"1337:O:{index}".encode()
        bucket = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") % 10_000
        return "train" if bucket < 8_000 else "validation" if bucket < 9_000 else "test"

    return next(index for index in range(10_000) if assigned(index) == split)


def _write_rest_sidecar(root: Path, set_name: str, index: str, rest: str) -> None:
    sidecar = root / set_name / "chunk_0000/_REST.json"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.loads(sidecar.read_text())
        if sidecar.exists()
        else {
            "rule": "installed = same up axis as inside its product (yaw free); anything else = fallen",
            "max_tilt_deg": 2.0,
            "images": {},
        }
    )
    payload["images"].setdefault(index, {})["o03"] = rest
    sidecar.write_text(json.dumps(payload))


def _label_payload(set_name: str, index: str, variant: str, object_id: str, rest: str | None) -> dict:
    target = {
        "uid": "o03",
        "object_id": object_id,
        "kind": "part",
        "variant": variant,
        "stable_pose": 2,
    }
    if rest is not None:
        target["rest"] = rest
    return {
        "spec": {
            "set": set_name,
            "index": int(index),
            "wrist_target_uid": "o03",
            "objects": [target],
        },
        "cameras": {
            "wrist": [
                {
                    "uid": "o03",
                    "object_id": object_id,
                    "variant": variant,
                    "bbox_xyxy": [10, 20, 50, 80],
                    "visible_px": 2400,
                }
            ]
        },
    }


def _write_scene(
    root: Path,
    set_name: str,
    index: str,
    variant: str,
    *,
    object_id: str = "M1_01",
    rest: str | None = None,
) -> Path:
    chunk = root / set_name / "chunk_0000"
    for relative in ("labels", "rgb/wrist", "depth/wrist", "seg/wrist"):
        (chunk / relative).mkdir(parents=True, exist_ok=True)
    rgb = chunk / "rgb/wrist" / f"{index}.jpg"
    Image.new("RGB", (100, 100), (20, 30, 40)).save(rgb)
    (chunk / "depth/wrist" / f"{index}.png").write_bytes(b"depth")
    (chunk / "seg/wrist" / f"{index}.png").write_bytes(b"seg")
    label_path = chunk / "labels" / f"{index}.json"
    label_rest = rest if set_name == "T" else None
    label_path.write_text(json.dumps(_label_payload(set_name, index, variant, object_id, label_rest)))
    if set_name in ("C", "I") and rest is not None:
        _write_rest_sidecar(root, set_name, index, rest)
    return rgb


def _write_inventory(new_root: Path, old_root: Path, chunks: dict[str, int], labels_per_chunk: int) -> None:
    for set_name, chunk_count in chunks.items():
        root = new_root if set_name in ("O", "T") else old_root
        for chunk_index in range(chunk_count):
            labels = root / set_name / f"chunk_{chunk_index:04d}" / "labels"
            labels.mkdir(parents=True)
            first = chunk_index * labels_per_chunk
            for index in range(first, first + labels_per_chunk):
                (labels / f"{index:06d}.json").write_text("{}")


def test_patchcore_discovery_keeps_defects_out_of_training(tmp_path: Path) -> None:
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    train_index = f"{_o_index('train'):06d}"
    good_rgb = _write_scene(new_root, "O", train_index, "ok")
    _write_scene(old_root, "I", "000001", "missing_feature", rest="installed")

    samples = discover_patchcore_samples(new_root, old_root=old_root, rest_root=old_root)

    assert [(item.set_name, item.variant, item.eligibility, item.assigned_split) for item in samples] == [
        ("O", "ok", "train_eligible", "train/good"),
        ("I", "missing_feature", "evaluation_only", "test/bad"),
    ]
    assert samples[0].object_id == "M1_01"
    assert samples[0].stable_pose == 2
    assert samples[0].bbox_xyxy == (10, 20, 50, 80)
    assert samples[0].segmentation_value == 4
    assert samples[0].rgb_sha256 == hashlib.sha256(good_rgb.read_bytes()).hexdigest()
    assert samples[0].label_sha256 == hashlib.sha256(samples[0].label_path.read_bytes()).hexdigest()
    assert samples[0].depth_sha256 == hashlib.sha256(samples[0].depth_path.read_bytes()).hexdigest()
    assert samples[0].segmentation_sha256 == hashlib.sha256(samples[0].segmentation_path.read_bytes()).hexdigest()


def test_only_o_good_training_split_can_enter_patchcore_training(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    train_index = f"{_o_index('train'):06d}"
    test_index = f"{_o_index('test'):06d}"
    validation_index = f"{_o_index('validation'):06d}"
    _write_scene(new_root, "O", train_index, "ok")
    _write_scene(new_root, "O", test_index, "ok")
    _write_scene(new_root, "O", validation_index, "ok")
    _write_scene(old_root, "C", "000100", "ok")
    _write_rest_sidecar(old_root, "C", "000100", "installed")
    _write_scene(old_root, "I", "000101", "warp")
    _write_rest_sidecar(old_root, "I", "000101", "installed")

    samples = discover_patchcore_samples(new_root, old_root=old_root, rest_root=old_root)

    assert [(sample.sample_id, sample.eligibility, sample.assigned_split) for sample in samples] == [
        (f"O/chunk_0000/{train_index}", "train_eligible", "train/good"),
        (f"O/chunk_0000/{test_index}", "false_alarm_only", "o_test_good"),
        ("C/chunk_0000/000100", "evaluation_only", "test/good"),
        ("I/chunk_0000/000101", "evaluation_only", "test/bad"),
    ]


def test_ci_rest_sidecar_is_required_and_fallen_samples_are_excluded(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    _write_scene(old_root, "C", "000100", "ok")

    with pytest.raises(DatasetBoundaryError, match="_REST.json"):
        discover_patchcore_samples(new_root, old_root=old_root, rest_root=old_root)

    _write_rest_sidecar(old_root, "C", "000100", "fallen")
    _write_scene(new_root, "T", "000200", "ok", object_id="M1_04", rest="installed")
    samples = discover_patchcore_samples(new_root, old_root=old_root, rest_root=old_root)

    assert [sample.sample_id for sample in samples] == ["T/chunk_0000/000200"]


@pytest.mark.parametrize(
    ("field", "value"),
    [("rule", "unreviewed"), ("max_tilt_deg", 90.0)],
)
def test_ci_rest_sidecar_requires_reviewed_metadata(tmp_path: Path, field: str, value: str | float) -> None:
    old_root = tmp_path / "old"
    _write_scene(old_root, "C", "000100", "ok", rest="installed")
    sidecar = old_root / "C/chunk_0000/_REST.json"
    payload = json.loads(sidecar.read_text())
    payload[field] = value
    sidecar.write_text(json.dumps(payload))

    with pytest.raises(DatasetBoundaryError, match="_REST.json provenance"):
        discover_patchcore_samples(tmp_path / "new", old_root=old_root, rest_root=old_root)


def test_ci_rest_sidecar_rejects_malformed_json(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    _write_scene(old_root, "C", "000100", "ok", rest="installed")
    (old_root / "C/chunk_0000/_REST.json").write_text("not json")

    with pytest.raises(DatasetBoundaryError, match="cannot read label"):
        discover_patchcore_samples(tmp_path / "new", old_root=old_root, rest_root=old_root)


def test_discovery_rejects_unreviewed_defect_variant(tmp_path: Path) -> None:
    old_root = tmp_path / "old"
    _write_scene(old_root, "C", "000100", "typo_defect", rest="installed")

    with pytest.raises(DatasetBoundaryError, match="unreviewed defect variant"):
        discover_patchcore_samples(tmp_path / "new", old_root=old_root, rest_root=old_root)


def test_discovery_rejects_changed_defect_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"defects": [], "variants": {}}')

    with pytest.raises(DatasetBoundaryError, match="manifest is missing or changed"):
        discover_patchcore_samples(
            tmp_path / "new",
            old_root=tmp_path / "old",
            rest_root=tmp_path / "rest",
            defect_manifest=manifest,
        )


def test_ci_fallen_artifacts_are_validated_before_pose_filter(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    _write_scene(old_root, "C", "000100", "ok")
    _write_rest_sidecar(old_root, "C", "000100", "fallen")
    (old_root / "C/chunk_0000/depth/wrist/000100.png").unlink()

    with pytest.raises(DatasetBoundaryError, match="required anomaly artifact is missing"):
        discover_patchcore_samples(new_root, old_root=old_root, rest_root=old_root)


def test_t_discovery_accepts_only_installed_rear_guards_as_evaluation_only(tmp_path: Path) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest="installed")
    _write_scene(tmp_path, "T", "000001", "warp", object_id="M2_04", rest="installed")
    _write_scene(tmp_path, "T", "000002", "ok", object_id="M3_04", rest="installed")
    _write_scene(tmp_path, "T", "000003", "ok", object_id="M3_04", rest="fallen")
    _write_scene(tmp_path, "T", "000004", "ok", object_id="M4_04", rest="installed")

    samples = discover_patchcore_samples(tmp_path, old_root=tmp_path / "old", rest_root=tmp_path / "rest")

    assert [
        (sample.sample_id, sample.object_id, sample.variant, sample.eligibility, sample.rest) for sample in samples
    ] == [
        ("T/chunk_0000/000000", "M1_04", "ok", "evaluation_only", "installed"),
        ("T/chunk_0000/000001", "M2_04", "warp", "evaluation_only", "installed"),
        ("T/chunk_0000/000002", "M3_04", "ok", "evaluation_only", "installed"),
    ]


def test_discovery_reads_ci_rest_from_a_separate_derived_root(tmp_path: Path) -> None:
    new_root = tmp_path / "new"
    old_root = tmp_path / "old"
    rest_root = tmp_path / "derived-rest"
    _write_scene(old_root, "C", "000000", "ok", object_id="M1_04")
    _write_rest_sidecar(rest_root, "C", "000000", "installed")
    _write_scene(new_root, "T", "000100", "warp", object_id="M1_04", rest="installed")

    samples = discover_patchcore_samples(new_root, old_root=old_root, rest_root=rest_root)

    assert [(sample.sample_id, sample.eligibility) for sample in samples] == [
        ("C/chunk_0000/000000", "evaluation_only"),
        ("T/chunk_0000/000100", "evaluation_only"),
    ]


def test_t_evaluation_samples_never_enter_patchcore_training(tmp_path: Path) -> None:
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    _write_scene(new_root, "O", f"{_o_index('train'):06d}", "ok", object_id="M1_04")
    _write_scene(new_root, "O", f"{_o_index('test'):06d}", "ok", object_id="M1_04")
    _write_scene(old_root, "C", "000005", "ok", object_id="M1_04", rest="installed")
    _write_scene(old_root, "I", "000006", "missing_feature", object_id="M1_04", rest="installed")
    _write_scene(new_root, "T", "000100", "ok", object_id="M1_04", rest="installed")
    _write_scene(new_root, "T", "000101", "warp", object_id="M1_04", rest="installed")

    samples = discover_patchcore_samples(new_root, old_root=old_root, rest_root=old_root)
    output = tmp_path / "prepared"
    summary = prepare_patchcore_folders(
        samples,
        output,
        expected_object_ids=("M1_04",),
    )

    assert (summary.train_good, summary.test_good, summary.test_bad, summary.o_test_good) == (1, 2, 2, 1)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
    t_rows = [row for row in rows if row["sample_id"].startswith("T/")]
    assert [(row["variant"], row["rest"], row["split"]) for row in t_rows] == [
        ("ok", "installed", "test/good"),
        ("warp", "installed", "test/bad"),
    ]
    assert not list((output / "M1_04/train/good").glob("T__*.png"))


@pytest.mark.parametrize("rest", ["installed", "fallen"])
def test_t_discovery_validates_artifacts_before_pose_filter(tmp_path: Path, rest: str) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest=rest)
    (tmp_path / "T/chunk_0000/depth/wrist/000000.png").unlink()

    with pytest.raises(DatasetBoundaryError, match="required anomaly artifact is missing"):
        discover_patchcore_samples(tmp_path, old_root=tmp_path / "old", rest_root=tmp_path / "rest")


@pytest.mark.parametrize("rest", [None, "sideways"])
def test_t_discovery_rejects_missing_or_unknown_rest_provenance(tmp_path: Path, rest: str | None) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest=rest)

    with pytest.raises(DatasetBoundaryError, match="rest provenance"):
        discover_patchcore_samples(tmp_path, old_root=tmp_path / "old", rest_root=tmp_path / "rest")


def test_t_discovery_rejects_label_path_provenance_mismatch(tmp_path: Path) -> None:
    _write_scene(tmp_path, "T", "000000", "ok", object_id="M1_04", rest="installed")
    label_path = tmp_path / "T/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["spec"]["set"] = "S"
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="provenance"):
        discover_patchcore_samples(tmp_path, old_root=tmp_path / "old", rest_root=tmp_path / "rest")


def test_patchcore_discovery_requires_visible_wrist_target(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok", rest="installed")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["cameras"]["wrist"] = []
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="visible wrist target"):
        discover_patchcore_samples(tmp_path / "new", old_root=tmp_path, rest_root=tmp_path)


@pytest.mark.parametrize(("field", "value"), [("set", "REAL"), ("index", 99)])
def test_patchcore_discovery_rejects_label_provenance_mismatch(tmp_path: Path, field: str, value: str | int) -> None:
    _write_scene(tmp_path, "C", "000000", "ok", rest="installed")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["spec"][field] = value
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="provenance"):
        discover_patchcore_samples(tmp_path / "new", old_root=tmp_path, rest_root=tmp_path)


def test_patchcore_discovery_rejects_noncanonical_chunk(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok", rest="installed")
    (tmp_path / "C/chunk_0000").rename(tmp_path / "C/chunk_rejected")

    with pytest.raises(DatasetBoundaryError, match="chunk name"):
        discover_patchcore_samples(tmp_path / "new", old_root=tmp_path, rest_root=tmp_path)


def test_patchcore_discovery_rejects_invalid_bbox(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok", rest="installed")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["cameras"]["wrist"][0]["bbox_xyxy"] = [5, 5, 1, 1]
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="metadata"):
        discover_patchcore_samples(tmp_path / "new", old_root=tmp_path, rest_root=tmp_path)


def test_patchcore_discovery_wraps_malformed_target(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok", rest="installed")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["spec"]["objects"] = [None]
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="invalid wrist target"):
        discover_patchcore_samples(tmp_path / "new", old_root=tmp_path, rest_root=tmp_path)


def test_patchcore_discovery_wraps_invalid_visible_pixel_count(tmp_path: Path) -> None:
    _write_scene(tmp_path, "C", "000000", "ok", rest="installed")
    label_path = tmp_path / "C/chunk_0000/labels/000000.json"
    label = json.loads(label_path.read_text())
    label["cameras"]["wrist"][0]["visible_px"] = "bad"
    label_path.write_text(json.dumps(label))

    with pytest.raises(DatasetBoundaryError, match="visible wrist target"):
        discover_patchcore_samples(tmp_path / "new", old_root=tmp_path, rest_root=tmp_path)


def test_patchcore_discovery_rejects_symlinked_set(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_scene(outside, "C", "000000", "ok", rest="installed")
    synthetic = tmp_path / "synthetic"
    synthetic.mkdir()
    (synthetic / "C").symlink_to(outside / "C", target_is_directory=True)

    with pytest.raises(DatasetBoundaryError, match="symlink"):
        discover_patchcore_samples(tmp_path / "new", old_root=synthetic, rest_root=synthetic)


def test_patchcore_discovery_rejects_symlinked_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_scene(outside, "C", "000000", "ok", rest="installed")
    synthetic = tmp_path / "synthetic"
    synthetic.symlink_to(outside, target_is_directory=True)

    with pytest.raises(DatasetBoundaryError, match="root.*symlink"):
        discover_patchcore_samples(tmp_path / "new", old_root=synthetic, rest_root=synthetic)


def test_source_inventory_requires_exact_reviewed_chunks_and_labels(tmp_path: Path, monkeypatch) -> None:
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    chunks = {"O": 2, "T": 1, "C": 1, "I": 1}
    monkeypatch.setattr(source_contract, "SOURCE_CHUNK_COUNTS", chunks)
    monkeypatch.setattr(source_contract, "SOURCE_LABELS_PER_CHUNK", 2)
    _write_inventory(new_root, old_root, chunks, 2)

    inventory = source_contract.validate_patchcore_source_inventory(new_root, old_root)

    assert inventory == {
        "O": {"chunks": 2, "labels": 4},
        "T": {"chunks": 1, "labels": 2},
        "C": {"chunks": 1, "labels": 2},
        "I": {"chunks": 1, "labels": 2},
    }


@pytest.mark.parametrize("missing", ["chunk", "label"])
def test_source_inventory_rejects_incomplete_copy(tmp_path: Path, monkeypatch, missing: str) -> None:
    new_root, old_root = tmp_path / "new", tmp_path / "old"
    chunks = {"O": 2, "T": 1, "C": 1, "I": 1}
    monkeypatch.setattr(source_contract, "SOURCE_CHUNK_COUNTS", chunks)
    monkeypatch.setattr(source_contract, "SOURCE_LABELS_PER_CHUNK", 2)
    _write_inventory(new_root, old_root, chunks, 2)
    if missing == "chunk":
        (new_root / "O/chunk_0001").rename(new_root / "O/missing_chunk")
    else:
        (old_root / "C/chunk_0000/labels/000001.json").unlink()

    with pytest.raises(DatasetBoundaryError, match="inventory is incomplete"):
        source_contract.validate_patchcore_source_inventory(new_root, old_root)
