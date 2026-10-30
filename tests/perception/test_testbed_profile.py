import json
from pathlib import Path

import pytest

from perception.coco import export_coco
from perception.manifest import ManifestConfig, ManifestError, build_manifests
from perception.rosie.gates import RosieGateError, verify_production_manifest
from perception.rosie.prepare import preparation_plan
from perception.taxonomy import load_ontology
from tests.perception.test_manifest import _annotation, _read_jsonl, _write_image, _write_taxonomy

REPO = Path(__file__).resolve().parents[2]


def _write_testbed_chunk(root: Path, set_name: str, n: int, rest: dict[str, str | None], cosmos: int = 0) -> Path:
    """Two cameras per scene; `rest` gives each object's state on the scene, as the test-bed labels store it."""
    chunk = root / "synthetic" / set_name / "chunk_0000"
    (chunk / "labels").mkdir(parents=True)
    ids = {"o00": "F01", "o01": "M1_01"}
    for index in range(n):
        stem = f"{index:06d}"
        objects = [{"uid": uid, "object_id": oid, "rest": rest.get(uid)} for uid, oid in ids.items()]
        cameras = {}
        for camera in ("table", "side"):
            _write_image(chunk / "rgb" / camera / f"{stem}.jpg")
            cameras[camera] = [_annotation("o00", "F01", [4, 5, 24, 25]), _annotation("o01", "M1_01", [28, 8, 50, 35])]
        spec = {"set": set_name, "index": index, "objects": objects, "wrist_target_uid": None}
        (chunk / "labels" / f"{stem}.json").write_text(json.dumps({"spec": spec, "cameras": cameras}))
    accepted = {}
    for index in range(cosmos):
        _write_image(chunk / "rgb_cosmos" / "table" / f"{index:06d}.jpg")
        accepted[f"table/{index:06d}"] = {"edge_fscore": 0.95, "accepted": True}
    (chunk / "_COSMOS.json").write_text(json.dumps(accepted))
    (chunk / "_DONE.json").write_text(json.dumps({"scenes": n, "code_commit": "abc1234"}))
    (chunk / "_QA.json").write_text(json.dumps({"ok": True, "labelled_scenes": n, "expected_scenes": n}))
    return chunk


def _config(root: Path, **overrides) -> ManifestConfig:
    values = {
        "data_root": root,
        "output_dir": root / "out",
        "taxonomy_path": _write_taxonomy(root),
        "detector_sets": ("N",),
        "classifier_sets": (),
        "require_cosmos": False,
        "include_cosmos": False,
    }
    return ManifestConfig(**{**values, **overrides})


def test_fallen_part_gets_its_fallen_class_and_installed_part_keeps_its_identity(tmp_path: Path) -> None:
    _write_testbed_chunk(tmp_path, "N", 2, {"o00": None, "o01": "fallen"})
    summary = build_manifests(_config(tmp_path, fallen_classes=True, profile="testbed"))
    labels = tuple(json.loads(summary.metadata_path.read_text())["labels"]["detector"])
    assert labels[-1] == "M1_01__fallen" and "F01__fallen" not in labels
    boxes = {o["uid"]: o for o in _read_jsonl(summary.detector_path)[0]["objects"]}
    assert (boxes["o01"]["label"], boxes["o01"]["rest"], labels[boxes["o01"]["class_id"]]) == (
        "M1_01__fallen", "fallen", "M1_01__fallen")
    assert labels[boxes["o00"]["class_id"]] == "F01"
    coco = json.loads(next(iter(export_coco(summary.detector_path, tmp_path / "coco", labels).values())).read_text())
    assert {c["name"] for c in coco["categories"]} >= {"M1_01", "M1_01__fallen"}


def test_without_fallen_classes_a_fallen_part_keeps_its_identity_class(tmp_path: Path) -> None:
    _write_testbed_chunk(tmp_path, "N", 1, {"o01": "fallen"})
    summary = build_manifests(_config(tmp_path))
    box = next(o for o in _read_jsonl(summary.detector_path)[0]["objects"] if o["uid"] == "o01")
    assert box["label"] == "M1_01"


def test_a_fallen_base_or_unknown_rest_state_is_rejected(tmp_path: Path) -> None:
    _write_testbed_chunk(tmp_path / "a", "N", 1, {"o00": "fallen"})
    with pytest.raises(ManifestError, match="base F01 has no fallen class"):
        build_manifests(_config(tmp_path / "a", fallen_classes=True))
    _write_testbed_chunk(tmp_path / "b", "N", 1, {"o01": "sideways"})
    with pytest.raises(ManifestError, match="unknown rest state"):
        build_manifests(_config(tmp_path / "b", fallen_classes=True))


def test_per_set_cosmos_range_replaces_the_default_only_for_that_set(tmp_path: Path) -> None:
    _write_testbed_chunk(tmp_path, "N", 10, {}, cosmos=5)            # 5 of 20 images = 0.25
    gated = {"require_cosmos": True, "include_cosmos": True}
    with pytest.raises(ManifestError, match="Cosmos coverage for N is 0.2500; expected 0.10-0.20"):
        build_manifests(_config(tmp_path, **gated))
    summary = build_manifests(_config(tmp_path, cosmos_ranges=(("N", 0.20, 0.30),), **gated))
    assert json.loads(summary.metadata_path.read_text())["sets"]["N"]["cosmos_accepted"] == 5


CHUNK_COUNTS = {"N": 40, "R": 20, "Q": 8, "O": 48}


def _metadata(
    tmp_path: Path,
    fractions: dict[str, float],
    chunks: dict[str, int] | None = None,
    **extra,
) -> Path:
    counts = {**CHUNK_COUNTS, **(chunks or {})}
    sets = {
        name: {
            "native_records": 1000,
            "cosmos_accepted": round(fraction * 1000),
            "chunks": [f"chunk_{index:04d}" for index in range(counts.get(name, 1))],
        }
        for name, fraction in fractions.items()
    }
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"schema_version": 2, "content_hash_algorithm": "sha256", "require_cosmos": True,
                                "include_cosmos": True, "sets": sets, **extra}))
    return path


def test_testbed_gate_uses_its_own_ranges_and_requires_the_fallen_classes(tmp_path: Path) -> None:
    good = {"N": 0.25, "R": 0.25, "Q": 0.25, "O": 0.50}
    verify_production_manifest(_metadata(tmp_path, good, profile="testbed", fallen_classes=True))
    with pytest.raises(RosieGateError, match="set O"):
        verify_production_manifest(_metadata(tmp_path, {**good, "O": 0.15}, profile="testbed", fallen_classes=True))
    with pytest.raises(RosieGateError, match="fallen classes"):
        verify_production_manifest(_metadata(tmp_path, good, profile="testbed"))
    with pytest.raises(RosieGateError, match="unknown manifest profile"):
        verify_production_manifest(_metadata(tmp_path, good, profile="other", fallen_classes=True))


def test_testbed_plan_trains_only_the_new_sets_with_fallen_classes(tmp_path: Path) -> None:
    taxonomy = REPO / "contracts/taxonomy.json"
    full = preparation_plan(tmp_path / "d", tmp_path / "o", taxonomy, False, "testbed")
    smoke = preparation_plan(tmp_path / "d", tmp_path / "o", taxonomy, True, "testbed")
    assert (full["detector_sets"], full["classifier_sets"], full["fallen_classes"]) == (["N", "R", "Q"], ["O"], True)
    assert (smoke["detector_sets"], smoke["classifier_sets"], smoke["require_cosmos"]) == (["N"], ["O"], False)
    assert len(load_ontology(taxonomy, fallen_classes=True).detector_labels) == 30 + 3 + 24
    with pytest.raises(ValueError, match="unknown profile"):
        preparation_plan(tmp_path / "d", tmp_path / "o", taxonomy, False, "old")


def test_testbed_gate_refuses_missing_chunks_and_extra_sets(tmp_path: Path) -> None:
    good = {"N": 0.25, "R": 0.25, "Q": 0.25, "O": 0.50}
    with pytest.raises(RosieGateError, match=r"set O is incomplete: 47 chunks, expected 48"):
        verify_production_manifest(
            _metadata(tmp_path, good, {"O": 47}, profile="testbed", fallen_classes=True)
        )
    with pytest.raises(RosieGateError, match="must hold exactly sets"):
        verify_production_manifest(
            _metadata(tmp_path, {**good, "A": 0.25}, profile="testbed", fallen_classes=True)
        )


def test_frozen_manifest_records_the_chunks_it_read(tmp_path: Path) -> None:
    _write_testbed_chunk(tmp_path, "N", 1, {"o00": None, "o01": "installed"})

    summary = build_manifests(_config(tmp_path, fallen_classes=True, profile="testbed"))

    assert json.loads(summary.metadata_path.read_text())["sets"]["N"]["chunks"] == ["chunk_0000"]


def test_production_trainers_read_only_the_complete_testbed_profile(tmp_path: Path) -> None:
    from perception.classifier.train import ClassifierTrainingError
    from perception.classifier.train import _validate_manifest_mode as classifier_gate
    from perception.detector.train import DetectorTrainError
    from perception.detector.train import _validate_manifest_mode as detector_gate

    metadata_path = _metadata(
        tmp_path,
        {"N": 0.25, "R": 0.25, "Q": 0.25, "O": 0.50},
        profile="testbed",
        fallen_classes=True,
    )
    metadata = json.loads(metadata_path.read_text())
    detector_gate(metadata, smoke=False)
    classifier_gate(metadata, smoke=False)
    with pytest.raises(DetectorTrainError, match="requires a 'testbed' manifest"):
        detector_gate({**metadata, "profile": "legacy"}, smoke=False)
    with pytest.raises(ClassifierTrainingError, match="requires a 'testbed' manifest"):
        classifier_gate({**metadata, "profile": "legacy"}, smoke=False)
    detector_gate({**metadata, "profile": "legacy", "require_cosmos": False}, smoke=True)


def test_loose_part_requires_rest_but_foreign_objects_and_bases_do_not(tmp_path: Path) -> None:
    from perception.manifest_annotations import annotation

    label2id = {"foreign_cube_10mm": 0, "F01": 1, "M1_01": 2, "M1_01__fallen": 3}
    for object_id in ("foreign_cube_10mm", "F01"):
        raw = {
            "uid": "o05",
            "object_id": object_id,
            "state": "on_table",
            "bbox_xyxy": [1, 1, 9, 9],
        }
        found = annotation(raw, label2id, 64, 48, "N:000000/side", fallen_classes=True)
        assert (found.label, found.rest) == (object_id, None)

    with pytest.raises(ManifestError, match="part M1_01 on_table has no rest state"):
        annotation(
            {"uid": "o06", "object_id": "M1_01", "state": "on_table", "bbox_xyxy": [1, 1, 9, 9]},
            label2id,
            64,
            48,
            "N:000000/side",
            fallen_classes=True,
        )
