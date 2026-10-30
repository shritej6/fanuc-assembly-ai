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


CHUNK_COUNTS = {"N": 40, "R": 20, "Q": 8, "O": 48, "P": 24}


def _metadata(tmp_path: Path, fractions: dict[str, float], chunks: dict[str, int] | None = None, **extra) -> Path:
    counts = {**CHUNK_COUNTS, **(chunks or {})}
    sets = {
        name: {"native_records": 1000, "cosmos_accepted": round(f * 1000),
               "chunks": [f"chunk_{i:04d}" for i in range(counts.get(name, 1))]}
        for name, f in fractions.items()
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


def _write_build_chunk(root: Path, states: list[str]) -> None:
    chunk = root / "synthetic" / "P" / "chunk_0000"
    (chunk / "labels").mkdir(parents=True)
    for index, state in enumerate(states):
        stem = f"{index:06d}"
        _write_image(chunk / "rgb" / "wrist" / f"{stem}.jpg")
        box = {**_annotation("o01", "M1_01", [8, 6, 40, 30]), "state": "installed", "build_state": state}
        spec = {"set": "P", "index": index, "objects": [{"uid": "o01", "object_id": "M1_01"}], "wrist_target_uid": "o01"}
        (chunk / "labels" / f"{stem}.json").write_text(json.dumps({"spec": spec, "cameras": {"wrist": [box]}}))
    (chunk / "_DONE.json").write_text(json.dumps({"scenes": len(states), "code_commit": "abc1234"}))
    (chunk / "_QA.json").write_text(json.dumps({"ok": True, "labelled_scenes": len(states),
                                                "expected_scenes": len(states)}))


def test_set_p_targets_are_classified_by_build_state_and_keep_the_exact_part(tmp_path: Path) -> None:
    from perception.classifier.dataset import CropDataset
    from perception.crops import export_crops
    from perception.taxonomy import BUILD_STATES

    _write_build_chunk(tmp_path, ["rotated", "seated"])
    summary = build_manifests(_config(tmp_path, detector_sets=(), classifier_sets=("P",),
                                      classifier_target="build_state", profile="testbed_build"))
    metadata = json.loads(summary.metadata_path.read_text())
    assert metadata["labels"]["classifier"] == list(BUILD_STATES) and metadata["labels"]["lookalike_groups"] == []
    target = _read_jsonl(summary.classifier_path)[0]["target"]
    assert (target["object_id"], target["label"], target["class_id"]) == ("M1_01", "rotated", 2)
    rows = [json.loads(line) for line in export_crops(summary.classifier_path, tmp_path / "crops").read_text().splitlines()]
    assert {row["label"] for row in rows} == {"rotated", "seated"}
    for split in {row["split"] for row in rows}:
        dataset = CropDataset(tmp_path / "crops" / "crops.jsonl", split, lambda image: image, BUILD_STATES)
        assert all(dataset[i]["labels"] == BUILD_STATES.index(dataset.rows[i]["label"]) for i in range(len(dataset)))


def test_unknown_build_state_is_rejected(tmp_path: Path) -> None:
    _write_build_chunk(tmp_path, ["wobbly"])
    with pytest.raises(ManifestError, match="unknown build_state"):
        build_manifests(_config(tmp_path, detector_sets=(), classifier_sets=("P",), classifier_target="build_state"))


def test_build_state_plan_and_gate_cover_only_set_p(tmp_path: Path) -> None:
    plan = preparation_plan(tmp_path / "d", tmp_path / "o", REPO / "contracts/taxonomy.json", False, "testbed_build")
    assert (plan["detector_sets"], plan["classifier_sets"], plan["classifier_target"]) == ([], ["P"], "build_state")
    verify_production_manifest(_metadata(tmp_path, {"P": 0.5}, profile="testbed_build",
                                         classifier_target="build_state"))
    with pytest.raises(RosieGateError, match="build states"):
        verify_production_manifest(_metadata(tmp_path, {"P": 0.5}, profile="testbed_build"))


def test_build_state_trainer_requires_six_labels_in_their_frozen_order() -> None:
    from perception.classifier.train import ClassifierTrainingError, validate_classifier_identity
    from perception.taxonomy import BUILD_STATES

    config = {"task": "classifier", "model": {"num_labels": 6}}
    metadata = {"classifier_target": "build_state",
                "labels": {"classifier": list(BUILD_STATES), "lookalike_groups": []}}
    assert validate_classifier_identity(config, metadata) == (BUILD_STATES, ())
    with pytest.raises(ClassifierTrainingError, match="6 build-state labels"):
        validate_classifier_identity({**config, "model": {"num_labels": 30}}, metadata)
    shuffled = {**metadata, "labels": {"classifier": list(reversed(BUILD_STATES)), "lookalike_groups": []}}
    with pytest.raises(ClassifierTrainingError, match="exactly"):
        validate_classifier_identity(config, shuffled)


def test_build_state_configs_load_and_other_label_counts_stay_rejected(tmp_path: Path) -> None:
    import yaml

    from perception.training_config import ConfigError, load_training_config

    for hw in ("t4", "h100"):
        assert load_training_config(REPO / f"perception/configs/classifier_build_{hw}.yaml")["model"]["num_labels"] == 6
    bad = yaml.safe_load((REPO / "perception/configs/classifier_build_t4.yaml").read_text())
    bad["model"]["num_labels"] = 7
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ConfigError, match="30 part labels or the 6"):
        load_training_config(tmp_path / "bad.yaml")


def test_testbed_gates_refuse_a_set_with_a_missing_chunk_or_an_extra_set(tmp_path: Path) -> None:
    good = {"N": 0.25, "R": 0.25, "Q": 0.25, "O": 0.50}
    with pytest.raises(RosieGateError, match=r"set O is incomplete: 47 chunks, expected 48; missing \['chunk_0047'\]"):
        verify_production_manifest(_metadata(tmp_path, good, {"O": 47}, profile="testbed", fallen_classes=True))
    with pytest.raises(RosieGateError, match="set P is incomplete: 14 chunks, expected 24"):
        verify_production_manifest(_metadata(tmp_path, {"P": 0.5}, {"P": 14}, profile="testbed_build",
                                             classifier_target="build_state"))
    with pytest.raises(RosieGateError, match="must hold exactly sets"):
        verify_production_manifest(_metadata(tmp_path, {**good, "A": 0.25}, profile="testbed", fallen_classes=True))


def test_frozen_manifest_records_the_chunks_it_read(tmp_path: Path) -> None:
    _write_build_chunk(tmp_path, ["seated"])
    summary = build_manifests(_config(tmp_path, detector_sets=(), classifier_sets=("P",),
                                      classifier_target="build_state", profile="testbed_build"))
    assert json.loads(summary.metadata_path.read_text())["sets"]["P"]["chunks"] == ["chunk_0000"]


def _training_metadata(profile: str, chunks: dict[str, int] | None = None, **extra) -> dict:
    fractions = {"testbed": {"N": 0.25, "R": 0.25, "Q": 0.25, "O": 0.50}, "testbed_build": {"P": 0.5}}.get(profile, {"A": 0.15})
    counts = {**CHUNK_COUNTS, **(chunks or {})}
    sets = {
        name: {"native_records": 1000, "cosmos_accepted": round(f * 1000),
               "chunks": [f"chunk_{i:04d}" for i in range(counts.get(name, 1))]}
        for name, f in fractions.items()
    }
    return {"schema_version": 2, "content_hash_algorithm": "sha256", "require_cosmos": True, "include_cosmos": True,
            "profile": profile, "sets": sets, **extra}


def test_production_trainers_read_only_their_approved_complete_profile() -> None:
    from perception.classifier.train import ClassifierTrainingError
    from perception.classifier.train import _validate_manifest_mode as classifier_gate
    from perception.detector.train import DetectorTrainError
    from perception.detector.train import _validate_manifest_mode as detector_gate

    main = _training_metadata("testbed", fallen_classes=True)
    build = _training_metadata("testbed_build", classifier_target="build_state")
    detector_gate(main, smoke=False)
    classifier_gate(main, smoke=False)
    classifier_gate(build, smoke=False)
    with pytest.raises(DetectorTrainError, match="requires a 'testbed or testbed_weak' manifest, got profile 'legacy'"):
        detector_gate(_training_metadata("legacy"), smoke=False)
    with pytest.raises(DetectorTrainError, match="set O is incomplete"):
        detector_gate(_training_metadata("testbed", {"O": 47}, fallen_classes=True), smoke=False)
    with pytest.raises(ClassifierTrainingError, match="requires a 'testbed_build' manifest"):
        classifier_gate({**main, "classifier_target": "build_state"}, smoke=False)
    with pytest.raises(ClassifierTrainingError, match="set P is incomplete: 14 chunks"):
        classifier_gate(_training_metadata("testbed_build", {"P": 14}, classifier_target="build_state"), smoke=False)
    detector_gate({**_training_metadata("legacy"), "require_cosmos": False}, smoke=True)   # smoke stays exempt


def test_a_loose_part_without_a_rest_state_is_refused_not_trained_as_installed(tmp_path: Path) -> None:
    _write_testbed_chunk(tmp_path, "N", 1, {"o00": None, "o01": None})

    with pytest.raises(ManifestError, match="part M1_01 on_table has no rest state"):
        build_manifests(_config(tmp_path, fallen_classes=True, profile="testbed"))


def test_foreign_objects_and_bases_need_no_rest_state() -> None:
    """Regression: the rest-state check refused `foreign_cube_10mm on_table` and stopped T27 prepare on the box."""
    from perception.manifest_annotations import annotation

    label2id = {"foreign_cube_10mm": 0, "F01": 1, "M1_01": 2, "M1_01__fallen": 3}
    for object_id in ("foreign_cube_10mm", "F01"):
        raw = {"uid": "o05", "object_id": object_id, "state": "on_table", "bbox_xyxy": [1, 1, 9, 9]}
        found = annotation(raw, label2id, 64, 48, "N:000000/side", rest=None, fallen_classes=True)
        assert (found.label, found.rest) == (object_id, None)
    with pytest.raises(ManifestError, match="has no rest state"):
        annotation({"uid": "o06", "object_id": "M1_01", "state": "on_table", "bbox_xyxy": [1, 1, 9, 9]},
                   label2id, 64, 48, "N:000000/side", rest=None, fallen_classes=True)


def test_weak_profile_adds_s_and_t_without_cosmos():
    from perception.rosie.gates import TESTBED_WEAK_CHUNKS, TESTBED_WEAK_RANGES
    from perception.rosie.prepare import preparation_plan

    plan = preparation_plan(Path("data"), Path("out"), Path("contracts/taxonomy.json"), smoke=False,
                            profile="testbed_weak")
    assert plan["detector_sets"] == ["N", "R", "Q", "S"] and plan["classifier_sets"] == ["O", "T"]
    assert plan["fallen_classes"] is True
    assert TESTBED_WEAK_RANGES["S"] == TESTBED_WEAK_RANGES["T"] == (0.0, 0.0)
    assert TESTBED_WEAK_CHUNKS == {"N": 40, "R": 20, "Q": 8, "O": 48, "S": 6, "T": 3}


def test_build_state_training_still_refuses_the_weak_profile():
    import pytest

    from perception.rosie.gates import RosieGateError, verify_training_metadata

    with pytest.raises(RosieGateError, match="testbed_build"):
        verify_training_metadata({"profile": "testbed_weak"}, "build_state")


def test_finetune_config_pins_its_init_weights():
    import pytest

    from perception.training_config import ConfigError, load_training_config

    config = load_training_config(Path("perception/configs/rtdetr_a100_finetune.yaml"))
    assert config["model"]["init_from"] == "runs/detector_init" and len(config["model"]["init_sha256"]) == 64
    unpinned = Path("perception/configs/rtdetr_a100_finetune.yaml").read_text().replace("  init_sha256:", "  x_sha256:")
    tmp = Path(pytest.importorskip("tempfile").mkdtemp()) / "c.yaml"
    tmp.write_text(unpinned)
    with pytest.raises(ConfigError, match="init_sha256"):
        load_training_config(tmp)
