import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from perception.manifest import ManifestConfig, ManifestError, build_manifests, split_for_scene


def _write_taxonomy(root: Path) -> Path:
    path = root / "taxonomy.json"
    path.write_text(
        json.dumps(
            {
                "objects": [
                    {"object_id": "F01", "lookalikes": []},
                    {"object_id": "M1_01", "lookalikes": []},
                ]
            }
        )
    )
    return path


def _annotation(uid: str, object_id: str, bbox: list[int]) -> dict:
    return {
        "uid": uid,
        "object_id": object_id,
        "kind": "base" if object_id.startswith("F") else "part",
        "state": "on_table",
        "variant": "ok",
        "build_state": None,
        "bbox_xyxy": bbox,
        "visible_px": 100,
        "occlusion": None,
    }


def _write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), (220, 220, 220)).save(path)


def _write_chunk(
    root: Path,
    set_name: str,
    indices: list[int],
    cameras: tuple[str, ...],
    *,
    missing_target: set[int] | None = None,
) -> Path:
    chunk = root / "synthetic" / set_name / "chunk_0000"
    (chunk / "labels").mkdir(parents=True)
    missing_target = missing_target or set()
    for index in indices:
        stem = f"{index:06d}"
        objects = [
            {"uid": "o00", "object_id": "F01"},
            {"uid": "o01", "object_id": "M1_01"},
        ]
        camera_labels = {}
        for camera in cameras:
            _write_image(chunk / "rgb" / camera / f"{stem}.jpg")
            camera_labels[camera] = [
                _annotation("o00", "F01", [4, 5, 24, 25]),
                _annotation("o01", "M1_01", [28, 8, 50, 35]),
            ]
            if index in missing_target:
                camera_labels[camera] = camera_labels[camera][1:]
        spec = {
            "set": set_name,
            "index": index,
            "objects": objects,
            "wrist_target_uid": "o00" if cameras == ("wrist",) else None,
        }
        (chunk / "labels" / f"{stem}.json").write_text(json.dumps({"spec": spec, "cameras": camera_labels}))
    (chunk / "_DONE.json").write_text(
        json.dumps({"scenes": len(indices), "images_this_run": len(indices) * len(cameras), "code_commit": "abc1234"})
    )
    (chunk / "_QA.json").write_text(
        json.dumps(
            {
                "ok": True,
                "labelled_scenes": len(indices),
                "expected_scenes": len(indices),
                "errors": 0,
            }
        )
    )
    return chunk


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_split_is_deterministic_with_literal_scene_buckets() -> None:
    assert split_for_scene("A", 1, seed=17) == "train"
    assert split_for_scene("A", 13, seed=17) == "validation"
    assert split_for_scene("A", 0, seed=17) == "test"
    assert split_for_scene("A", 1, seed=17) == split_for_scene("A", 1, seed=17)


def test_builder_groups_cameras_and_accepted_cosmos_with_the_native_scene(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    detector_chunk = _write_chunk(tmp_path, "A", [0, 1, 13], ("table", "side"))
    cosmos_path = detector_chunk / "rgb_cosmos" / "table" / "000001.jpg"
    _write_image(cosmos_path)
    Image.new("RGB", (64, 48), (10, 20, 30)).save(cosmos_path)
    detector_chunk.joinpath("_COSMOS.json").write_text(
        json.dumps(
            {
                "table/000001": {"edge_fscore": 0.91, "accepted": True},
                "side/000000": {"edge_fscore": 0.42, "accepted": False},
            }
        )
    )
    _write_chunk(tmp_path, "B", [1, 2], ("wrist",), missing_target={2})

    summary = build_manifests(
        ManifestConfig(
            data_root=tmp_path,
            output_dir=tmp_path / "manifests",
            taxonomy_path=taxonomy,
            detector_sets=("A",),
            classifier_sets=("B",),
            split_seed=17,
            require_cosmos=False,
        )
    )

    detector = _read_jsonl(summary.detector_path)
    classifier = _read_jsonl(summary.classifier_path)
    scene_one = [record for record in detector if record["scene_id"] == "A:000001"]
    assert len(detector) == 7
    assert {(record["camera"], record["variant"], record["split"]) for record in scene_one} == {
        ("table", "native", "train"),
        ("side", "native", "train"),
        ("table", "cosmos", "train"),
    }
    assert {record["split"] for record in detector if record["scene_id"] == "A:000000"} == {"test"}
    assert {record["split"] for record in detector if record["scene_id"] == "A:000013"} == {"validation"}
    assert len(classifier) == 1
    assert classifier[0]["target"]["object_id"] == "F01"
    assert classifier[0]["target"]["class_id"] == 0
    assert summary.skipped_classifier_targets == {"B": 1}
    assert summary.detector_sha256 == hashlib.sha256(summary.detector_path.read_bytes()).hexdigest()
    assert summary.classifier_sha256 == hashlib.sha256(summary.classifier_path.read_bytes()).hexdigest()
    metadata = json.loads(summary.metadata_path.read_text())
    assert metadata["labels"]["classifier"] == ["F01", "M1_01"]
    assert metadata["labels"]["detector"] == [
        "F01",
        "M1_01",
        "foreign_screw_m3",
        "foreign_tape_roll",
        "foreign_cube_10mm",
    ]
    assert metadata["schema_version"] == 2
    assert metadata["content_hash_algorithm"] == "sha256"
    for record in detector + classifier:
        assert record["schema_version"] == 2
        assert record["image_sha256"] == _sha256(Path(record["image_path"]))
    native = next(record for record in scene_one if record["camera"] == "table" and record["variant"] == "native")
    cosmos = next(record for record in scene_one if record["camera"] == "table" and record["variant"] == "cosmos")
    assert native["image_sha256"] != cosmos["image_sha256"]


def test_manifest_sha_changes_when_valid_same_size_image_bytes_change(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table",))
    first = build_manifests(
        ManifestConfig(
            data_root=tmp_path,
            output_dir=tmp_path / "first",
            taxonomy_path=taxonomy,
            detector_sets=("A",),
            classifier_sets=(),
            require_cosmos=False,
            include_cosmos=False,
        )
    )
    image_path = chunk / "rgb/table/000001.jpg"
    first_digest = _read_jsonl(first.detector_path)[0]["image_sha256"]

    Image.new("RGB", (64, 48), (10, 20, 30)).save(image_path)
    second = build_manifests(
        ManifestConfig(
            data_root=tmp_path,
            output_dir=tmp_path / "second",
            taxonomy_path=taxonomy,
            detector_sets=("A",),
            classifier_sets=(),
            require_cosmos=False,
            include_cosmos=False,
        )
    )

    assert first_digest != _sha256(image_path)
    assert _read_jsonl(second.detector_path)[0]["image_sha256"] == _sha256(image_path)
    assert first.detector_sha256 != second.detector_sha256


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing_done", "_DONE.json"),
        ("qa_not_ok", "QA failed"),
        ("count_mismatch", "scene count mismatch"),
    ],
)
def test_incomplete_or_inconsistent_chunks_are_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table", "side"))
    if mutation == "missing_done":
        chunk.joinpath("_DONE.json").unlink()
    elif mutation == "qa_not_ok":
        chunk.joinpath("_QA.json").write_text(json.dumps({"ok": False}))
    else:
        chunk.joinpath("_QA.json").write_text(
            json.dumps({"ok": True, "labelled_scenes": 1, "expected_scenes": 2})
        )

    with pytest.raises(ManifestError, match=message):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_forbidden_data_root_is_rejected_before_discovery(tmp_path: Path) -> None:
    root = tmp_path / "rejected"
    taxonomy = _write_taxonomy(tmp_path)

    with pytest.raises(ManifestError, match="forbidden"):
        build_manifests(
            ManifestConfig(
                data_root=root,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=(),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


@pytest.mark.parametrize("orphan", [False, True])
def test_cosmos_requires_acceptance_evidence_and_a_matching_file(tmp_path: Path, orphan: bool) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table", "side"))
    if orphan:
        _write_image(chunk / "rgb_cosmos" / "table" / "000001.jpg")
        chunk.joinpath("_COSMOS.json").write_text("{}")
        message = "orphan Cosmos"
    else:
        chunk.joinpath("_COSMOS.json").write_text(
            json.dumps({"table/000001": {"edge_fscore": 0.92, "accepted": True}})
        )
        message = "accepted Cosmos file is missing"

    with pytest.raises(ManifestError, match=message):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_production_manifest_enforces_per_set_cosmos_fraction(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    _write_chunk(tmp_path, "A", list(range(10)), ("table", "side"))

    with pytest.raises(ManifestError, match="Cosmos coverage"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=True,
            )
        )


def test_cosmos_fraction_uses_eligible_image_records_not_scene_count(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", list(range(10)), ("table", "side"))
    _write_image(chunk / "rgb_cosmos" / "table" / "000001.jpg")
    chunk.joinpath("_COSMOS.json").write_text(
        json.dumps({"table/000001": {"edge_fscore": 0.95, "accepted": True}})
    )

    with pytest.raises(ManifestError, match="Cosmos coverage"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=True,
            )
        )


def test_native_smoke_manifest_excludes_partial_cosmos_and_limits_scenes(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1, 2, 3], ("table", "side"))
    _write_image(chunk / "rgb_cosmos" / "table" / "000001.jpg")
    chunk.joinpath("_COSMOS.json").write_text(
        json.dumps({"table/000001": {"edge_fscore": 0.95, "accepted": True}})
    )

    summary = build_manifests(
        ManifestConfig(
            data_root=tmp_path,
            output_dir=tmp_path / "out",
            taxonomy_path=taxonomy,
            detector_sets=("A",),
            classifier_sets=(),
            require_cosmos=False,
            include_cosmos=False,
            max_scenes_per_set=2,
        )
    )

    records = _read_jsonl(summary.detector_path)
    metadata = json.loads(summary.metadata_path.read_text())
    assert len(records) == 4
    assert {record["variant"] for record in records} == {"native"}
    assert metadata["include_cosmos"] is False
    assert metadata["max_scenes_per_set"] == 2


@pytest.mark.parametrize(
    ("include_cosmos", "max_scenes", "message"),
    [(False, None, "include Cosmos"), (True, 0, "max_scenes_per_set")],
)
def test_manifest_rejects_inconsistent_mode_configuration(
    tmp_path: Path,
    include_cosmos: bool,
    max_scenes: int | None,
    message: str,
) -> None:
    taxonomy = _write_taxonomy(tmp_path)

    with pytest.raises(ManifestError, match=message):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=(),
                classifier_sets=(),
                require_cosmos=True,
                include_cosmos=include_cosmos,
                max_scenes_per_set=max_scenes,
            )
        )


def test_malformed_label_is_reported_as_a_manifest_error(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table", "side"))
    chunk.joinpath("labels/000001.json").write_text("{broken")

    with pytest.raises(ManifestError, match="cannot parse label"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_accepted_cosmos_key_must_match_a_labelled_camera(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table", "side"))
    _write_image(chunk / "rgb_cosmos" / "ghost" / "000001.jpg")
    chunk.joinpath("_COSMOS.json").write_text(
        json.dumps({"ghost/000001": {"edge_fscore": 0.95, "accepted": True}})
    )

    with pytest.raises(ManifestError, match="no matching native record"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_symlinked_chunk_cannot_escape_the_synthetic_set(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    external = _write_chunk(tmp_path / "real-cell", "A", [1], ("table", "side"))
    data_root = tmp_path / "approved"
    set_dir = data_root / "synthetic" / "A"
    set_dir.mkdir(parents=True)
    set_dir.joinpath("chunk_0000").symlink_to(external, target_is_directory=True)

    with pytest.raises(ManifestError, match="outside approved"):
        build_manifests(
            ManifestConfig(
                data_root=data_root,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_symlinked_synthetic_root_cannot_redefine_the_approved_boundary(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    external_root = tmp_path / "external"
    _write_chunk(external_root, "A", [1], ("table",))
    data_root = tmp_path / "approved"
    data_root.mkdir()
    data_root.joinpath("synthetic").symlink_to(external_root / "synthetic", target_is_directory=True)

    with pytest.raises(ManifestError, match="outside approved"):
        build_manifests(
            ManifestConfig(
                data_root=data_root,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_internal_synthetic_symlink_cannot_redirect_to_real_data(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    data_root = tmp_path / "approved"
    _write_chunk(data_root / "real", "A", [1], ("table",))
    data_root.joinpath("synthetic").symlink_to(
        data_root / "real" / "synthetic",
        target_is_directory=True,
    )

    with pytest.raises(ManifestError, match="symlink"):
        build_manifests(
            ManifestConfig(
                data_root=data_root,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_symlinked_rgb_root_cannot_redefine_the_chunk_boundary(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table",))
    external_rgb = tmp_path / "external-rgb"
    chunk.joinpath("rgb").rename(external_rgb)
    chunk.joinpath("rgb").symlink_to(external_rgb, target_is_directory=True)

    with pytest.raises(ManifestError, match="outside approved"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_internal_rgb_symlink_cannot_redirect_to_superseded_data(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table",))
    superseded_rgb = chunk / "superseded" / "rgb"
    superseded_rgb.parent.mkdir()
    chunk.joinpath("rgb").rename(superseded_rgb)
    chunk.joinpath("rgb").symlink_to(superseded_rgb, target_is_directory=True)

    with pytest.raises(ManifestError, match="symlink"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_symlinked_cosmos_root_cannot_redefine_the_chunk_boundary(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table",))
    external_cosmos = tmp_path / "external-cosmos"
    _write_image(external_cosmos / "table" / "000001.jpg")
    chunk.joinpath("rgb_cosmos").symlink_to(external_cosmos, target_is_directory=True)
    chunk.joinpath("_COSMOS.json").write_text(
        json.dumps({"table/000001": {"edge_fscore": 0.95, "accepted": True}})
    )

    with pytest.raises(ManifestError, match="outside approved"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )


def test_label_camera_path_cannot_escape_the_chunk(tmp_path: Path) -> None:
    taxonomy = _write_taxonomy(tmp_path)
    chunk = _write_chunk(tmp_path, "A", [1], ("table",))
    external = tmp_path / "real-cell" / "camera"
    _write_image(external / "000001.jpg")
    label_path = chunk / "labels" / "000001.json"
    label = json.loads(label_path.read_text())
    label["cameras"] = {str(external.resolve()): label["cameras"]["table"]}
    label_path.write_text(json.dumps(label))

    with pytest.raises(ManifestError, match="camera|outside approved"):
        build_manifests(
            ManifestConfig(
                data_root=tmp_path,
                output_dir=tmp_path / "out",
                taxonomy_path=taxonomy,
                detector_sets=("A",),
                classifier_sets=(),
                require_cosmos=False,
            )
        )
