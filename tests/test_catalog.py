"""Behavioral tests for the CAD-derived product catalog and recipe graph."""

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

import assembly.catalog as catalog_module
from assembly.catalog import (
    CatalogLoadError,
    CatalogResourceError,
    CatalogSchemaError,
    CatalogSemanticError,
    load_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "assembly" / "catalog.yaml"
OBJECT_GEOMETRY_SOURCE_PATH = ROOT / "assembly" / "sources" / "object_geometry.5a24e10.json"
GRASP_FRAMES_SOURCE_PATH = ROOT / "assembly" / "sources" / "grasp_frames.20c6f9b.json"
OBJECT_GEOMETRY_SHA256 = "488b2325c2048741aef9538bb678a48d845e9b75cd9f2a7f7c73a5781ae84325"
GRASP_FRAMES_SHA256 = "99611a032fb74a02740df405f1ef2d2d73f54f17cfc5676899cf9244e6d629c1"
CONTRACTS = ROOT / "contracts"

EXPECTED_PRODUCTS = [
    (1, "Rotary Knob Module", "A"),
    (2, "Indicator Light Module", "A"),
    (3, "Push Button Module", "B"),
    (4, "Sensor Connector Module", "C"),
    (5, "Control I/O Connector Module", "C"),
    (6, "Power Connector Module", "C"),
]

EXPECTED_PART_OWNERS = {
    "M1_01": 1,
    "M1_02": 1,
    "M1_03": 1,
    "M1_04": 1,
    "M2_01": 2,
    "M2_02": 2,
    "M2_03": 2,
    "M2_04": 2,
    "M3_01": 3,
    "M3_02": 3,
    "M3_03": 3,
    "M3_04": 3,
    "M4_01": 4,
    "M4_02": 4,
    "M4_03": 4,
    "M4_04": 4,
    "M5_01": 5,
    "M5_02": 5,
    "M5_03": 5,
    "M5_04": 5,
    "M6_01": 6,
    "M6_02": 6,
    "M6_03": 6,
    "M6_04": 6,
}


def _catalog_data() -> dict:
    assert CATALOG_PATH.exists(), "assembly/catalog.yaml has not been created"
    return yaml.safe_load(CATALOG_PATH.read_text())


def _verified_json_source(path: Path, expected_sha256: str) -> dict:
    contents = path.read_bytes()
    assert hashlib.sha256(contents).hexdigest() == expected_sha256
    return json.loads(contents)


def _schema_registry() -> Registry:
    registry = Registry()
    for path in CONTRACTS.glob("*.schema.json"):
        contents = json.loads(path.read_text())
        registry = registry.with_resource(path.name, Resource.from_contents(contents))
    return registry


def _write_catalog(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _write_json_source(path: Path, data: dict) -> str:
    contents = json.dumps(data, indent=1)
    path.write_text(contents)
    return hashlib.sha256(contents.encode()).hexdigest()


def _copy_schema_set(destination: Path) -> Path:
    for source in CONTRACTS.glob("*.schema.json"):
        (destination / source.name).write_text(source.read_text())
    return destination


def _replace_nested(data: dict, keys: tuple[str | int, ...], value: object) -> None:
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def test_catalog_yaml_conforms_to_contract() -> None:
    schema = json.loads((CONTRACTS / "catalog.schema.json").read_text())
    errors = list(Draft202012Validator(schema, registry=_schema_registry()).iter_errors(_catalog_data()))
    assert errors == []


@pytest.mark.parametrize("offset", [[0, 0], [0, 0, 0, 0]])
def test_catalog_contract_requires_exact_three_axis_insert_offset(offset: list[int]) -> None:
    document = deepcopy(_catalog_data())
    document["products"][0]["steps"][0]["insert_offset_mm"] = offset
    schema = json.loads((CONTRACTS / "catalog.schema.json").read_text())

    errors = list(Draft202012Validator(schema, registry=_schema_registry()).iter_errors(document))

    assert errors


def test_catalog_has_exact_products_names_and_base_types() -> None:
    products = _catalog_data()["products"]
    actual = [(item["product"], item["name"], item["base_type"]) for item in products]
    assert actual == EXPECTED_PRODUCTS


def test_catalog_covers_each_taxonomy_part_under_its_owner_once() -> None:
    products = _catalog_data()["products"]
    actual_pairs = [(step["object_id"], product["product"]) for product in products for step in product["steps"]]
    assert Counter(part for part, _ in actual_pairs) == Counter(EXPECTED_PART_OWNERS.keys())
    assert dict(actual_pairs) == EXPECTED_PART_OWNERS


def test_catalog_records_immutable_cad_geometry_provenance() -> None:
    catalog = _catalog_data()

    assert catalog["geometry_source"] == {
        "file": "sdg/data/object_geometry.json",
        "generator": "sdg/tools/object_geometry.py",
        "commit": "5a24e10d5ac7c9a6b15e9779385138678244819d",
        "commit_timestamp": "2026-10-27T22:11:27-05:00",
        "snapshot": "assembly/sources/object_geometry.5a24e10.json",
        "sha256": OBJECT_GEOMETRY_SHA256,
    }
    assert catalog["grasp_source"] == {
        "file": "cad/grasp_frames.json",
        "generator": "cad/grasp_frames.py",
        "commit": "20c6f9bab0aa44b3bae4cd915a1c957705d74da4",
        "commit_timestamp": "2026-10-29T15:34:19-05:00",
        "snapshot": "assembly/sources/grasp_frames.20c6f9b.json",
        "sha256": GRASP_FRAMES_SHA256,
    }


def test_catalog_installation_transforms_match_cad_source_within_005_mm() -> None:
    source = _verified_json_source(OBJECT_GEOMETRY_SOURCE_PATH, OBJECT_GEOMETRY_SHA256)
    expected_objects = {
        object_id: item["installed_in_base_frame"]
        for object_id, item in source["objects"].items()
        if object_id in EXPECTED_PART_OWNERS
    }
    actual = {step["object_id"]: step for product in _catalog_data()["products"] for step in product["steps"]}

    assert set(actual) == set(expected_objects)
    for object_id, transform in expected_objects.items():
        expected_offset = [row[3] for row in transform[:3]]
        expected_rotation = [row[:3] for row in transform[:3]]
        offset_error = max(abs(a - b) for a, b in zip(actual[object_id]["insert_offset_mm"], expected_offset))
        rotation_error = max(
            abs(a - b)
            for actual_row, expected_row in zip(actual[object_id]["installed_rotation_matrix"], expected_rotation)
            for a, b in zip(actual_row, expected_row)
        )
        assert offset_error <= 0.05, object_id
        assert rotation_error <= catalog_module.ROTATION_MATRIX_TOLERANCE, object_id


def test_catalog_gripper_rolls_match_single_t29_grasp_frame() -> None:
    source = _verified_json_source(GRASP_FRAMES_SOURCE_PATH, GRASP_FRAMES_SHA256)
    actual = {
        step["object_id"]: step["gripper_roll_deg"]
        for product in _catalog_data()["products"]
        for step in product["steps"]
    }

    assert set(actual) == set(EXPECTED_PART_OWNERS)
    for object_id, actual_roll in actual.items():
        grips = source["objects"][object_id]["grips"]
        assert len(grips) == 1, f"{object_id} has ambiguous T29 grips"
        expected_roll = 90.0 if grips[0]["closing_axis"] == "X" else 0.0
        assert grips[0]["roll_offset_deg"] == expected_roll
        assert actual_roll == expected_roll


def test_catalog_grasp_source_uses_team_v_printed_finger_contacts() -> None:
    source = _verified_json_source(GRASP_FRAMES_SOURCE_PATH, GRASP_FRAMES_SHA256)

    assert source["source"] == (
        "cad/products build() (printed CAD) + cad/fingers (Team V fingers), cad/grasp_frames.py"
    )
    assert source["gripper_gap_mm"] == {"closed_max": 4.5, "open_min": 16.5}
    assert source["objects"]["M1_04"]["grips"][0]["tcp_local_mm"] == [0.0, 0.0, 27.9]


def test_load_catalog_rejects_source_digest_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "tampered-grasp-source.json"
    path.write_bytes(GRASP_FRAMES_SOURCE_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(catalog_module, "GRASP_SOURCE_SNAPSHOT_PATH", path)

    with pytest.raises(CatalogResourceError, match="grasp source sha256"):
        load_catalog()


def test_load_catalog_rejects_ambiguous_multiple_grips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = deepcopy(_verified_json_source(GRASP_FRAMES_SOURCE_PATH, GRASP_FRAMES_SHA256))
    source["objects"]["M1_01"]["grips"].append(deepcopy(source["objects"]["M1_01"]["grips"][0]))
    source_path = tmp_path / "ambiguous-grasp-source.json"
    source_sha256 = _write_json_source(source_path, source)
    catalog_data = deepcopy(_catalog_data())
    catalog_data["grasp_source"]["sha256"] = source_sha256
    catalog_path = _write_catalog(tmp_path / "ambiguous-grasp-catalog.yaml", catalog_data)
    trusted_source = catalog_module.GraspSource(**catalog_data["grasp_source"])
    monkeypatch.setattr(catalog_module, "TRUSTED_GRASP_SOURCE", trusted_source, raising=False)
    monkeypatch.setattr(catalog_module, "GRASP_SOURCE_SNAPSHOT_PATH", source_path, raising=False)

    with pytest.raises(CatalogSemanticError, match="M1_01.*exactly one grip"):
        load_catalog(catalog_path)


def test_load_catalog_rejects_roll_not_matching_grasp_source(tmp_path: Path) -> None:
    invalid = deepcopy(_catalog_data())
    invalid["products"][0]["steps"][0]["gripper_roll_deg"] = 89
    path = _write_catalog(tmp_path / "wrong-source-roll.yaml", invalid)

    with pytest.raises(CatalogSemanticError, match="M1_01 gripper_roll_deg.*T29"):
        load_catalog(path)


def test_guard_installation_uses_cad_part_origin_and_rotation() -> None:
    step = _catalog_data()["products"][0]["steps"][3]

    assert step["object_id"] == "M1_04"
    assert step["insert_offset_mm"] == [0.0, 21.5, 21.0]
    assert step["installed_rotation_matrix"] == [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]


def test_load_catalog_returns_typed_products_and_steps() -> None:
    catalog = load_catalog()

    actual = [(product.product, product.name, product.base_type) for product in catalog.products]
    assert actual == EXPECTED_PRODUCTS
    assert catalog.product(4).steps[0].insert_offset_mm == (0.0, 0.0, 0.0)
    assert catalog.product(1).steps[3].installed_rotation_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    )
    assert catalog.product(1).steps[0].gripper_roll_deg == 90.0
    assert catalog.product(4).steps[0].gripper_roll_deg == 0.0


def test_load_catalog_rejects_contract_violation_with_typed_error(tmp_path: Path) -> None:
    invalid = deepcopy(_catalog_data())
    del invalid["products"][0]["name"]
    path = _write_catalog(tmp_path / "missing-name.yaml", invalid)

    with pytest.raises(CatalogSchemaError, match="name"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("products", 5, "product"), 5, "product IDs"),
        (("products", 0, "steps", 3, "step"), 3, "steps must be exactly 1..4"),
        (("products", 0, "steps", 0, "object_id"), "M2_01", "expected M1_01"),
        (("products", 2, "base_type"), "A", "base type A does not match"),
        (("products", 0, "steps", 0, "insert_offset_mm"), [0, float("inf"), 0], "three finite numbers"),
    ],
)
def test_load_catalog_rejects_semantic_errors(
    tmp_path: Path,
    keys: tuple[str | int, ...],
    value: object,
    message: str,
) -> None:
    invalid = deepcopy(_catalog_data())
    _replace_nested(invalid, keys, value)
    path = _write_catalog(tmp_path / "semantic-error.yaml", invalid)

    with pytest.raises(CatalogSemanticError, match=message):
        load_catalog(path)


@pytest.mark.parametrize(
    ("keys", "unknown_key"),
    [
        ((), "unreviewed_catalog_field"),
        (("products", 0), "unreviewed_product_field"),
        (("products", 0, "steps", 0), "unreviewed_step_field"),
    ],
)
def test_load_catalog_rejects_unknown_contract_keys(
    tmp_path: Path, keys: tuple[str | int, ...], unknown_key: str
) -> None:
    invalid = deepcopy(_catalog_data())
    target = invalid
    for key in keys:
        target = target[key]
    target[unknown_key] = "unreviewed"
    path = _write_catalog(tmp_path / "unknown-contract-key.yaml", invalid)

    with pytest.raises(CatalogSchemaError, match=unknown_key):
        load_catalog(path)


def test_load_catalog_rejects_gripper_roll_outside_contract(tmp_path: Path) -> None:
    invalid = deepcopy(_catalog_data())
    invalid["products"][0]["steps"][0]["gripper_roll_deg"] = 180.0001
    path = _write_catalog(tmp_path / "unsafe-gripper-roll.yaml", invalid)

    with pytest.raises(CatalogSchemaError, match="gripper_roll_deg"):
        load_catalog(path)


@pytest.mark.parametrize(
    "rotation",
    [
        pytest.param([[float("nan"), 0, 0], [0, 1, 0], [0, 0, 1]], id="nan"),
        pytest.param([[float("inf"), 0, 0], [0, 1, 0], [0, 0, 1]], id="infinite"),
        pytest.param([[1, 0, 0], [0, 1, 0], [0, 0, 0]], id="singular"),
        pytest.param([[1, 0, 0], [0, 1, 0], [0.1, 0, 1]], id="non-orthogonal"),
        pytest.param([[1, 0, 0], [0, 1, 0], [0, 0, -1]], id="reflection"),
    ],
)
def test_load_catalog_rejects_invalid_installed_rotation_matrices(tmp_path: Path, rotation: list[list[float]]) -> None:
    invalid = deepcopy(_catalog_data())
    invalid["products"][0]["steps"][0]["installed_rotation_matrix"] = rotation
    path = _write_catalog(tmp_path / "invalid-installed-rotation.yaml", invalid)

    with pytest.raises(CatalogSemanticError, match="proper rotation"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("missing.yaml", None, "Cannot read catalog"),
        ("invalid.yaml", "products: [", "Cannot parse catalog"),
    ],
)
def test_load_catalog_wraps_io_and_yaml_failures(
    tmp_path: Path,
    filename: str,
    contents: str | None,
    message: str,
) -> None:
    path = tmp_path / filename
    if contents is not None:
        path.write_text(contents)

    with pytest.raises(CatalogLoadError, match=message):
        load_catalog(path)


def test_load_catalog_rejects_duplicate_yaml_mapping_keys(tmp_path: Path) -> None:
    duplicate = CATALOG_PATH.read_text().replace(
        "        insert_offset_mm: [0, 0, 0]\n",
        "        insert_offset_mm: [0, 0, 0]\n        insert_offset_mm: [1, 2, 3]\n",
        1,
    )
    path = tmp_path / "duplicate-key.yaml"
    path.write_text(duplicate)

    with pytest.raises(CatalogLoadError, match="duplicate key 'insert_offset_mm'"):
        load_catalog(path)


def test_load_catalog_wraps_invalid_utf8(tmp_path: Path) -> None:
    path = tmp_path / "invalid-utf8.yaml"
    path.write_bytes(b"products: \xff")

    with pytest.raises(CatalogLoadError, match="Cannot read catalog"):
        load_catalog(path)


@pytest.mark.parametrize(
    ("attribute", "filename", "contents", "label"),
    [
        ("CATALOG_SCHEMA_PATH", "missing-schema.json", None, "schema"),
        ("CATALOG_SCHEMA_PATH", "malformed-schema.json", "{", "schema"),
        ("TAXONOMY_PATH", "missing-taxonomy.json", None, "taxonomy"),
        ("TAXONOMY_PATH", "malformed-taxonomy.json", "{", "taxonomy"),
        ("TAXONOMY_PATH", "invalid-utf8-taxonomy.json", b"\xff", "taxonomy"),
    ],
)
def test_load_catalog_wraps_resource_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attribute: str,
    filename: str,
    contents: str | bytes | None,
    label: str,
) -> None:
    path = tmp_path / filename
    if isinstance(contents, str):
        path.write_text(contents)
    elif contents is not None:
        path.write_bytes(contents)
    monkeypatch.setattr(catalog_module, attribute, path, raising=False)

    with pytest.raises(CatalogResourceError, match=label):
        load_catalog()


def test_load_catalog_wraps_unresolvable_schema_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unresolvable.schema.json"
    path.write_text(
        json.dumps(
            {
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": "unresolvable.schema.json",
                "$ref": "missing.schema.json",
            }
        )
    )
    monkeypatch.setattr(catalog_module, "CATALOG_SCHEMA_PATH", path, raising=False)

    with pytest.raises(CatalogResourceError, match="reference"):
        load_catalog()


@pytest.mark.parametrize("failure", ["invalid-structure", "unknown-dialect"])
def test_load_catalog_validates_every_registry_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    schema_dir = _copy_schema_set(tmp_path)
    common_path = schema_dir / "common.schema.json"
    common = json.loads(common_path.read_text())
    if failure == "invalid-structure":
        common["type"] = 7
    else:
        common["$schema"] = "https://example.invalid/unknown-dialect"
    common_path.write_text(json.dumps(common))
    monkeypatch.setattr(catalog_module, "CONTRACTS_PATH", schema_dir)
    monkeypatch.setattr(catalog_module, "CATALOG_SCHEMA_PATH", schema_dir / "catalog.schema.json")

    with pytest.raises(CatalogResourceError, match="common.schema.json"):
        load_catalog()
