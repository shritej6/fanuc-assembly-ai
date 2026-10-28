"""Behavioral tests for the CAD-derived product catalog and recipe graph."""

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
    UnknownBaseError,
    UnknownProductError,
    load_catalog,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "assembly" / "catalog.yaml"
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

EXPECTED_INSTALLATION = {
    "M1_01": ([0, 0, 0], 0),
    "M1_02": ([0, 0, 7], 0),
    "M1_03": ([0, 0, 18], 0),
    "M1_04": ([0, 17.5, 4], 0),
    "M2_01": ([0, 0, 0], 0),
    "M2_02": ([0, 0, 7], 0),
    "M2_03": ([0, 0, 18], 0),
    "M2_04": ([0, 17.5, 4], 0),
    "M3_01": ([0, 0, 0], 0),
    "M3_02": ([0, 0, 7], 0),
    "M3_03": ([0, 0, 18], 0),
    "M3_04": ([0, 17.5, 4], 0),
    "M4_01": ([0, 0, 0], 90),
    "M4_02": ([0, 0, 5], 0),
    "M4_03": ([0, 0, 18], 0),
    "M4_04": ([0, -11.5, 18], 0),
    "M5_01": ([0, 0, 0], 90),
    "M5_02": ([0, 0, 5], 0),
    "M5_03": ([0, 0, 18], 0),
    "M5_04": ([0, -11.5, 18], 0),
    "M6_01": ([0, 0, 0], 90),
    "M6_02": ([0, 0, 5], 0),
    "M6_03": ([0, 0, 18], 0),
    "M6_04": ([0, -11.5, 18], 0),
}


def _catalog_data() -> dict:
    assert CATALOG_PATH.exists(), "assembly/catalog.yaml has not been created"
    return yaml.safe_load(CATALOG_PATH.read_text())


def _schema_registry() -> Registry:
    registry = Registry()
    for path in CONTRACTS.glob("*.schema.json"):
        contents = json.loads(path.read_text())
        registry = registry.with_resource(path.name, Resource.from_contents(contents))
    return registry


def _write_catalog(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def _replace_nested(data: dict, keys: tuple[str | int, ...], value: object) -> None:
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def test_catalog_yaml_conforms_to_contract() -> None:
    schema = json.loads((CONTRACTS / "catalog.schema.json").read_text())
    errors = list(Draft202012Validator(schema, registry=_schema_registry()).iter_errors(_catalog_data()))
    assert errors == []


def test_catalog_has_exact_products_names_and_base_types() -> None:
    products = _catalog_data()["products"]
    actual = [(item["product"], item["name"], item["base_type"]) for item in products]
    assert actual == EXPECTED_PRODUCTS


def test_catalog_covers_each_taxonomy_part_under_its_owner_once() -> None:
    products = _catalog_data()["products"]
    actual_pairs = [
        (step["object_id"], product["product"])
        for product in products
        for step in product["steps"]
    ]
    assert Counter(part for part, _ in actual_pairs) == Counter(EXPECTED_PART_OWNERS.keys())
    assert dict(actual_pairs) == EXPECTED_PART_OWNERS


def test_catalog_uses_exact_cad_local_offsets_and_gripper_rolls() -> None:
    products = _catalog_data()["products"]
    actual = {
        step["object_id"]: (step["insert_offset_mm"], step["gripper_roll_deg"])
        for product in products
        for step in product["steps"]
    }
    assert actual == EXPECTED_INSTALLATION


def test_load_catalog_returns_typed_products_and_steps() -> None:
    catalog = load_catalog()

    actual = [(product.product, product.name, product.base_type) for product in catalog.products]
    assert actual == EXPECTED_PRODUCTS
    assert catalog.product(4).steps[0].insert_offset_mm == (0.0, 0.0, 0.0)
    assert catalog.product(4).steps[0].gripper_roll_deg == 90.0


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
        (("products", 0, "steps", 0, "insert_offset_mm"), [0, 0], "three finite numbers"),
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


def test_shared_bases_select_products_by_type_not_exact_fixture_id() -> None:
    catalog = load_catalog()
    actual = {
        base_id: tuple(product.product for product in catalog.compatible_products_for_base(base_id))
        for base_id in ("F01", "F02", "F03", "F04", "F05", "F06")
    }

    assert actual == {
        "F01": (1, 2),
        "F02": (1, 2),
        "F03": (3,),
        "F04": (4, 5, 6),
        "F05": (4, 5, 6),
        "F06": (4, 5, 6),
    }


def test_catalog_lookup_failures_use_typed_domain_errors() -> None:
    catalog = load_catalog()

    with pytest.raises(UnknownProductError, match="99"):
        catalog.product(99)
    with pytest.raises(UnknownBaseError, match="F99"):
        catalog.compatible_products_for_base("F99")


def test_recipe_graph_traverses_forward_then_exact_reverse() -> None:
    graph = load_catalog().recipe_graph(4)

    assert graph.nodes == ("base:C", "M4_01", "M4_02", "M4_03", "M4_04")
    assert graph.edges == (
        ("base:C", "M4_01"),
        ("M4_01", "M4_02"),
        ("M4_02", "M4_03"),
        ("M4_03", "M4_04"),
    )
    assert graph.assembly_traversal() == ("base:C", "M4_01", "M4_02", "M4_03", "M4_04")
    assert graph.dismantling_traversal() == ("M4_04", "M4_03", "M4_02", "M4_01")


def test_load_catalog_accepts_unique_products_in_any_yaml_order(tmp_path: Path) -> None:
    reordered = deepcopy(_catalog_data())
    reordered["products"].reverse()
    path = _write_catalog(tmp_path / "reordered.yaml", reordered)

    assert tuple(product.product for product in load_catalog(path).products) == (6, 5, 4, 3, 2, 1)
