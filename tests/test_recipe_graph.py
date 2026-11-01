"""Behavioral tests for catalog lookup and recipe traversal.
Authored by Shritej Gunda.
"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from assembly.catalog import UnknownBaseError, UnknownProductError, load_catalog

CATALOG_PATH = Path(__file__).resolve().parents[1] / "assembly" / "catalog.yaml"


def _catalog_data() -> dict:
    return yaml.safe_load(CATALOG_PATH.read_text())


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
    path = tmp_path / "reordered.yaml"
    path.write_text(yaml.safe_dump(reordered, sort_keys=False))

    assert tuple(product.product for product in load_catalog(path).products) == (6, 5, 4, 3, 2, 1)
