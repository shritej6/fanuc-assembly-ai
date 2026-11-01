"""Tests for assembly sheets.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("cadquery")
from cad import assembly_sheets as sheets

TAXONOMY = sheets.load_taxonomy()
PRODUCTS = [(n, importlib.import_module(name)) for n, name in enumerate(sheets.MODULES, start=1)]


@pytest.fixture(scope="module")
def built() -> dict[int, tuple]:
    return {n: mod.build() for n, mod in PRODUCTS}


def test_there_is_one_sheet_per_product() -> None:
    assert [n for n, _ in PRODUCTS] == [1, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("product", range(1, 7))
def test_each_product_builds_on_its_own_base_in_four_ordered_steps(built: dict, product: int) -> None:
    fixture, parts = built[product]
    steps = sheets.steps(parts, TAXONOMY)

    assert fixture.name[:3] == f"F0{product}"
    assert [s["step"] for s in steps] == ["01", "02", "03", "04"]
    assert [s["object_id"] for s in steps] == [f"M{product}_0{i}" for i in range(1, 5)]
    assert all(TAXONOMY[s["object_id"]]["product"] == product for s in steps)


@pytest.mark.parametrize("product", range(1, 7))
def test_look_alikes_come_from_the_taxonomy(built: dict, product: int) -> None:
    for s in sheets.steps(built[product][1], TAXONOMY):
        assert s["lookalikes"] == TAXONOMY[s["object_id"]]["lookalikes"]


def test_the_gripper_turns_90_degrees_exactly_for_parts_gripped_along_y(built: dict) -> None:
    for _, parts in built.values():
        for part, s in zip(parts, sheets.steps(parts, TAXONOMY), strict=True):
            along_y = bool(part.grips) and part.grips[0].axis == "Y"
            assert s["gripper"] == ("turn gripper 90°" if along_y else "normal")
    turned = [s["object_id"] for _, parts in built.values() for s in sheets.steps(parts, TAXONOMY)
              if s["gripper"] != "normal"]
    assert turned, "no part needs the turned gripper: the rule would be untested"


def test_the_generator_writes_six_pages_and_one_pdf(tmp_path) -> None:
    sheets.main(tmp_path)

    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "Assembly_Instructions_All_Products.pdf", *[f"Product_{n}.png" for n in range(1, 7)]]
    assert all(p.stat().st_size > 50_000 for p in tmp_path.iterdir())
