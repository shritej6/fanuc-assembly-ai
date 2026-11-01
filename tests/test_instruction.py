"""Trial instruction parsing is the only source of requested products.
Authored by Shritej Gunda.
"""

from inspect import signature
from itertools import permutations

import pytest

from assembly.catalog import load_catalog
from assembly.instruction import InstructionError, parse_requested_products


@pytest.mark.parametrize("products", tuple(permutations((1, 3, 6))))
def test_parse_requested_products_preserves_unseen_prompt_order(products: tuple[int, ...]) -> None:
    prompt = f"Please assemble products {products[0]}, {products[1]}, and {products[2]} on this trial."

    assert parse_requested_products(prompt, load_catalog()) == products


@pytest.mark.parametrize(
    "prompt",
    [
        "assemble the usual set",
        "assemble products 1, 1, and 3",
        "assemble products 1, 3, and 7",
        "assemble products 1 3",
        "assemble products 1, banana 3",
        "",
        None,
    ],
)
def test_parse_requested_products_fails_closed_on_ambiguous_or_invalid_prompt(prompt: object) -> None:
    with pytest.raises(InstructionError):
        parse_requested_products(prompt, load_catalog())


def test_controller_constructor_cannot_accept_a_fixed_product_scenario() -> None:
    from assembly.controller import AssemblyController

    assert "products" not in signature(AssemblyController).parameters


def test_parse_requested_products_ignores_numbers_outside_explicit_product_list() -> None:
    prompt = "Assemble products 2 and 5 for trial 99 using batch 2048."

    assert parse_requested_products(prompt, load_catalog()) == (2, 5)
