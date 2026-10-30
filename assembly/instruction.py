"""Fail-closed parsing of products requested by the trial prompt."""

import re

from assembly.catalog import Catalog


class InstructionError(ValueError):
    """A trial instruction does not unambiguously select catalog products."""


def parse_requested_products(instruction: object, catalog: Catalog) -> tuple[int, ...]:
    if not isinstance(instruction, str) or not instruction.strip():
        raise InstructionError("trial.instruction must be a non-empty string")
    marker = re.search(r"\bproducts?\b", instruction, flags=re.IGNORECASE)
    if marker is None:
        raise InstructionError("trial.instruction must explicitly name product IDs")
    tail = instruction[marker.end() :]
    listed = re.match(
        r"\s*(\d+(?:\s*(?:,\s*(?:and\s+)?|\band\s+)\d+)*)",
        tail,
        flags=re.IGNORECASE,
    )
    if listed is None:
        raise InstructionError("trial.instruction must include a product list after 'products'")
    remainder = tail[listed.end() :]
    if re.match(r"\s*(?:,|\band\b|\d)", remainder, flags=re.IGNORECASE):
        raise InstructionError("trial.instruction contains a malformed product list")
    product_ids = tuple(int(value) for value in re.findall(r"\b\d+\b", listed.group(1)))
    if not product_ids or len(set(product_ids)) != len(product_ids):
        raise InstructionError("trial.instruction must name a non-empty unique product sequence")
    known = {product.product for product in catalog.products}
    if any(product not in known for product in product_ids):
        raise InstructionError("trial.instruction contains an unknown product ID")
    return product_ids
