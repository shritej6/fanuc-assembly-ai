"""Typed access to CAD-local assembly recipes.

Insertion offsets are expressed in the detected fixture's CAD-local frame.
They require a measured fixture-pose transform and are never absolute UF0
robot targets.
"""

import json
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from math import isfinite
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from referencing import Registry, Resource
from referencing.exceptions import CannotDetermineSpecification, Unresolvable
from yaml.constructor import ConstructorError
from yaml.nodes import MappingNode
from yaml.resolver import BaseResolver

BaseType = Literal["A", "B", "C"]
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
CONTRACTS_PATH = ROOT / "contracts"
CATALOG_SCHEMA_PATH = CONTRACTS_PATH / "catalog.schema.json"
TAXONOMY_PATH = CONTRACTS_PATH / "taxonomy.json"


class CatalogError(Exception):
    """Base class for catalog domain failures."""


class CatalogSchemaError(CatalogError):
    """Raised when catalog data violates the shared JSON schema."""


class CatalogLoadError(CatalogError):
    """Raised when a catalog file cannot be read or parsed."""


class CatalogResourceError(CatalogError):
    """Raised when a schema or taxonomy resource cannot be used."""


class CatalogSemanticError(CatalogError):
    """Raised when schema-valid catalog data is unsafe or inconsistent."""


class UnknownProductError(CatalogError):
    """Raised when a recipe is requested for an unknown product."""


class UnknownBaseError(CatalogError):
    """Raised when a fixture object is absent from the taxonomy."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that fails closed on duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            if key in mapping:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True, slots=True)
class RecipeStep:
    """One part installation relative to the detected fixture pose."""

    step: int
    object_id: str
    insert_offset_mm: tuple[float, float, float]
    gripper_roll_deg: float


@dataclass(frozen=True, slots=True)
class ProductRecipe:
    product: int
    name: str
    base_type: BaseType
    steps: tuple[RecipeStep, ...]


@dataclass(frozen=True, slots=True)
class RecipeGraph:
    """A linear dependency graph rooted at a shared fixture type."""

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]

    def assembly_traversal(self) -> tuple[str, ...]:
        return self.nodes

    def dismantling_traversal(self) -> tuple[str, ...]:
        return tuple(reversed(self.nodes[1:]))


@dataclass(frozen=True, slots=True)
class Catalog:
    products: tuple[ProductRecipe, ...]
    _base_types_by_object: tuple[tuple[str, BaseType], ...]

    def product(self, product_id: int) -> ProductRecipe:
        for product in self.products:
            if product.product == product_id:
                return product
        raise UnknownProductError(f"Unknown product {product_id}")

    def base_type_for_object(self, base_object_id: str) -> BaseType:
        try:
            return dict(self._base_types_by_object)[base_object_id]
        except KeyError as error:
            raise UnknownBaseError(f"Unknown base object {base_object_id}") from error

    def compatible_products_for_base(self, base_object_id: str) -> tuple[ProductRecipe, ...]:
        base_type = self.base_type_for_object(base_object_id)
        return tuple(product for product in self.products if product.base_type == base_type)

    def recipe_graph(self, product_id: int) -> RecipeGraph:
        product = self.product(product_id)
        nodes = (f"base:{product.base_type}", *(step.object_id for step in product.steps))
        return RecipeGraph(nodes=nodes, edges=tuple(pairwise(nodes)))


def _schema_registry() -> Registry:
    registry = Registry()
    for path in CONTRACTS_PATH.glob("*.schema.json"):
        contents = _read_json_resource(path, f"schema {path.name}")
        try:
            resource = Resource.from_contents(contents)
        except CannotDetermineSpecification as error:
            raise CatalogResourceError(f"Cannot identify schema resource {path}: {error}") from error
        registry = registry.with_resource(path.name, resource)
    return registry


def _read_catalog_data(path: Path) -> Any:
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CatalogLoadError(f"Cannot read catalog {path}: {error}") from error
    try:
        return yaml.load(contents, Loader=_UniqueKeySafeLoader)
    except yaml.YAMLError as error:
        raise CatalogLoadError(f"Cannot parse catalog {path}: {error}") from error


def _read_json_resource(path: Path, label: str) -> Any:
    try:
        contents = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise CatalogResourceError(f"Cannot read {label} resource {path}: {error}") from error
    try:
        return json.loads(contents)
    except json.JSONDecodeError as error:
        raise CatalogResourceError(f"Cannot parse {label} resource {path}: {error}") from error


def _validate_schema(data: Any) -> None:
    schema = _read_json_resource(CATALOG_SCHEMA_PATH, "catalog schema")
    try:
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema, registry=_schema_registry())
        errors = sorted(validator.iter_errors(data), key=lambda error: error.json_path)
    except SchemaError as error:
        raise CatalogResourceError(f"Invalid catalog schema resource {CATALOG_SCHEMA_PATH}: {error}") from error
    except Unresolvable as error:
        raise CatalogResourceError(f"Cannot resolve catalog schema reference: {error}") from error
    if errors:
        details = "; ".join(f"{error.json_path}: {error.message}" for error in errors)
        raise CatalogSchemaError(f"Catalog schema validation failed: {details}")


def _taxonomy_objects() -> list[dict[str, Any]]:
    taxonomy = _read_json_resource(TAXONOMY_PATH, "taxonomy")
    if not isinstance(taxonomy, dict) or not isinstance(taxonomy.get("objects"), list):
        raise CatalogResourceError(f"Malformed taxonomy resource {TAXONOMY_PATH}: missing objects list")
    return taxonomy["objects"]


def _taxonomy_reference() -> tuple[dict[str, int], dict[int, str]]:
    try:
        objects = _taxonomy_objects()
        part_owners = {
            item["object_id"]: item["product"]
            for item in objects
            if item["object_id"].startswith("M")
        }
        base_types = {
            item["product"]: item["base_type"]
            for item in objects
            if item["object_id"].startswith("F")
        }
    except (AttributeError, KeyError, TypeError) as error:
        raise CatalogResourceError(f"Malformed taxonomy resource {TAXONOMY_PATH}: {error}") from error
    return part_owners, base_types


def _base_type_pairs() -> tuple[tuple[str, BaseType], ...]:
    try:
        return tuple(
            (item["object_id"], cast(BaseType, item["base_type"]))
            for item in _taxonomy_objects()
            if item["object_id"].startswith("F")
        )
    except (AttributeError, KeyError, TypeError) as error:
        raise CatalogResourceError(f"Malformed taxonomy resource {TAXONOMY_PATH}: {error}") from error


def _validate_product_ids(products: list[dict[str, Any]]) -> None:
    product_ids = [product["product"] for product in products]
    if sorted(product_ids) != [1, 2, 3, 4, 5, 6]:
        raise CatalogSemanticError("Catalog product IDs must be exactly 1..6 with no duplicates")


def _valid_offset(offset: Any) -> bool:
    return (
        isinstance(offset, list)
        and len(offset) == 3
        and all(isinstance(value, (int, float)) and isfinite(value) for value in offset)
    )


def _validate_step(product_id: int, step: dict[str, Any], part_owners: dict[str, int]) -> None:
    expected_id = f"M{product_id}_{step['step']:02d}"
    object_id = step["object_id"]
    if object_id != expected_id:
        raise CatalogSemanticError(f"Product {product_id} step {step['step']} expected {expected_id}, got {object_id}")
    if part_owners.get(object_id) != product_id:
        raise CatalogSemanticError(f"{object_id} is not owned by product {product_id} in the taxonomy")
    if not _valid_offset(step.get("insert_offset_mm")):
        raise CatalogSemanticError(f"{object_id} insert_offset_mm must contain three finite numbers")
    roll = step.get("gripper_roll_deg")
    if not isinstance(roll, (int, float)) or not isfinite(roll):
        raise CatalogSemanticError(f"{object_id} gripper_roll_deg must be finite")


def _validate_product(
    product: dict[str, Any],
    part_owners: dict[str, int],
    base_types: dict[int, str],
) -> None:
    product_id = product["product"]
    base_type = product["base_type"]
    if base_type != base_types.get(product_id):
        expected = base_types.get(product_id)
        raise CatalogSemanticError(f"Product {product_id} base type {base_type} does not match taxonomy type {expected}")
    step_numbers = [step["step"] for step in product["steps"]]
    if step_numbers != [1, 2, 3, 4]:
        raise CatalogSemanticError(f"Product {product_id} steps must be exactly 1..4 in order")
    for step in product["steps"]:
        _validate_step(product_id, step, part_owners)


def _validate_coverage(products: list[dict[str, Any]], part_owners: dict[str, int]) -> None:
    actual = [step["object_id"] for product in products for step in product["steps"]]
    if Counter(actual) != Counter(part_owners.keys()):
        raise CatalogSemanticError("Catalog must cover every non-base taxonomy object exactly once")


def _validate_semantics(data: dict[str, Any]) -> None:
    products = data["products"]
    part_owners, base_types = _taxonomy_reference()
    _validate_product_ids(products)
    for product in products:
        _validate_product(product, part_owners, base_types)
    _validate_coverage(products, part_owners)


def _make_step(data: dict[str, Any]) -> RecipeStep:
    offset = tuple(float(value) for value in data["insert_offset_mm"])
    return RecipeStep(
        step=int(data["step"]),
        object_id=str(data["object_id"]),
        insert_offset_mm=offset,
        gripper_roll_deg=float(data["gripper_roll_deg"]),
    )


def _make_product(data: dict[str, Any]) -> ProductRecipe:
    return ProductRecipe(
        product=int(data["product"]),
        name=str(data["name"]),
        base_type=data["base_type"],
        steps=tuple(_make_step(step) for step in data["steps"]),
    )


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> Catalog:
    """Load the catalog into immutable domain objects."""

    data = _read_catalog_data(Path(path))
    _validate_schema(data)
    _validate_semantics(data)
    products = tuple(_make_product(product) for product in data["products"])
    return Catalog(products, _base_type_pairs())
