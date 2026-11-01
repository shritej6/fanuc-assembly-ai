"""Tests for contracts.
Authored by Shritej Gunda.
"""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
SCHEMAS = sorted(CONTRACTS.glob("*.schema.json"))


def _registry() -> Registry:
    reg = Registry()
    for path in SCHEMAS:
        reg = reg.with_resource(path.name, Resource.from_contents(json.loads(path.read_text())))
    return reg


@pytest.mark.parametrize("path", SCHEMAS, ids=lambda p: p.name)
def test_schema_is_valid(path: Path) -> None:
    Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_taxonomy_conforms() -> None:
    schema = json.loads((CONTRACTS / "taxonomy.schema.json").read_text())
    data = json.loads((CONTRACTS / "taxonomy.json").read_text())
    errors = list(Draft202012Validator(schema, registry=_registry()).iter_errors(data))
    assert errors == []


def test_taxonomy_covers_all_30_objects() -> None:
    data = json.loads((CONTRACTS / "taxonomy.json").read_text())
    ids = {o["object_id"] for o in data["objects"]}
    assert len(ids) == 30
    assert {f"F0{p}" for p in range(1, 7)} <= ids


def test_lookalikes_are_symmetric() -> None:
    objs = {o["object_id"]: set(o["lookalikes"]) for o in json.loads((CONTRACTS / "taxonomy.json").read_text())["objects"]}
    for oid, others in objs.items():
        for other in others:
            assert oid in objs[other], f"{oid} lists {other} but not the reverse"


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(json.loads((CONTRACTS / name).read_text()), registry=_registry())


def test_inspection_reports_error_codes() -> None:
    v = _validator("inspection.schema.json")
    verdict = {"instance_id": "p7", "verdict": "uncertain", "action": "ask_operator", "checks_agree": False,
               "error_codes": ["LOOKALIKE_UNRESOLVED", "OCCLUDED"]}
    assert list(v.iter_errors(verdict)) == []
    assert list(v.iter_errors({**verdict, "error_codes": ["SOMETHING_WRONG"]}))
    assert list(v.iter_errors({**verdict, "error_codes": ["OCCLUDED", "OCCLUDED"]}))


def test_plan_step_names_its_error_code() -> None:
    v = _validator("plan_step.schema.json")
    step = {"step_id": "s3", "op": "move_to_reject", "product": 1, "error_code": "GUARD_TOPPLED"}
    assert list(v.iter_errors(step)) == []
    assert list(v.iter_errors({**step, "error_code": None})) == []
    assert list(v.iter_errors({**step, "error_code": "FLIPPED"}))


def test_readme_explains_every_error_code() -> None:
    codes = json.loads((CONTRACTS / "common.schema.json").read_text())["$defs"]["error_code"]["enum"]
    readme = (CONTRACTS / "README.md").read_text()
    assert len(codes) == 11
    assert all(f"| `{c}` |" in readme for c in codes)


def test_same_size_control_housings_require_a_lookalike_check() -> None:
    objects = {
        item["object_id"]: set(item["lookalikes"])
        for item in json.loads((CONTRACTS / "taxonomy.json").read_text())["objects"]
    }

    assert "M3_01" in objects["M1_01"]
    assert "M1_01" in objects["M3_01"]
