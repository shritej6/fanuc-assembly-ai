import json
from pathlib import Path

import pytest

from perception.taxonomy import FOREIGN_LABELS, load_ontology

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "taxonomy.json"
CATALOG_IDS = (
    "F01",
    "F02",
    "F03",
    "F04",
    "F05",
    "F06",
    "M1_01",
    "M1_02",
    "M1_03",
    "M1_04",
    "M2_01",
    "M2_02",
    "M2_03",
    "M2_04",
    "M3_01",
    "M3_02",
    "M3_03",
    "M3_04",
    "M4_01",
    "M4_02",
    "M4_03",
    "M4_04",
    "M5_01",
    "M5_02",
    "M5_03",
    "M5_04",
    "M6_01",
    "M6_02",
    "M6_03",
    "M6_04",
)


def test_contract_order_is_the_stable_classifier_mapping() -> None:
    ontology = load_ontology(CONTRACT)

    assert ontology.classifier_labels == CATALOG_IDS
    assert ontology.classifier_label2id == {label: index for index, label in enumerate(CATALOG_IDS)}


def test_detector_keeps_all_catalog_ids_and_appends_foreign_distractors() -> None:
    ontology = load_ontology(CONTRACT)

    assert FOREIGN_LABELS == (
        "foreign_screw_m3",
        "foreign_tape_roll",
        "foreign_cube_10mm",
    )
    assert ontology.detector_labels == CATALOG_IDS + FOREIGN_LABELS
    assert ontology.detector_label2id["foreign_cube_10mm"] == 32


def test_lookalike_components_are_reported_as_complete_groups() -> None:
    ontology = load_ontology(CONTRACT)

    assert ontology.lookalike_groups == (
        ("F01", "F02", "F03", "F04", "F05", "F06"),
        ("M1_01", "M2_01", "M3_01"),
        ("M1_02", "M2_02"),
        ("M1_04", "M2_04", "M3_04"),
        ("M4_01", "M5_01", "M6_01"),
        ("M4_03", "M5_03", "M6_03"),
        ("M4_04", "M5_04", "M6_04"),
    )


@pytest.mark.parametrize(
    ("objects", "message"),
    [
        (
            [
                {"object_id": "A", "lookalikes": []},
                {"object_id": "A", "lookalikes": []},
            ],
            "duplicate object_id",
        ),
        (
            [{"object_id": "A", "lookalikes": ["B"]}],
            "unknown look-alike",
        ),
        (
            [
                {"object_id": "A", "lookalikes": ["B"]},
                {"object_id": "B", "lookalikes": []},
            ],
            "not symmetric",
        ),
    ],
)
def test_invalid_taxonomy_links_are_rejected(tmp_path: Path, objects: list[dict], message: str) -> None:
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps({"objects": objects}))

    with pytest.raises(ValueError, match=message):
        load_ontology(path)
