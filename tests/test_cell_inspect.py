"""T28: inspection verdicts follow the contract and the measured thresholds, and refuse missing evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from perception.cell.inspect import Evidence, InspectError, Thresholds, inspect

CONTRACTS = Path(__file__).resolve().parents[1] / "contracts"
REGISTRY = Registry().with_resources(
    (path.name, Resource.from_contents(json.loads(path.read_text()))) for path in CONTRACTS.glob("*.schema.json"))
VALIDATOR = Draft202012Validator(json.loads((CONTRACTS / "inspection.schema.json").read_text()), registry=REGISTRY)
TH = Thresholds(classifier_min=0.8, pose_fit_min=0.6, patchcore_max={"M4_04": 0.5, "M5_04": 0.5}, source="test")


def _ev(**kw) -> Evidence:
    base = {"detector_label": "M4_04", "classifier": ("M4_04", 0.97), "pose_fit": ("M4_04", 0.9),
            "patchcore_score": 0.2}
    return Evidence(**(base | kw))


def _check(result: dict) -> dict:
    assert list(VALIDATOR.iter_errors(result)) == []
    return result


def test_good_part_is_used():
    r = _check(inspect("p1", _ev(), TH))
    assert (r["verdict"], r["action"], r["checks_agree"], r["object_id"]) == ("ok", "use", True, "M4_04")


def test_patchcore_threshold_requests_operator_decision_without_reject_motion():
    r = _check(inspect("p1", _ev(patchcore_score=0.9), TH))
    assert (r["verdict"], r["action"], r["checks_agree"], r["object_id"]) == (
        "uncertain", "ask_operator", True, "M4_04")
    assert r["warnings"] == ["patchcore_threshold_exceeded"]


def test_fallen_part_is_rejected_with_the_detector_identity():
    r = _check(inspect("p1", _ev(detector_label="M3_03__fallen", classifier=None, pose_fit=None), TH))
    assert (r["verdict"], r["action"], r["object_id"]) == ("flipped", "reject", "M3_03")


def test_foreign_object_is_rejected():
    r = _check(inspect("p1", _ev(detector_label="foreign_screw_m3", classifier=None, pose_fit=None), TH))
    assert (r["verdict"], r["action"], r["object_id"]) == ("foreign", "reject", None)


def test_classifier_identity_wins_over_the_detector_label():
    r = _check(inspect("p1", _ev(detector_label="M4_04", classifier=("M5_04", 0.95), pose_fit=("M5_04", 0.8)), TH))
    assert (r["object_id"], r["action"]) == ("M5_04", "use")


def test_low_classifier_confidence_goes_to_the_operator():
    r = _check(inspect("p1", _ev(classifier=("M4_04", 0.6)), TH))
    assert (r["verdict"], r["action"], r["checks_agree"]) == ("uncertain", "ask_operator", False)


@pytest.mark.parametrize(
    ("fit", "warning"),
    [
        (("M5_04", 0.9), "foundationpose_identity_disagreement"),
        (("M4_04", 0.3), "foundationpose_low_fit"),
        (None, "foundationpose_unavailable"),
    ],
)
def test_foundationpose_warning_does_not_block_classifier_acceptance(fit, warning):
    r = _check(inspect("p1", _ev(pose_fit=fit), TH))
    assert (r["verdict"], r["action"], r["checks_agree"], r["object_id"]) == (
        "ok", "use", False, "M4_04")
    assert r["warnings"] == [warning]


def test_classifier_evidence_remains_required():
    with pytest.raises(InspectError, match="classifier"):
        inspect("p1", _ev(classifier=None), TH)


def test_missing_patchcore_is_an_advisory_not_an_acceptance_gate():
    r = _check(inspect("p1", _ev(patchcore_score=None), TH))
    assert (r["verdict"], r["action"], r["object_id"]) == ("ok", "use", "M4_04")
    assert r["warnings"] == ["patchcore_unavailable"]


def test_bases_need_no_patchcore():
    r = _check(inspect("p1", _ev(detector_label="F04", classifier=("F04", 0.99), pose_fit=("F04", 0.9),
                                 patchcore_score=None), TH))
    assert r["action"] == "use"


def test_advisory_patchcore_thresholds_may_be_partial_but_must_name_their_source(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"classifier_min": 0.8, "pose_fit_min": 0.6, "patchcore_max": {"M4_04": 0.5},
                                "source": "val run 1"}))
    assert Thresholds.load(path, ("M4_04",)).source == "val run 1"
    partial = Thresholds.load(path, ("M4_04", "M5_04"))
    assert partial.patchcore_max == {"M4_04": 0.5}
    path.write_text(json.dumps({"classifier_min": float("nan"), "pose_fit_min": 0.6, "patchcore_max": {},
                                "source": "x"}))
    with pytest.raises(InspectError, match="finite"):
        Thresholds.load(path, ())
    path.write_text(json.dumps({"classifier_min": 0.8, "pose_fit_min": 0.6, "patchcore_max": {}}))
    with pytest.raises(InspectError, match="measurement"):
        Thresholds.load(path, ())


def test_advisory_patchcore_threshold_map_may_be_omitted_but_must_be_a_mapping(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"classifier_min": 0.8, "pose_fit_min": 0.6, "source": "val run 1"}))

    assert Thresholds.load(path, ("M4_04",)).patchcore_max == {}

    path.write_text(
        json.dumps({"classifier_min": 0.8, "pose_fit_min": 0.6, "patchcore_max": [], "source": "val run 1"})
    )
    with pytest.raises(InspectError, match="mapping"):
        Thresholds.load(path, ("M4_04",))


def test_same_shape_bases_agree_but_different_types_do_not():
    agree = _check(inspect("p1", _ev(detector_label="F05", classifier=("F05", 0.9), pose_fit=("F04", 0.9),
                                     patchcore_score=None), TH))
    assert (agree["checks_agree"], agree["action"]) == (True, "use")
    differ = _check(inspect("p1", _ev(detector_label="F03", classifier=("F03", 0.9), pose_fit=("F01", 0.9),
                                      patchcore_score=None), TH))
    assert (differ["checks_agree"], differ["action"], differ["object_id"]) == (False, "use", "F03")
    assert differ["warnings"] == ["foundationpose_identity_disagreement"]
