import copy

import pytest

from perception.provenance import (
    ProvenanceMismatchError,
    assert_resume_compatible,
    provenance_from_manifest,
)

CURRENT = {
    "manifest_sha256": "manifest-abc",
    "taxonomy_sha256": "taxonomy-abc",
    "split_seed": 1337,
    "labels": ["F01", "M1_01"],
    "pretrained_model": "PekingU/rtdetr_r50vd",
    "pretrained_revision": "revision-123",
}


def test_exact_provenance_match_allows_resume() -> None:
    assert_resume_compatible(CURRENT, copy.deepcopy(CURRENT))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("manifest_sha256", "manifest-other"),
        ("taxonomy_sha256", "taxonomy-other"),
        ("labels", ["M1_01", "F01"]),
        ("pretrained_revision", "revision-other"),
    ],
)
def test_data_or_model_identity_change_blocks_resume(key: str, value: object) -> None:
    checkpoint = copy.deepcopy(CURRENT)
    checkpoint[key] = value

    with pytest.raises(ProvenanceMismatchError, match=key):
        assert_resume_compatible(CURRENT, checkpoint)


def test_manifest_metadata_is_reduced_to_task_specific_resume_identity() -> None:
    metadata = {
        "git_commit": "abc1234",
        "split_seed": 17,
        "taxonomy_sha256": "taxonomy-abc",
        "labels": {"detector": ["F01", "foreign_screw_m3"], "classifier": ["F01"]},
        "detector": {"sha256": "detector-abc"},
        "classifier": {"sha256": "classifier-abc"},
    }

    provenance = provenance_from_manifest(
        metadata,
        task="detector",
        pretrained_model="PekingU/rtdetr_r50vd",
        pretrained_revision="revision-123",
    )

    assert provenance == {
        "manifest_sha256": "detector-abc",
        "taxonomy_sha256": "taxonomy-abc",
        "split_seed": 17,
        "labels": ["F01", "foreign_screw_m3"],
        "pretrained_model": "PekingU/rtdetr_r50vd",
        "pretrained_revision": "revision-123",
        "code_commit": "abc1234",
    }
