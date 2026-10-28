import copy
from pathlib import Path

import pytest

from perception.provenance import (
    ProvenanceMismatchError,
    assert_resume_compatible,
    current_git_commit,
    derived_data_identity,
    provenance_from_manifest,
)

CURRENT = {
    "manifest_sha256": "manifest-abc",
    "taxonomy_sha256": "taxonomy-abc",
    "split_seed": 1337,
    "labels": ["F01", "M1_01"],
    "pretrained_model": "PekingU/rtdetr_r50vd",
    "pretrained_revision": "revision-123",
    "manifest_code_commit": "manifest-code-abc",
    "derived_data_sha256": "derived-abc",
    "trainer_code_commit": "code-abc",
    "packages": {"torch": "2.5.1+cu118"},
    "training_config": {"learning_rate": 0.001, "max_steps": 2, "fp16": True},
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
        ("derived_data_sha256", "derived-other"),
        ("trainer_code_commit", "code-other"),
        ("packages", {"torch": "different"}),
    ],
)
def test_data_or_model_identity_change_blocks_resume(key: str, value: object) -> None:
    checkpoint = copy.deepcopy(CURRENT)
    checkpoint[key] = value

    with pytest.raises(ProvenanceMismatchError, match=key):
        assert_resume_compatible(CURRENT, checkpoint)


def test_only_bounded_smoke_step_increase_is_resume_compatible() -> None:
    current = copy.deepcopy(CURRENT)
    current["training_config"]["max_steps"] = 3
    assert_resume_compatible(current, CURRENT)

    current["training_config"]["learning_rate"] = 0.1
    with pytest.raises(ProvenanceMismatchError, match="training_config"):
        assert_resume_compatible(current, CURRENT)

    decreased = copy.deepcopy(CURRENT)
    decreased["training_config"]["max_steps"] = 1
    with pytest.raises(ProvenanceMismatchError, match="max_steps"):
        assert_resume_compatible(decreased, CURRENT)


def test_derived_data_identity_changes_when_consumed_bytes_change(tmp_path: Path) -> None:
    train = tmp_path / "train.json"
    validation = tmp_path / "validation.json"
    train.write_text("train-v1")
    validation.write_text("validation-v1")

    first = derived_data_identity({"train": train, "validation": validation})
    train.write_text("train-v2")
    second = derived_data_identity({"train": train, "validation": validation})

    assert first["derived_files"]["train"] != second["derived_files"]["train"]
    assert first["derived_data_sha256"] != second["derived_data_sha256"]


def test_archive_commit_override_requires_a_full_git_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "a" * 40
    monkeypatch.setenv("FANUC_CODE_COMMIT", commit)

    assert current_git_commit() == commit

    monkeypatch.setenv("FANUC_CODE_COMMIT", "not-a-commit")
    with pytest.raises(RuntimeError, match="FANUC_CODE_COMMIT"):
        current_git_commit()


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
        "manifest_code_commit": "abc1234",
    }
