"""Behavior tests for the 8xH100 test-bed driver's stage status lines."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "perception" / "brev" / "h100_testbed.sh"


def _fake_box(tmp_path: Path, classifier_exit: int) -> dict[str, str]:
    """A repo whose launchers are stubs, and an nvidia-smi that reports two GPUs."""
    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    (repo / "perception" / "rosie").mkdir(parents=True)
    bin_dir.mkdir()
    (repo / ".t13-code-commit").write_text("abc1234\n")
    (repo / "perception" / "rosie" / "classifier.sbatch").write_text(f"echo \"$CONFIG\"\nexit {classifier_exit}\n")
    (bin_dir / "nvidia-smi").write_text("#!/bin/sh\necho GPU 0\necho GPU 1\n")
    (bin_dir / "nvidia-smi").chmod(0o755)
    return {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "REPO": str(repo),
            "STATUS": str(tmp_path / "status.txt"), "ART": str(tmp_path / "art"), "RUNS": str(tmp_path / "runs")}


def _run(env: dict[str, str], start_at: str) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    result = subprocess.run(["bash", str(SCRIPT)], env={**env, "START_AT": start_at}, check=False,
                            capture_output=True, text=True)
    status = Path(env["STATUS"])
    lines = [line.split(" ", 1)[1] for line in status.read_text().splitlines()] if status.exists() else []
    return result, lines


def test_start_at_skips_earlier_stages_and_runs_from_the_named_one(tmp_path: Path) -> None:
    result, lines = _run(_fake_box(tmp_path, classifier_exit=0), "build_classifier")

    assert result.returncode == 0, result.stderr
    assert lines[1:] == [
        "env skipped (START_AT=build_classifier)",
        "prepare_main skipped (START_AT=build_classifier)",
        "prepare_build skipped (START_AT=build_classifier)",
        "detector skipped (START_AT=build_classifier)",
        "classifier skipped (START_AT=build_classifier)",
        "build_classifier start",
        "build_classifier ok",
        "ALL_DONE",
    ]


def test_a_failure_inside_a_stage_is_recorded_and_never_reaches_all_done(tmp_path: Path) -> None:
    result, lines = _run(_fake_box(tmp_path, classifier_exit=3), "build_classifier")

    assert result.returncode != 0
    assert lines[-2:] == ["build_classifier start", "build_classifier FAILED"]


def test_unknown_start_stage_is_refused_before_anything_is_marked(tmp_path: Path) -> None:
    result, lines = _run(_fake_box(tmp_path, classifier_exit=0), "detectr")

    assert result.returncode == 2
    assert "unknown START_AT=detectr" in result.stderr
    assert lines == []
