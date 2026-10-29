"""Behavior tests for the 8xH100 test-bed driver's stage status lines."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "perception" / "brev" / "h100_testbed.sh"


def _fake_box(tmp_path: Path, classifier_exit: int, gpu_mib: int = 81559) -> dict[str, str]:
    """A repo whose launchers are stubs, and an nvidia-smi that reports GPUs of `gpu_mib` MiB.

    Like the real tool, a memory query without --id writes one line per GPU over time, so a reader that closes the
    pipe early makes it die of SIGPIPE.
    """
    repo, bin_dir = tmp_path / "repo", tmp_path / "bin"
    (repo / "perception" / "rosie").mkdir(parents=True)
    bin_dir.mkdir()
    (repo / ".t13-code-commit").write_text("abc1234\n")
    (repo / "perception" / "rosie" / "classifier.sbatch").write_text(f'echo "$CONFIG"\nexit {classifier_exit}\n')
    (repo / "perception" / "rosie" / "detector.sbatch").write_text('echo "$CONFIG"\n')
    (bin_dir / "nvidia-smi").write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *-L*) echo GPU 0; echo GPU 1 ;;\n"
        f"  *--id=0*) echo {gpu_mib} ;;\n"
        f"  *) for gpu in 0 1 2 3 4 5 6 7; do echo {gpu_mib}; sleep 0.05; done ;;\n"
        "esac\n"
    )
    (bin_dir / "nvidia-smi").chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "REPO": str(repo),
        "STATUS": str(tmp_path / "status.txt"),
        "ART": str(tmp_path / "art"),
        "RUNS": str(tmp_path / "runs"),
    }


def _run(env: dict[str, str], start_at: str, skip: str = "") -> tuple[subprocess.CompletedProcess[str], list[str]]:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        env={**env, "START_AT": start_at, "SKIP": skip},
        check=False,
        capture_output=True,
        text=True,
    )
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
        "PARTIAL_DONE missing: env prepare_main prepare_build detector classifier",
    ]


def test_all_done_needs_every_stage_ok_in_this_or_an_earlier_run(tmp_path: Path) -> None:
    env = _fake_box(tmp_path, classifier_exit=0)
    earlier = ["env", "prepare_main", "prepare_build", "detector"]
    Path(env["STATUS"]).write_text("".join(f"2026-10-29T12:00:00Z {name} ok\n" for name in earlier))

    result, lines = _run(env, "classifier")

    assert result.returncode == 0, result.stderr
    assert lines[-1] == "ALL_DONE"


def test_a_failure_inside_a_stage_is_recorded_and_never_reaches_all_done(tmp_path: Path) -> None:
    result, lines = _run(_fake_box(tmp_path, classifier_exit=3), "build_classifier")

    assert result.returncode != 0
    assert lines[-2:] == ["build_classifier start", "build_classifier FAILED"]


def test_unknown_start_stage_is_refused_before_anything_is_marked(tmp_path: Path) -> None:
    result, lines = _run(_fake_box(tmp_path, classifier_exit=0), "detectr")

    assert result.returncode == 2
    assert "unknown stage detectr" in result.stderr
    assert lines == []


def test_detector_config_follows_gpu_memory(tmp_path: Path) -> None:
    for gpu_mib, expected in [(81559, "rtdetr_h100.yaml"), (40960, "rtdetr_a100_40g.yaml")]:
        box = tmp_path / str(gpu_mib)
        box.mkdir()
        result, lines = _run(_fake_box(box, classifier_exit=0, gpu_mib=gpu_mib), "detector")

        assert result.returncode == 0, result.stderr
        assert lines[0].endswith(f"gpu_mib={gpu_mib} detector_config={expected}")
        assert result.stdout.splitlines()[0].endswith(f"perception/configs/{expected}")


def test_skip_leaves_out_only_the_named_stages(tmp_path: Path) -> None:
    result, lines = _run(_fake_box(tmp_path, classifier_exit=0), "detector", skip="detector")

    assert result.returncode == 0, result.stderr
    assert "detector skipped (SKIP)" in lines
    assert "detector start" not in lines
    assert lines[-5:] == [
        "classifier start",
        "classifier ok",
        "build_classifier start",
        "build_classifier ok",
        "PARTIAL_DONE missing: env prepare_main prepare_build detector",
    ]


def test_unknown_skip_stage_is_refused_before_anything_is_marked(tmp_path: Path) -> None:
    result, lines = _run(_fake_box(tmp_path, classifier_exit=0), "env", skip="detectr")

    assert result.returncode == 2
    assert "unknown stage detectr" in result.stderr
    assert lines == []
