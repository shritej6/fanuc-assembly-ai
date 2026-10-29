"""Behavior tests for the PatchCore stage: it succeeds only when all 24 M parts are freshly measured."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "perception" / "brev" / "h100_t14.sh"

# Stands in for the PatchCore venv's python: pip / import checks pass, the crop export writes nothing of note, and
# train_patchcore writes metrics.json with $FAKE_STATUS (or $FAKE_STATUS_<part> for one part; "none" writes
# nothing, as a crash would). Inline -c code runs on the real interpreter.
FAKE_PYTHON = f"""#!/bin/sh
case "$1 $2" in
  "-m pip") exit 0 ;;
  "-m perception.anomaly.testbed") mkdir -p "$8"; exit 0 ;;
  "-m perception.anomaly.train_patchcore")
    part="$6"; out="$8"; mkdir -p "$out/$part"
    eval status=\\${{FAKE_STATUS_$part:-$FAKE_STATUS}}
    [ "$status" = none ] && exit 1   # the part crashed before writing metrics
    echo "{{\\"status\\": \\"$status\\"}}" > "$out/$part/metrics.json"; exit 0 ;;
esac
case "$2" in *anomalib*) echo fake; exit 0 ;; esac
exec {sys.executable} "$@"
"""


def _fake_box(tmp_path: Path, audit_exit: int = 0) -> dict[str, str]:
    home, venv = tmp_path / "home", tmp_path / "pc-venv"
    (home / ".local" / "bin").mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    stubs = {
        home / ".local" / "bin" / "uv": "#!/bin/sh\nexit 0\n",
        venv / "bin" / "python": FAKE_PYTHON,
        venv / "bin" / "pip-audit": f"#!/bin/sh\nexit {audit_exit}\n",
        tmp_path / "bin" / "nvidia-smi": "#!/bin/sh\necho GPU 0\necho GPU 1\n",
    }
    for path, text in stubs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        path.chmod(0o755)
    return {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
        "REPO": str(REPO),
        "PC_VENV": str(venv),
        "PC": str(tmp_path / "pc"),
        "STATUS": str(tmp_path / "status.txt"),
        "FAKE_STATUS": "measured",
    }


def _run(env: dict[str, str]) -> tuple[int, list[str]]:
    result = subprocess.run(["bash", str(SCRIPT)], env=env, check=False, capture_output=True, text=True)
    lines = [line.split(" ", 1)[1] for line in Path(env["STATUS"]).read_text().splitlines()]
    return result.returncode, lines


def test_all_24_measured_parts_pass(tmp_path: Path) -> None:
    code, lines = _run(_fake_box(tmp_path))

    assert code == 0
    assert lines[-2:] == ["patchcore all 24 parts measured", "patchcore ok"]


def test_a_part_that_was_only_fitted_fails_the_stage(tmp_path: Path) -> None:
    env = {**_fake_box(tmp_path), "FAKE_STATUS_M3_02": "fitted_too_few_test_crops"}

    code, lines = _run(env)

    assert code != 0
    assert lines[-1] == "patchcore parts not measured: M3_02=fitted_too_few_test_crops"


def test_a_failed_security_audit_blocks_patchcore_before_any_training(tmp_path: Path) -> None:
    code, lines = _run(_fake_box(tmp_path, audit_exit=1))

    assert code != 0
    assert lines[-1].startswith("patchcore_env blocked: strict audit")
    assert not (tmp_path / "pc" / "runs").exists()


def test_metrics_left_by_an_earlier_run_do_not_count(tmp_path: Path) -> None:
    env = {**_fake_box(tmp_path), "FAKE_STATUS_M1_01": "none"}
    stale = tmp_path / "pc" / "runs" / "M1_01" / "metrics.json"
    stale.parent.mkdir(parents=True)
    stale.write_text('{"status": "measured"}')

    code, lines = _run(env)

    assert code != 0
    assert lines[-1] == "patchcore parts not measured: M1_01=no metrics.json"
