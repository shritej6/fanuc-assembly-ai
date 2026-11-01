"""The Slurm render wrapper succeeds only when the chunk itself is complete, whatever Isaac's exit code says.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from sdg import chunk_marks as cm

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "sdg" / "rosie" / "render_chunk.sbatch"
SPEC = {"index": 0, "cameras": ["side"], "outputs": ["rgb"], "seed": 0}

# Stands in for `singularity`: `overlay create` succeeds; `exec` plays Isaac per $FAKE_ISAAC and always exits 0, as
# Isaac does after an exception.
FAKE_SINGULARITY = """#!/bin/bash
[ "$1" = overlay ] && exit 0
out="${@: -3:1}"; mkdir -p "$out"
case "$FAKE_ISAAC" in
  crash) echo "Traceback: IK failed" > "$out/_FAILED.txt" ;;
  render) mkdir -p "$out/rgb/side" "$out/seg/side" "$out/labels"
          echo x > "$out/rgb/side/000000.jpg"; echo x > "$out/seg/side/000000.png"
          printf '%s' "$FAKE_LABEL" > "$out/labels/000000.json"; printf '%s' "$FAKE_DONE" > "$out/_DONE.json" ;;
esac
exit 0
"""

def _run(tmp_path: Path, isaac: str) -> tuple[int, Path, str]:
    data, bin_dir = tmp_path / "data", tmp_path / "bin"
    specs = data / "specs" / "A" / "chunk_0000.jsonl"
    specs.parent.mkdir(parents=True)
    specs.write_text(json.dumps(SPEC) + "\n")
    bin_dir.mkdir()
    for name, text in {"singularity": FAKE_SINGULARITY, "nvidia-smi": "#!/bin/sh\necho 550.0\n"}.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "REPO": str(REPO), "SET": "A",
           "DATA_ROOT": str(data), "ISAAC_SIF": "isaac.sif", "SLURM_ARRAY_TASK_ID": "0", "SLURM_ARRAY_JOB_ID": "1",
           "FAKE_ISAAC": isaac, "FAKE_LABEL": json.dumps({"spec": SPEC, "cameras": {"side": []}}),
           "FAKE_DONE": json.dumps({"scenes": 1, "specs_sha256": cm.specs_digest(specs), "depth_cams": []})}
    result = subprocess.run(["bash", str(SCRIPT)], env=env, check=False, capture_output=True, text=True)
    return result.returncode, data / "synthetic" / "A" / "chunk_0000", result.stdout + result.stderr


def test_isaac_exiting_0_after_a_crash_fails_the_task(tmp_path: Path) -> None:
    code, _, log = _run(tmp_path, "crash")

    assert code == 1
    assert "chunk A/0000 FAILED:" in log and "IK failed" in log


def test_isaac_exiting_0_without_finishing_fails_the_task(tmp_path: Path) -> None:
    code, _, log = _run(tmp_path, "nothing")

    assert code == 1
    assert "_DONE.json missing, unreadable or made from other specs" in log


def test_a_complete_chunk_passes(tmp_path: Path) -> None:
    code, out, log = _run(tmp_path, "render")

    assert code == 0, log
    assert cm.chunk_done(out, tmp_path / "data" / "specs" / "A" / "chunk_0000.jsonl")
