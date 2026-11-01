"""Back up closed data files incrementally to a private Hugging Face dataset repo.
Authored by Shritej Gunda.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".git", ".cache"}


class BackupError(RuntimeError):
    """An upload could not be verified on the remote repo."""


@dataclass(frozen=True)
class Root:
    """A local directory to back up and its prefix in the repo."""
    local: Path
    prefix: str
    closed: bool = False

    @classmethod
    def parse(cls, spec: str) -> Root:
        """Parse a <dir>:<prefix>[:closed] root spec."""
        parts = spec.split(":")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != "closed"):
            raise argparse.ArgumentTypeError(f"root must be <dir>:<prefix>[:closed], got {spec!r}")
        return cls(Path(parts[0]).expanduser(), parts[1].strip("/"), len(parts) == 3)


def closed_names(root: Path) -> set[str]:
    """Return the file names listed in the root's closed lists."""
    names: set[str] = set()
    for lst in root.glob("*closed*.txt"):
        names |= {Path(line.strip()).name for line in lst.read_text().splitlines() if line.strip()}
    return names


def candidates(root: Root, now: float, min_idle_s: float) -> dict[str, Path]:
    """remote path -> local file for every file of the root that is safe to read now."""
    closed = closed_names(root.local)
    out: dict[str, Path] = {}
    for f in sorted(root.local.rglob("*")):
        if not f.is_file() or SKIP_DIRS & set(f.relative_to(root.local).parts):
            continue
        if f.suffix == ".hdf5" and not root.closed and f.name not in closed and now - f.stat().st_mtime < min_idle_s:
            continue  # may still be written
        out[f"{root.prefix}/{f.relative_to(root.local).as_posix()}"] = f
    return out


def md5_of(path: Path) -> str:
    """Return the md5 hex digest of a file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def pending(files: dict[str, Path], manifest: dict) -> dict[str, dict]:
    """Files whose content differs from the manifest (md5 only recomputed when size or mtime changed)."""
    out = {}
    for remote, f in files.items():
        st, old = f.stat(), manifest.get(remote)
        if old and old["size"] == st.st_size and old["mtime"] == st.st_mtime:
            continue
        digest = md5_of(f)
        if old and old["md5"] == digest:
            old["mtime"] = st.st_mtime  # touched, same content: nothing to upload
            continue
        out[remote] = {"local": str(f), "md5": digest, "size": st.st_size, "mtime": st.st_mtime}
    return out


def batches(todo: dict[str, dict], max_bytes: int) -> list[list[str]]:
    """Commit groups of at most max_bytes (one oversized file goes alone)."""
    groups: list[list[str]] = []
    size = 0
    for remote in sorted(todo):
        if not groups or size + todo[remote]["size"] > max_bytes:
            groups.append([])
            size = 0
        groups[-1].append(remote)
        size += todo[remote]["size"]
    return groups


def push(api, repo: str, todo: dict[str, dict], manifest: dict, max_bytes: int) -> int:
    """Upload, verify against the remote file list, record verified files in the manifest. Returns files pushed."""
    from huggingface_hub import CommitOperationAdd

    for group in batches(todo, max_bytes):
        ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=todo[r]["local"]) for r in group]
        api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                          commit_message=f"backup {len(group)} files {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())}")
    remote = set(api.list_repo_files(repo_id=repo, repo_type="dataset"))
    missing = sorted(set(todo) - remote)
    for r in set(todo) - set(missing):
        manifest[r] = {k: todo[r][k] for k in ("md5", "size", "mtime")}
    if missing:
        raise BackupError(f"{len(missing)} uploaded files not listed on {repo}: {missing[:5]}")
    return len(todo)


def main() -> None:
    """Parse the command line and back up new or changed files."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--root", type=Root.parse, action="append", required=True)
    ap.add_argument("--manifest", type=Path, default=Path("~/t12_data/a7/hf_backup_manifest.json").expanduser())
    ap.add_argument("--min-idle-s", type=float, default=1800)
    ap.add_argument("--max-commit-gb", type=float, default=4.0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    manifest = json.loads(a.manifest.read_text()) if a.manifest.exists() else {}
    now, files = time.time(), {}
    for root in a.root:
        if root.local.is_dir():
            files.update(candidates(root, now, a.min_idle_s))
    todo = pending(files, manifest)
    gb = sum(t["size"] for t in todo.values()) / 1e9
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if a.dry_run or not todo:
        print(f"{stamp} backup {'dry-run' if a.dry_run else 'nothing new'}: {len(todo)} files {gb:.2f} GB", flush=True)
        return
    from huggingface_hub import HfApi

    api = HfApi()
    if not api.repo_exists(a.repo, repo_type="dataset"):  # a write token without "create repo" scope still pushes
        api.create_repo(a.repo, repo_type="dataset", private=True)
    try:
        n = push(api, a.repo, todo, manifest, int(a.max_commit_gb * 1e9))
    finally:
        a.manifest.parent.mkdir(parents=True, exist_ok=True)
        a.manifest.write_text(json.dumps(manifest, indent=0, sort_keys=True))
    print(f"{stamp} backup ok: {n} files {gb:.2f} GB verified on hf:{a.repo} ({len(manifest)} files total)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BackupError as e:
        sys.exit(f"backup FAILED: {e}")
