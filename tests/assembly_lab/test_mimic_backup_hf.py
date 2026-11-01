"""mimic/backup_hf.py: which files are safe to upload, the md5 manifest, and verification against the remote list."""

import os
import sys
import time
import types

import pytest

from sim.assembly_lab.mimic import backup_hf as b


def _touch(p, data=b"x", age_s=0.0):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def test_open_hdf5_waits_for_closed_list_or_idle(tmp_path):
    _touch(tmp_path / "pick_insert" / "new.hdf5")                      # being written
    _touch(tmp_path / "pick_insert" / "old.hdf5", age_s=4000)          # idle
    _touch(tmp_path / "pick_insert" / "listed.hdf5")                   # agent 3 closed it
    _touch(tmp_path / "pick_insert" / "new.hdf5.meta.jsonl")           # sidecars go up on every change
    (tmp_path / "run10_closed.txt").write_text("demos/pick_insert/listed.hdf5\n")
    _touch(tmp_path / "__pycache__" / "x.pyc")
    got = b.candidates(b.Root(tmp_path, "raw/demos"), time.time(), 1800)
    assert sorted(got) == ["raw/demos/pick_insert/listed.hdf5", "raw/demos/pick_insert/new.hdf5.meta.jsonl",
                           "raw/demos/pick_insert/old.hdf5", "raw/demos/run10_closed.txt"]
    closed_tree = b.candidates(b.Root(tmp_path, "raw/gen", closed=True), time.time(), 1800)
    assert "raw/gen/pick_insert/new.hdf5" in closed_tree  # gathered chunks are closed by construction


def test_root_spec_parsing():
    assert b.Root.parse("~/d:raw/gen:closed").closed
    assert b.Root.parse("/d:raw/demos/").prefix == "raw/demos"
    with pytest.raises(Exception, match="root must be"):
        b.Root.parse("/d:raw:open")


def test_manifest_skips_unchanged_and_touched_files(tmp_path):
    f = _touch(tmp_path / "a.hdf5", b"abc", age_s=4000)
    files = {"raw/a.hdf5": f}
    todo = b.pending(files, {})
    assert todo["raw/a.hdf5"]["md5"] == "900150983cd24fb0d6963f7d28e17f72"
    manifest = {"raw/a.hdf5": {k: todo["raw/a.hdf5"][k] for k in ("md5", "size", "mtime")}}
    assert b.pending(files, manifest) == {}
    os.utime(f, None)                         # touched, same bytes
    assert b.pending(files, manifest) == {} and manifest["raw/a.hdf5"]["mtime"] == f.stat().st_mtime
    f.write_bytes(b"abd")
    assert list(b.pending(files, manifest)) == ["raw/a.hdf5"]


def test_batches_respect_the_commit_size():
    todo = {"a": {"size": 3}, "b": {"size": 3}, "c": {"size": 9}, "d": {"size": 1}}
    assert b.batches(todo, 6) == [["a", "b"], ["c"], ["d"]]


class _Api:
    def __init__(self, drop=()):
        self.commits, self.files, self.drop = [], set(), set(drop)

    def create_commit(self, repo_id, repo_type, operations, commit_message):
        self.commits.append([op.path_in_repo for op in operations])
        self.files |= {op.path_in_repo for op in operations} - self.drop

    def list_repo_files(self, repo_id, repo_type):
        return sorted(self.files)


@pytest.fixture
def fake_hub(monkeypatch):
    mod = types.ModuleType("huggingface_hub")
    mod.CommitOperationAdd = lambda path_in_repo, path_or_fileobj: types.SimpleNamespace(path_in_repo=path_in_repo)
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)


def test_push_records_only_verified_files(tmp_path, fake_hub):
    todo = {r: {"local": str(tmp_path / r), "md5": r, "size": 1, "mtime": 0.0} for r in ("x", "y")}
    manifest = {}
    assert b.push(_Api(), "o/r", todo, manifest, 10) == 2 and set(manifest) == {"x", "y"}
    manifest = {}
    with pytest.raises(b.BackupError, match="1 uploaded files not listed"):
        b.push(_Api(drop={"y"}), "o/r", todo, manifest, 10)
    assert set(manifest) == {"x"}  # y is retried on the next push
