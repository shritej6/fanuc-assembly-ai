"""Production-manifest gates shared by the Rosie launch scripts.
Authored by Shritej Gunda."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_SETS = ("A", "F", "G", "B", "H", "K")
MIN_COSMOS_FRACTION = 0.10
MAX_COSMOS_FRACTION = 0.20
# Kept independent of prepare.py so a wrongly prepared manifest cannot loosen its own gate.
TESTBED_RANGES = {"N": (0.20, 0.30), "R": (0.20, 0.30), "Q": (0.20, 0.30), "O": (0.45, 0.55)}
TESTBED_BUILD_RANGES = {"P": (0.45, 0.55)}
# A test-bed set trains only with every chunk present; chunk ids run from chunk_0000 without gaps.
TESTBED_CHUNKS = {"N": 40, "R": 20, "Q": 8, "O": 48}
TESTBED_BUILD_CHUNKS = {"P": 24}
# Weak-spot sets are rendered without Cosmos, so any Cosmos image there is an error.
TESTBED_WEAK_RANGES = {**TESTBED_RANGES, "S": (0.0, 0.0), "T": (0.0, 0.0)}
TESTBED_WEAK_CHUNKS = {**TESTBED_CHUNKS, "S": 6, "T": 3}


class RosieGateError(ValueError):
    """Raised when Rosie training inputs are incomplete or unsafe."""


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RosieGateError(f"cannot read frozen manifest metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RosieGateError(f"manifest metadata must be an object: {path}")
    return value


def _cosmos_fraction(set_name: str, raw: Any) -> float:
    if not isinstance(raw, dict):
        raise RosieGateError(f"missing manifest statistics for set {set_name}")
    native_records, accepted = raw.get("native_records"), raw.get("cosmos_accepted")
    if not isinstance(native_records, int) or native_records < 1:
        raise RosieGateError(f"invalid native record count for set {set_name}: {native_records!r}")
    if not isinstance(accepted, int) or accepted < 0:
        raise RosieGateError(f"invalid accepted Cosmos count for set {set_name}: {accepted!r}")
    return accepted / native_records


def verify_production_manifest(path: Path) -> dict[str, Any]:
    """Require every set of the manifest's profile to carry its measured Cosmos acceptance share."""
    return verify_production_metadata(_read_metadata(path))


# Which profile each production trainer may read: only the test-bed sets train.
TRAINING_PROFILES = {"detector": ("testbed", "testbed_weak"), "identity": ("testbed", "testbed_weak"),
                     "build_state": ("testbed_build",)}


def verify_training_metadata(metadata: dict[str, Any], target: str) -> dict[str, Any]:
    """Gate a production trainer: its own profile, then every production check of that profile."""
    expected, profile = TRAINING_PROFILES[target], metadata.get("profile", "legacy")
    if profile not in expected:
        raise RosieGateError(f"{target} training requires a {' or '.join(expected)!r} manifest, got profile {profile!r}")
    return verify_production_metadata(metadata)


def verify_production_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """The checks of verify_production_manifest on metadata that is already loaded."""
    if metadata.get("schema_version") != 2 or metadata.get("content_hash_algorithm") != "sha256":
        raise RosieGateError("production manifest requires schema 2 with SHA-256 image binding")
    if metadata.get("require_cosmos") is not True or metadata.get("include_cosmos") is not True:
        raise RosieGateError("production training requires a Cosmos-gated manifest")
    sets = metadata.get("sets")
    if not isinstance(sets, dict):
        raise RosieGateError("production manifest lacks per-set statistics")
    for set_name, (low, high) in _expected_ranges(metadata).items():
        fraction = _cosmos_fraction(set_name, sets.get(set_name))
        if not low <= fraction <= high:
            raise RosieGateError(f"Cosmos acceptance for set {set_name} is {fraction:.4f}; expected {low:.2f}-{high:.2f}")
    _verify_complete_chunks(metadata.get("profile", "legacy"), sets)
    return metadata


def _verify_complete_chunks(profile: str, sets: dict[str, Any]) -> None:
    expected = {"testbed": TESTBED_CHUNKS, "testbed_build": TESTBED_BUILD_CHUNKS,
                "testbed_weak": TESTBED_WEAK_CHUNKS}.get(profile)
    if expected is None:
        return
    if set(sets) != set(expected):
        raise RosieGateError(f"{profile} manifest must hold exactly sets {sorted(expected)}, found {sorted(sets)}")
    for set_name, count in expected.items():
        chunks = sets[set_name].get("chunks")
        wanted = [f"chunk_{index:04d}" for index in range(count)]
        if not isinstance(chunks, list) or sorted(chunks) != wanted:
            found = len(chunks) if isinstance(chunks, list) else chunks
            missing = sorted(set(wanted) - set(chunks)) if isinstance(chunks, list) else wanted
            raise RosieGateError(f"set {set_name} is incomplete: {found} chunks, expected {count}; missing {missing}")


def _expected_ranges(metadata: dict[str, Any]) -> dict[str, tuple[float, float]]:
    profile = metadata.get("profile", "legacy")
    if profile in ("testbed", "testbed_weak"):
        if metadata.get("fallen_classes") is not True:
            raise RosieGateError("test-bed production manifest must carry the fallen classes")
        return TESTBED_RANGES if profile == "testbed" else TESTBED_WEAK_RANGES
    if profile == "testbed_build":
        if metadata.get("classifier_target") != "build_state":
            raise RosieGateError("set-P production manifest must train the build states")
        return TESTBED_BUILD_RANGES
    if profile != "legacy":
        raise RosieGateError(f"unknown manifest profile {profile!r}")
    return {name: (MIN_COSMOS_FRACTION, MAX_COSMOS_FRACTION) for name in EXPECTED_SETS}


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    verify_production_manifest(args.metadata)
    print(f"Cosmos production gate passed: {args.metadata.resolve()}")


if __name__ == "__main__":
    main()
