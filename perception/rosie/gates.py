"""Production-manifest gates shared by the Rosie launch scripts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXPECTED_SETS = ("A", "F", "G", "B", "H", "K")
MIN_COSMOS_FRACTION = 0.10
MAX_COSMOS_FRACTION = 0.20


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
    scenes, accepted = raw.get("scenes"), raw.get("cosmos_accepted")
    if not isinstance(scenes, int) or scenes < 1:
        raise RosieGateError(f"invalid scene count for set {set_name}: {scenes!r}")
    if not isinstance(accepted, int) or accepted < 0:
        raise RosieGateError(f"invalid accepted Cosmos count for set {set_name}: {accepted!r}")
    return accepted / scenes


def verify_production_manifest(path: Path) -> dict[str, Any]:
    """Require all six sets to carry a measured 10-20% Cosmos acceptance share."""
    metadata = _read_metadata(path)
    if metadata.get("require_cosmos") is not True or metadata.get("include_cosmos") is not True:
        raise RosieGateError("production training requires a Cosmos-gated manifest")
    sets = metadata.get("sets")
    if not isinstance(sets, dict):
        raise RosieGateError("production manifest lacks per-set statistics")
    for set_name in EXPECTED_SETS:
        fraction = _cosmos_fraction(set_name, sets.get(set_name))
        if not MIN_COSMOS_FRACTION <= fraction <= MAX_COSMOS_FRACTION:
            raise RosieGateError(
                f"Cosmos acceptance for set {set_name} is {fraction:.4f}; "
                f"expected {MIN_COSMOS_FRACTION:.2f}-{MAX_COSMOS_FRACTION:.2f}"
            )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args()
    verify_production_manifest(args.metadata)
    print(f"Cosmos production gate passed: {args.metadata.resolve()}")


if __name__ == "__main__":
    main()
