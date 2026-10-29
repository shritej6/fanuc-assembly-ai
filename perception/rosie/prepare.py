"""Freeze validated T13 manifests and export detector/classifier datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from perception.coco import export_coco
from perception.crops import export_crops
from perception.manifest import ManifestConfig, build_manifests

DETECTOR_SETS = ("A", "F", "G")
CLASSIFIER_SETS = ("B", "H", "K")
# Test-bed sets (user 2026-10-28: train only on these): parts stand as in their product, fallen ones are labelled.
# Cosmos restyled one camera of half the scenes in N/R/Q (measured 0.25 of images) and half the wrist images in O.
TESTBED = {
    "detector_sets": ("N", "R", "Q"),
    "classifier_sets": ("O",),
    "cosmos_ranges": (("N", 0.20, 0.30), ("R", 0.20, 0.30), ("Q", 0.20, 0.30), ("O", 0.45, 0.55)),
}
# Set P (insertion alignment from the wrist): its own manifest, the wrist classifier learns the six build states.
TESTBED_BUILD = {
    "detector_sets": (),
    "classifier_sets": ("P",),
    "cosmos_ranges": (("P", 0.45, 0.55),),
}
PROFILES = ("legacy", "testbed", "testbed_build")


def preparation_plan(
    data_root: Path,
    output_root: Path,
    taxonomy_path: Path,
    smoke: bool,
    profile: str = "legacy",
) -> dict[str, Any]:
    """Return the exact immutable-data policy before touching source data."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {PROFILES}")
    root = output_root.resolve()
    sets = {"testbed": TESTBED, "testbed_build": TESTBED_BUILD}.get(profile)
    detector_sets = sets["detector_sets"] if sets else DETECTOR_SETS
    classifier_sets = sets["classifier_sets"] if sets else CLASSIFIER_SETS
    return {
        "profile": profile,
        "fallen_classes": profile == "testbed",
        "classifier_target": "build_state" if profile == "testbed_build" else "identity",
        "cosmos_ranges": [list(item) for item in sets["cosmos_ranges"]] if sets else [],
        "mode": "smoke" if smoke else "full",
        "data_root": str(data_root.resolve()),
        "output_root": str(root),
        "manifest_dir": str(root / "manifests"),
        "coco_dir": str(root / "coco"),
        "crops_dir": str(root / "crops"),
        "taxonomy_path": str(taxonomy_path.resolve()),
        "detector_sets": list(detector_sets[:1]) if smoke else list(detector_sets),
        "classifier_sets": list(classifier_sets[:1]) if smoke else list(classifier_sets),
        "require_cosmos": not smoke,
        "include_cosmos": not smoke,
        "max_scenes_per_set": 128 if smoke else None,
    }


def _manifest_config(plan: dict[str, Any]) -> ManifestConfig:
    return ManifestConfig(
        data_root=Path(plan["data_root"]),
        output_dir=Path(plan["manifest_dir"]),
        taxonomy_path=Path(plan["taxonomy_path"]),
        detector_sets=tuple(plan["detector_sets"]),
        classifier_sets=tuple(plan["classifier_sets"]),
        require_cosmos=plan["require_cosmos"],
        include_cosmos=plan["include_cosmos"],
        max_scenes_per_set=plan["max_scenes_per_set"],
        profile=plan.get("profile", "legacy"),
        classifier_target=plan.get("classifier_target", "identity"),
        fallen_classes=plan.get("fallen_classes", False),
        cosmos_ranges=tuple((name, low, high) for name, low, high in plan.get("cosmos_ranges", [])),
    )


def execute_preparation(plan: dict[str, Any]) -> dict[str, Any]:
    """Build frozen manifests, COCO JSON, and deterministic wrist crops."""
    summary = build_manifests(_manifest_config(plan))
    metadata = json.loads(summary.metadata_path.read_text())
    labels = tuple(metadata["labels"]["detector"])
    coco = export_coco(summary.detector_path, Path(plan["coco_dir"]), labels)
    crops = export_crops(summary.classifier_path, Path(plan["crops_dir"]))
    return {
        "metadata": str(summary.metadata_path),
        "detector_records": summary.detector_records,
        "classifier_records": summary.classifier_records,
        "detector_sha256": summary.detector_sha256,
        "classifier_sha256": summary.classifier_sha256,
        "coco": {name: str(path) for name, path in coco.items()},
        "crops": str(crops),
        "skipped_classifier_targets": summary.skipped_classifier_targets,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--profile", choices=PROFILES, default="legacy")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = preparation_plan(args.data_root, args.output_root, args.taxonomy, args.smoke, args.profile)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.dry_run:
        result = execute_preparation(plan)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
