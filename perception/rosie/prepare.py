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


def preparation_plan(
    data_root: Path,
    output_root: Path,
    taxonomy_path: Path,
    smoke: bool,
) -> dict[str, Any]:
    """Return the exact immutable-data policy before touching source data."""
    root = output_root.resolve()
    return {
        "mode": "smoke" if smoke else "full",
        "data_root": str(data_root.resolve()),
        "output_root": str(root),
        "manifest_dir": str(root / "manifests"),
        "coco_dir": str(root / "coco"),
        "crops_dir": str(root / "crops"),
        "taxonomy_path": str(taxonomy_path.resolve()),
        "detector_sets": ["A"] if smoke else list(DETECTOR_SETS),
        "classifier_sets": ["B"] if smoke else list(CLASSIFIER_SETS),
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
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    plan = preparation_plan(args.data_root, args.output_root, args.taxonomy, args.smoke)
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.dry_run:
        result = execute_preparation(plan)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
