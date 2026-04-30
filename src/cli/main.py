from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline.layers import (
    build_layer_catalog_from_manifest,
    summarize_layer_catalog,
)
from src.pipeline.dataset import run_dataset_pipeline
from src.pipeline.runner import run_source_pipeline
from src.sources.registry import list_source_connectors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pirineus-raster",
        description=(
            "CLI for generating grid-aligned environmental raster datasets "
            "for the Pyrenees."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run a full dataset recipe from a run_config YAML.",
    )
    run_parser.add_argument(
        "run_config",
        help="Path to a run config YAML, e.g. configs/runs/pallars_worldclim_100m.yaml",
    )

    run_source_parser = subparsers.add_parser(
        "run-source",
        help="Developer command: run one source pipeline stage directly.",
    )
    run_source_parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to the project config YAML.",
    )
    run_source_parser.add_argument(
        "--source-config",
        required=True,
        help="Path to a source config YAML.",
    )
    run_source_parser.add_argument(
        "--stage",
        choices=["download", "clip", "build", "all"],
        default="build",
        help="Source pipeline stage to run.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a generated dataset manifest.",
    )
    inspect_parser.add_argument(
        "dataset_dir",
        help="Path to a generated dataset directory.",
    )

    subparsers.add_parser(
        "list-sources",
        help="List available source providers.",
    )

    return parser


def _load_manifest(dataset_dir: str | Path) -> dict:
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "metadata" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_dataset(dataset_dir: str | Path) -> None:
    manifest = _load_manifest(dataset_dir)

    if "layer_catalog" in manifest:
        layers = build_layer_catalog_from_manifest(manifest)
    else:
        layers = build_layer_catalog_from_manifest(manifest)

    summary = summarize_layer_catalog(layers)

    print("==============================")
    print("Pirineus Raster Dataset")
    print(f"Dataset: {manifest.get('dataset_name')}")
    print(f"Dir:     {manifest.get('dataset_dir')}")
    print("------------------------------")
    print(f"Sources: {manifest.get('n_sources')}")
    print(f"Rasters: {manifest.get('n_rasters')}")
    print(f"Layers:  {summary['n_layers']}")
    print(f"AOIs:    {', '.join(summary['aois']) if summary['aois'] else 'unknown'}")
    print(f"Res:     {summary['resolutions_m']}")
    print("------------------------------")
    print("Providers:")
    for provider in summary["providers"]:
        print(f"  - {provider}")

    print("Products:")
    for product in summary["products"]:
        print(f"  - {product}")

    print("Variables:")
    for variable in summary["variables"]:
        print(f"  - {variable}")

    print("==============================")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        run_dataset_pipeline(
            run_config_path=args.run_config,
        )

    elif args.command == "run-source":
        run_source_pipeline(
            project_config_path=args.project_config,
            source_config_path=args.source_config,
            stage=args.stage,
        )

    elif args.command == "inspect":
        inspect_dataset(args.dataset_dir)

    elif args.command == "list-sources":
        print("Available raster source providers:")
        for provider in list_source_connectors():
            print(f"  - {provider}")

    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()