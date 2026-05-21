from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.io.config import resolve_path
from src.workbench.api import serve_workbench_api
from src.workbench.catalog import source_catalog_from_config, workbench_catalog
from src.workbench.compiler import (
    load_and_compile_run_config,
    render_run_config_yaml,
    validate_researcher_run_config,
)


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

    make_grid_parser = subparsers.add_parser(
        "make-grid",
        help="Create a project-aligned reference grid for an AOI and resolution.",
    )
    make_grid_parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to the project config YAML.",
    )
    make_grid_parser.add_argument(
        "--aoi-config",
        default=None,
        help="Path to the AOI config YAML.",
    )
    make_grid_parser.add_argument(
        "--aoi",
        default=None,
        help="AOI name under configs/aoi/ or a direct AOI YAML path.",
    )
    make_grid_parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Target grid resolution in meters. Uses project default if omitted.",
    )
    make_grid_parser.add_argument(
        "--crs",
        default=None,
        help="Optional output CRS override, e.g. EPSG:3035 or EPSG:25831.",
    )
    make_grid_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing grid.",
    )

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a generated dataset manifest.",
    )
    inspect_parser.add_argument(
        "dataset_dir",
        help="Path to a generated dataset directory.",
    )

    validate_parser = subparsers.add_parser(
        "validate-dataset",
        help="Validate a generated dataset against its manifest and reference grid.",
    )
    validate_parser.add_argument(
        "dataset_dir",
        help="Path to a generated dataset directory.",
    )
    validate_parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Fail if sidecar JSON files are missing standardized metadata keys.",
    )

    validate_config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate a run YAML, including simplified workbench selections.",
    )
    validate_config_parser.add_argument(
        "run_config",
        help="Path to a run config YAML.",
    )

    render_parser = subparsers.add_parser(
        "render-run",
        help="Render a run YAML after expanding workbench convenience blocks.",
    )
    render_parser.add_argument(
        "run_config",
        help="Path to a run config YAML.",
    )
    render_parser.add_argument(
        "--output",
        default=None,
        help="Optional output YAML path. Prints to stdout if omitted.",
    )

    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Print project/source catalog information as JSON.",
    )
    catalog_parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to the project config YAML.",
    )
    catalog_parser.add_argument(
        "--source-config",
        default=None,
        help="Optional source config YAML. If omitted, prints the full workbench catalog.",
    )

    api_parser = subparsers.add_parser(
        "serve-config-api",
        help="Serve the local JSON API used by the React configuration workbench.",
    )
    api_parser.add_argument("--host", default="127.0.0.1")
    api_parser.add_argument("--port", default=8765, type=int)
    api_parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to the project config YAML.",
    )

    ui_parser = subparsers.add_parser(
        "serve-ui",
        help="Start both the config API and the React workbench with one command.",
    )
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--api-port", default=8765, type=int)
    ui_parser.add_argument("--ui-port", default=5173, type=int)
    ui_parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to the project config YAML.",
    )
    ui_parser.add_argument(
        "--ui-dir",
        default="ui",
        help="Path to the React workbench directory.",
    )

    credentials_parser = subparsers.add_parser(
        "check-credentials",
        help="Check WEkEO/HDA credentials and HDA package availability.",
    )
    credentials_parser.add_argument(
        "--setup",
        action="store_true",
        help="Interactively write ~/.hdarc before checking credentials.",
    )
    credentials_parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing ~/.hdarc when used with --setup.",
    )

    subparsers.add_parser(
        "list-sources",
        help="List available source providers.",
    )

    return parser


def _load_manifest(dataset_dir: str | Path) -> dict:
    dataset_dir = resolve_path(dataset_dir, must_exist=True)
    manifest_path = dataset_dir / "metadata" / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["_resolved_dataset_dir"] = str(dataset_dir)
    return manifest


def inspect_dataset(dataset_dir: str | Path) -> None:
    from src.pipeline.layers import (
        build_layer_catalog_from_manifest,
        summarize_layer_catalog,
    )

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
        from src.pipeline.dataset import run_dataset_pipeline

        run_dataset_pipeline(
            run_config_path=args.run_config,
        )

    elif args.command == "run-source":
        from src.pipeline.runner import run_source_pipeline

        run_source_pipeline(
            project_config_path=args.project_config,
            source_config_path=args.source_config,
            stage=args.stage,
        )

    elif args.command == "make-grid":
        from src.io.config import load_yaml
        from src.make_grid import create_grid, resolve_aoi_config_path
        from src.pipeline.project_overrides import normalize_crs

        project_config_path = resolve_path(args.project_config, must_exist=True)
        aoi_config_path = resolve_aoi_config_path(args)

        project_cfg = load_yaml(project_config_path)
        project_cfg["_config_path"] = str(project_config_path)
        if args.crs:
            normalized_crs = normalize_crs(args.crs)
            if normalized_crs != project_cfg.get("crs"):
                project_cfg["_default_crs"] = project_cfg.get("crs")
                project_cfg["_crs_overridden"] = True
                project_cfg["_grid_crs_suffix"] = normalized_crs.lower().replace(":", "")
            project_cfg["crs"] = normalized_crs
        aoi_cfg = load_yaml(aoi_config_path)

        default_resolution = int(project_cfg["grids"]["default_resolution_m"])
        resolution = int(args.resolution) if args.resolution is not None else default_resolution

        print("==============================")
        print("Create project grid")
        print(f"Project config: {project_config_path}")
        print(f"AOI config: {aoi_config_path}")
        print(f"AOI name: {aoi_cfg['name']}")
        print(f"Resolution: {resolution} m")
        print("==============================")

        create_grid(
            project_cfg=project_cfg,
            aoi_cfg=aoi_cfg,
            resolution=resolution,
            overwrite=args.overwrite,
        )

    elif args.command == "inspect":
        inspect_dataset(args.dataset_dir)

    elif args.command == "validate-dataset":
        from src.validation.validate_dataset import validate_dataset_dir

        report = validate_dataset_dir(
            dataset_dir=args.dataset_dir,
            strict_metadata=args.strict_metadata,
            write_report=True,
        )
        print("==============================")
        print("Pirineus Raster Dataset Validation")
        print(f"Dataset: {report['dataset_dir']}")
        print(f"Grid:    {report['grid_path']}")
        print(f"Rasters: {report['n_rasters']}")
        print(f"Failed:  {report['n_failed']}")
        print(f"Warned:  {report['n_warned']}")
        print(f"Report:  {report['dataset_dir']}/metadata/validation_report.json")
        print(f"Status:  {'PASS' if report['ok'] else 'FAIL'}")
        print("==============================")

        if not report["ok"]:
            failed = [
                item
                for item in report["rasters"]
                if not item.get("ok", False)
            ]
            for item in failed[:10]:
                print(f"  - {item.get('name')}: {item.get('errors')}")
            raise SystemExit(1)

    elif args.command == "validate-config":
        run_config_path = resolve_path(args.run_config, must_exist=True)
        run_cfg = load_and_compile_run_config(run_config_path)
        report = validate_researcher_run_config(
            run_cfg=run_cfg,
            run_config_path=run_config_path,
        )
        print("==============================")
        print("Pirineus Raster Config Validation")
        print(f"Run config:       {run_config_path}")
        print(f"Status:           {'PASS' if report['ok'] else 'FAIL'}")
        print(f"Estimated layers: {report['estimated_layers']}")
        print(f"  Sources:        {report.get('estimated_source_layers', 0)}")
        print(f"  Derived:        {report.get('estimated_derived_layers', 0)}")
        print("==============================")

        for source in report["sources"]:
            print(
                f"  - {source['id']}: {source['estimated_layers']} layers "
                f"({source.get('provider')}/{source.get('product')})"
            )

        for warning in report["warnings"]:
            print(f"WARNING: {warning}")

        for error in report["errors"]:
            print(f"ERROR: {error}")

        if not report["ok"]:
            raise SystemExit(1)

    elif args.command == "render-run":
        run_config_path = resolve_path(args.run_config, must_exist=True)
        run_cfg = load_and_compile_run_config(run_config_path)
        yaml_text = render_run_config_yaml(run_cfg, compile_groups=False)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(yaml_text, encoding="utf-8")
            print(f"Rendered run config: {output_path}")
        else:
            print(yaml_text)

    elif args.command == "catalog":
        if args.source_config:
            payload = source_catalog_from_config(args.source_config)
        else:
            payload = workbench_catalog(args.project_config)
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))

    elif args.command == "serve-config-api":
        serve_workbench_api(
            host=args.host,
            port=args.port,
            project_config_path=args.project_config,
        )

    elif args.command == "serve-ui":
        from src.cli.ui import serve_ui

        serve_ui(
            host=args.host,
            api_port=args.api_port,
            ui_port=args.ui_port,
            project_config_path=args.project_config,
            ui_dir=args.ui_dir,
        )

    elif args.command == "check-credentials":
        from src.cli.credentials import print_credentials_report

        print_credentials_report(
            setup=args.setup,
            force=args.force,
        )

    elif args.command == "list-sources":
        from src.sources.registry import list_source_connectors

        print("Available raster source providers:")
        for provider in list_source_connectors():
            print(f"  - {provider}")

    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
