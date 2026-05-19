import argparse
import json
from pathlib import Path

import rasterio

from src.io.config import load_yaml, resolve_path
from src.io.paths import build_resolution_suffix, ensure_dir, get_grid_path, get_project_base_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to project config YAML.",
    )
    parser.add_argument(
        "--aoi-config",
        default=None,
        help="Path to AOI config YAML. Recommended interface.",
    )
    parser.add_argument(
        "--aoi",
        default=None,
        help=(
            "AOI name or AOI config path. Backwards-compatible shortcut. "
            "Examples: experimental_pallars_sobira or configs/aoi/experimental_pallars_sobira.yaml"
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Grid resolution in meters",
    )
    return parser.parse_args()


def resolve_aoi_config_path(args) -> Path:
    if args.aoi_config is not None:
        return resolve_path(args.aoi_config, must_exist=True)

    if args.aoi is not None:
        aoi = Path(args.aoi)
        if aoi.suffix in {".yaml", ".yml"}:
            return resolve_path(aoi, must_exist=True)
        return resolve_path(Path("configs/aoi") / f"{args.aoi}.yaml", must_exist=True)

    return resolve_path("configs/aoi/experimental_pallars_sobira.yaml", must_exist=True)


def main():
    args = parse_args()

    project_config_path = resolve_path(args.project_config, must_exist=True)
    aoi_config_path = resolve_aoi_config_path(args)

    project_cfg = load_yaml(project_config_path)
    project_cfg["_config_path"] = str(project_config_path)
    aoi_cfg = load_yaml(aoi_config_path)

    available_resolutions = project_cfg["grids"]["available_resolutions_m"]
    default_resolution = int(project_cfg["grids"]["default_resolution_m"])
    resolution = args.resolution if args.resolution is not None else default_resolution
    resolution = int(resolution)

    if resolution not in available_resolutions:
        raise ValueError(
            f"Resolution {resolution} m is not listed in project config. "
            f"Available: {available_resolutions}"
        )

    expected_crs = project_cfg["crs"]
    expected_nodata = float(project_cfg["nodata"])
    expected_resolution = float(resolution)

    bounds_cfg = aoi_cfg["bounds"]
    expected_xmin = float(bounds_cfg["xmin"])
    expected_xmax = float(bounds_cfg["xmax"])
    expected_ymin = float(bounds_cfg["ymin"])
    expected_ymax = float(bounds_cfg["ymax"])

    expected_width = int((expected_xmax - expected_xmin) / expected_resolution)
    expected_height = int((expected_ymax - expected_ymin) / expected_resolution)

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution_m=resolution,
    )
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    validation_results = {
        "grid_path": str(grid_path),
        "resolution_m": resolution,
        "checks": {},
        "summary": {},
    }

    with rasterio.open(grid_path) as src:
        actual_crs = src.crs.to_string()
        actual_resolution_x = float(src.transform.a)
        actual_resolution_y = float(abs(src.transform.e))
        actual_nodata = float(src.nodata) if src.nodata is not None else None
        actual_width = src.width
        actual_height = src.height
        actual_bounds = src.bounds

        checks = {
            "crs_match": actual_crs == expected_crs,
            "resolution_x_match": actual_resolution_x == expected_resolution,
            "resolution_y_match": actual_resolution_y == expected_resolution,
            "nodata_match": actual_nodata == expected_nodata,
            "width_match": actual_width == expected_width,
            "height_match": actual_height == expected_height,
            "xmin_match": float(actual_bounds.left) == expected_xmin,
            "xmax_match": float(actual_bounds.right) == expected_xmax,
            "ymin_match": float(actual_bounds.bottom) == expected_ymin,
            "ymax_match": float(actual_bounds.top) == expected_ymax,
        }

        validation_results["checks"] = checks
        validation_results["summary"] = {
            "actual_crs": actual_crs,
            "expected_crs": expected_crs,
            "actual_resolution_x": actual_resolution_x,
            "actual_resolution_y": actual_resolution_y,
            "expected_resolution": expected_resolution,
            "actual_nodata": actual_nodata,
            "expected_nodata": expected_nodata,
            "actual_width": actual_width,
            "expected_width": expected_width,
            "actual_height": actual_height,
            "expected_height": expected_height,
            "actual_bounds": {
                "xmin": float(actual_bounds.left),
                "xmax": float(actual_bounds.right),
                "ymin": float(actual_bounds.bottom),
                "ymax": float(actual_bounds.top),
            },
            "expected_bounds": {
                "xmin": expected_xmin,
                "xmax": expected_xmax,
                "ymin": expected_ymin,
                "ymax": expected_ymax,
            },
        }

    all_ok = all(validation_results["checks"].values())

    validation_dir = (
        resolve_path(
            project_cfg["validation"]["output_dir"],
            base_path=get_project_base_dir(project_cfg),
        )
        / "grid"
        / build_resolution_suffix(resolution)
    )
    ensure_dir(validation_dir)

    out_json = validation_dir / f"validate_grid_{aoi_cfg['name']}_{build_resolution_suffix(resolution)}.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    print("Grid validation finished")
    print(f"  Grid: {grid_path}")
    print(f"  Report: {out_json}")
    print(f"  Status: {'PASS' if all_ok else 'FAIL'}")

    if not all_ok:
        failed_checks = [k for k, v in validation_results["checks"].items() if not v]
        raise ValueError(f"Grid validation failed: {failed_checks}")


if __name__ == "__main__":
    main()
