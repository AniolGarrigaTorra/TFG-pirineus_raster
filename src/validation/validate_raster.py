import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from src.io.config import load_yaml
from src.io.paths import build_resolution_suffix, ensure_dir, get_grid_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raster",
        required=True,
        help="Path to raster file to validate",
    )
    parser.add_argument(
        "--aoi",
        default="configs/aoi/experimental_pallars_sobira.yaml",
        help="Path to AOI config YAML",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Reference grid resolution in meters",
    )
    return parser.parse_args()


def compare_bounds(bounds_a, bounds_b) -> bool:
    return (
        float(bounds_a.left) == float(bounds_b.left)
        and float(bounds_a.right) == float(bounds_b.right)
        and float(bounds_a.bottom) == float(bounds_b.bottom)
        and float(bounds_a.top) == float(bounds_b.top)
    )


def main():
    args = parse_args()

    project_cfg = load_yaml("configs/project.yaml")
    aoi_cfg = load_yaml(args.aoi)

    available_resolutions = project_cfg["grids"]["available_resolutions_m"]
    default_resolution = int(project_cfg["grids"]["default_resolution_m"])
    resolution = args.resolution if args.resolution is not None else default_resolution
    resolution = int(resolution)

    if resolution not in available_resolutions:
        raise ValueError(
            f"Resolution {resolution} m is not listed in project config. "
            f"Available: {available_resolutions}"
        )

    raster_path = Path(args.raster)
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster file not found: {raster_path}")

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution_m=resolution,
    )
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid file not found: {grid_path}")

    with rasterio.open(grid_path) as grid:
        expected_crs = grid.crs.to_string()
        expected_resolution_x = float(grid.transform.a)
        expected_resolution_y = float(abs(grid.transform.e))
        expected_width = grid.width
        expected_height = grid.height
        expected_bounds = grid.bounds
        expected_transform = tuple(grid.transform)
        expected_nodata = float(grid.nodata) if grid.nodata is not None else None

    with rasterio.open(raster_path) as src:
        actual_crs = src.crs.to_string()
        actual_resolution_x = float(src.transform.a)
        actual_resolution_y = float(abs(src.transform.e))
        actual_width = src.width
        actual_height = src.height
        actual_bounds = src.bounds
        actual_transform = tuple(src.transform)
        actual_nodata = float(src.nodata) if src.nodata is not None else None
        arr = src.read(1).astype(np.float32)

    valid_mask = np.ones(arr.shape, dtype=bool)
    if actual_nodata is not None:
        valid_mask = arr != actual_nodata

    valid_values = arr[valid_mask]
    nodata_pixels = int((~valid_mask).sum())
    total_pixels = int(arr.size)
    nodata_pct = (nodata_pixels / total_pixels) * 100.0

    has_valid_data = valid_values.size > 0

    stats = {
        "valid_pixel_count": int(valid_values.size),
        "nodata_pixel_count": nodata_pixels,
        "total_pixel_count": total_pixels,
        "nodata_percentage": nodata_pct,
        "min": float(valid_values.min()) if has_valid_data else None,
        "max": float(valid_values.max()) if has_valid_data else None,
        "mean": float(valid_values.mean()) if has_valid_data else None,
        "std": float(valid_values.std()) if has_valid_data else None,
    }

    checks = {
        "crs_match": actual_crs == expected_crs,
        "resolution_x_match": actual_resolution_x == expected_resolution_x,
        "resolution_y_match": actual_resolution_y == expected_resolution_y,
        "width_match": actual_width == expected_width,
        "height_match": actual_height == expected_height,
        "bounds_match": compare_bounds(actual_bounds, expected_bounds),
        "transform_match": actual_transform == expected_transform,
        "nodata_match": actual_nodata == expected_nodata,
        "has_valid_data": has_valid_data,
    }

    validation_results = {
        "raster_path": str(raster_path),
        "grid_path": str(grid_path),
        "resolution_m": resolution,
        "checks": checks,
        "stats": stats,
        "summary": {
            "actual_crs": actual_crs,
            "expected_crs": expected_crs,
            "actual_resolution_x": actual_resolution_x,
            "actual_resolution_y": actual_resolution_y,
            "expected_resolution_x": expected_resolution_x,
            "expected_resolution_y": expected_resolution_y,
            "actual_width": actual_width,
            "expected_width": expected_width,
            "actual_height": actual_height,
            "expected_height": expected_height,
            "actual_nodata": actual_nodata,
            "expected_nodata": expected_nodata,
            "actual_bounds": {
                "xmin": float(actual_bounds.left),
                "xmax": float(actual_bounds.right),
                "ymin": float(actual_bounds.bottom),
                "ymax": float(actual_bounds.top),
            },
            "expected_bounds": {
                "xmin": float(expected_bounds.left),
                "xmax": float(expected_bounds.right),
                "ymin": float(expected_bounds.bottom),
                "ymax": float(expected_bounds.top),
            },
        },
    }

    all_ok = all(checks.values())

    validation_dir = Path(project_cfg["validation"]["output_dir"]) / "rasters" / build_resolution_suffix(resolution)
    ensure_dir(validation_dir)

    out_json = validation_dir / f"{raster_path.stem}_validation.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(validation_results, f, indent=2)

    print("Raster validation finished")
    print(f"  Raster: {raster_path}")
    print(f"  Report: {out_json}")
    print(f"  Status: {'PASS' if all_ok else 'FAIL'}")
    print(f"  Valid pixels: {stats['valid_pixel_count']}")
    print(f"  NoData %: {stats['nodata_percentage']:.2f}")

    if not all_ok:
        failed_checks = [k for k, v in checks.items() if not v]
        raise ValueError(f"Raster validation failed: {failed_checks}")


if __name__ == "__main__":
    main()