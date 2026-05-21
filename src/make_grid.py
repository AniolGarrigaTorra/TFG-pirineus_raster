import argparse
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.transform import from_origin

from src.io.config import load_yaml, resolve_path
from src.io.paths import ensure_dir, get_grid_dir, get_grid_path
from src.pipeline.project_overrides import normalize_crs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a project-aligned empty raster grid for a given AOI and resolution."
    )

    parser.add_argument(
        "--project-config",
        default="configs/project.yaml",
        help="Path to project config YAML.",
    )

    parser.add_argument(
        "--aoi-config",
        default=None,
        help="Path to AOI config YAML. Recommended new interface.",
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
        help="Target grid resolution in meters. If omitted, project default is used.",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="Optional output CRS override, e.g. EPSG:3035 or EPSG:25831.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing grid if it already exists.",
    )

    return parser.parse_args()


def resolve_aoi_config_path(args) -> Path:
    """
    Resolve AOI config path from either the new --aoi-config argument
    or the legacy --aoi argument.

    Priority:
      1. --aoi-config
      2. --aoi as path
      3. --aoi as name under configs/aoi/
      4. default experimental_pallars_sobira
    """
    if args.aoi_config is not None:
        return resolve_path(args.aoi_config, must_exist=True)

    if args.aoi is not None:
        aoi = Path(args.aoi)

        # If user passed a YAML path directly.
        if aoi.suffix in {".yaml", ".yml"}:
            return resolve_path(aoi, must_exist=True)

        # If user passed only the AOI name.
        return resolve_path(Path("configs/aoi") / f"{args.aoi}.yaml", must_exist=True)

    return resolve_path("configs/aoi/experimental_pallars_sobira.yaml", must_exist=True)


def validate_grid_inputs(
    project_cfg: dict,
    aoi_cfg: dict,
    resolution: int,
) -> None:
    crs = project_cfg["crs"]
    available_resolutions = project_cfg["grids"]["available_resolutions_m"]

    if resolution not in available_resolutions:
        raise ValueError(
            f"Resolution {resolution} m is not listed in project config. "
            f"Available: {available_resolutions}"
        )

    bounds = aoi_cfg["bounds"]
    xmin = int(bounds["xmin"])
    xmax = int(bounds["xmax"])
    ymin = int(bounds["ymin"])
    ymax = int(bounds["ymax"])

    width_m = xmax - xmin
    height_m = ymax - ymin

    if width_m <= 0 or height_m <= 0:
        raise ValueError(
            f"Invalid AOI bounds: xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax}"
        )

    if width_m % resolution != 0 or height_m % resolution != 0:
        raise ValueError(
            f"AOI bounds are not divisible by resolution {resolution} m: "
            f"width={width_m}, height={height_m}. "
            "Adjust AOI bounds so the grid aligns exactly."
        )


def _aoi_bounds_in_target_crs(aoi_cfg: dict, target_crs: str) -> dict[str, float]:
    bounds = aoi_cfg["bounds"]
    aoi_crs = aoi_cfg["crs"]

    if CRS.from_user_input(aoi_crs) == CRS.from_user_input(target_crs):
        return {
            "xmin": float(bounds["xmin"]),
            "ymin": float(bounds["ymin"]),
            "xmax": float(bounds["xmax"]),
            "ymax": float(bounds["ymax"]),
        }

    transformer = Transformer.from_crs(aoi_crs, target_crs, always_xy=True)
    xmin, ymin, xmax, ymax = transformer.transform_bounds(
        float(bounds["xmin"]),
        float(bounds["ymin"]),
        float(bounds["xmax"]),
        float(bounds["ymax"]),
        densify_pts=21,
    )
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def _snap_bounds_to_resolution(
    bounds: dict[str, float],
    resolution: int,
) -> dict[str, int]:
    xmin = int(np.floor(float(bounds["xmin"]) / resolution) * resolution)
    ymin = int(np.floor(float(bounds["ymin"]) / resolution) * resolution)
    xmax = int(np.ceil(float(bounds["xmax"]) / resolution) * resolution)
    ymax = int(np.ceil(float(bounds["ymax"]) / resolution) * resolution)
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def create_grid(
    project_cfg: dict,
    aoi_cfg: dict,
    resolution: int,
    overwrite: bool = False,
) -> Path:
    crs = project_cfg["crs"]
    nodata = float(project_cfg["nodata"])

    validate_grid_inputs(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution=resolution,
    )

    bounds = _snap_bounds_to_resolution(
        _aoi_bounds_in_target_crs(aoi_cfg, crs),
        resolution=resolution,
    )
    xmin = int(bounds["xmin"])
    xmax = int(bounds["xmax"])
    ymin = int(bounds["ymin"])
    ymax = int(bounds["ymax"])

    width_m = xmax - xmin
    height_m = ymax - ymin

    width = width_m // resolution
    height = height_m // resolution

    transform = from_origin(xmin, ymax, resolution, resolution)
    grid_array = np.zeros((height, width), dtype=np.float32)

    grid_dir = get_grid_dir(project_cfg)
    ensure_dir(grid_dir)

    out_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution_m=resolution,
    )

    if out_path.exists() and not overwrite:
        print("Grid already exists, skipping")
        print(f"  Path: {out_path}")
        print("  Use --overwrite to recreate it.")
        return out_path

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=grid_array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        dst.write(grid_array, 1)

        dst.update_tags(
            GRID_TYPE="project_alignment_grid",
            AOI_NAME=aoi_cfg["name"],
            CRS=crs,
            AOI_SOURCE_CRS=aoi_cfg.get("crs"),
            RESOLUTION_M=str(resolution),
            NODATA=str(nodata),
            BOUNDS=f"xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}",
            NOTE=(
                "Reference grid used to align all raster features. "
                "Values are placeholders; geometry, CRS, transform and shape are the important parts."
            ),
        )

    print("Grid created successfully")
    print(f"  Path: {out_path}")
    print(f"  AOI: {aoi_cfg['name']}")
    print(f"  CRS: {crs}")
    print(f"  Resolution: {resolution} m")
    print(f"  Width: {width} pixels")
    print(f"  Height: {height} pixels")
    print(f"  Bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    print(f"  Area (km²): {(width * resolution) * (height * resolution) / 1_000_000:.3f}")

    return out_path


def main():
    args = parse_args()

    project_config_path = resolve_path(args.project_config, must_exist=True)
    aoi_config_path = resolve_aoi_config_path(args)

    if not project_config_path.exists():
        raise FileNotFoundError(f"Project config not found: {project_config_path}")

    if not aoi_config_path.exists():
        raise FileNotFoundError(f"AOI config not found: {aoi_config_path}")

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


if __name__ == "__main__":
    main()
