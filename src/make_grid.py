import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.io.config import load_yaml
from src.io.paths import ensure_dir, get_grid_dir, get_grid_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aoi",
        default="configs/aoi/experimental_pallars_sobira.yaml",
        help="Path to AOI config YAML",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help="Target grid resolution in meters",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    project_cfg = load_yaml("configs/project.yaml")
    aoi_cfg = load_yaml(args.aoi)

    crs = project_cfg["crs"]
    nodata = float(project_cfg["nodata"])

    available_resolutions = project_cfg["grids"]["available_resolutions_m"]
    default_resolution = int(project_cfg["grids"]["default_resolution_m"])

    resolution = args.resolution if args.resolution is not None else default_resolution
    resolution = int(resolution)

    if resolution not in available_resolutions:
        raise ValueError(
            f"Resolution {resolution} m is not listed in project config. "
            f"Available: {available_resolutions}"
        )

    aoi_crs = aoi_cfg["crs"]
    if aoi_crs != crs:
        raise ValueError(f"AOI CRS ({aoi_crs}) does not match project CRS ({crs})")

    bounds = aoi_cfg["bounds"]
    xmin = int(bounds["xmin"])
    xmax = int(bounds["xmax"])
    ymin = int(bounds["ymin"])
    ymax = int(bounds["ymax"])

    width_m = xmax - xmin
    height_m = ymax - ymin

    if width_m % resolution != 0 or height_m % resolution != 0:
        raise ValueError(
            f"AOI bounds are not divisible by resolution {resolution} m: "
            f"width={width_m}, height={height_m}"
        )

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

    print("Grid created successfully")
    print(f"  Path: {out_path}")
    print(f"  AOI: {aoi_cfg['name']}")
    print(f"  CRS: {crs}")
    print(f"  Resolution: {resolution} m")
    print(f"  Width: {width} pixels")
    print(f"  Height: {height} pixels")
    print(f"  Bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    print(f"  Area (km²): {(width * resolution) * (height * resolution) / 1_000_000:.3f}")


if __name__ == "__main__":
    main()