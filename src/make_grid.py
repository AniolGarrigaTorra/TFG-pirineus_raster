from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.io.config import load_yaml

def main():
    #load config
    project_cfg = load_yaml("configs/project.yaml")
    aoi_cfg = load_yaml("configs/aoi/experimental_pallars_sobira.yaml")

    crs = project_cfg["crs"]
    resolution = project_cfg["resolution_m"]
    nodata = project_cfg["nodata"]

    #load bounds
    bounds = aoi_cfg["bounds"]
    xmin = bounds["xmin"]
    xmax = bounds["xmax"]
    ymin = bounds["ymin"]
    ymax = bounds["ymax"]

    width_m = xmax - xmin
    height_m = ymax - ymin

    #Safe checks
    if width_m%resolution != 0 or height_m%0 !=0:
        raise ValueError(
            f"AOI bounds are not divisible by resolution {resolution} m: "
            f"width={width_m}, height={height_m}"
        )
    
    aoi_crs = aoi_cfg["crs"]
    
    if aoi_crs != crs:
        raise ValueError(f"AOI CRS ({aoi_crs}) does not match project CRS ({crs})")
        
    #Params
    width = width_m // resolution
    height = height_m // resolution

    transform = from_origin(xmin, ymax, resolution, resolution)

    grid_array = np.zeros((height, width), dtype=np.float32)

    #Output path and final name
    interim_dir = Path(project_cfg["paths"]["interim_dir"])
    grid_subdir = project_cfg["grid"]["subdir"]
    out_dir = interim_dir / grid_subdir

    aoi_name = aoi_cfg["name"]
    resolution_suffix = project_cfg["naming"]["resolution_suffix"]
    out_path = out_dir / f"grid_base_{aoi_name}_{resolution_suffix}.tif"

    #Write
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

    print("Grid created:")
    print(f"  Path: {out_path}")
    print(f"  AOI: {aoi_name}")
    print(f"  CRS: {crs}")
    print(f"  Resolution: {resolution} m")
    print(f"  Width: {width} pixels")
    print(f"  Height: {height} pixels")
    print(f"  Bounds: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}")
    print(f"  Area (km²): {(width * resolution) * (height * resolution) / 1_000_000:.3f}")


if __name__ == "__main__":
    main()