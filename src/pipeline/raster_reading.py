from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from src.pipeline.grid_context import GridContext


def read_raster_to_grid(
    raster_path: Path,
    grid: GridContext,
    resampling: Resampling,
    band: int = 1,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Read one raster band and align it to the target project grid.

    Returns a float32 array with np.nan as internal nodata.
    Scale factor is applied after reprojection.
    """
    raster_path = Path(raster_path)

    if not raster_path.exists():
        raise FileNotFoundError(f"Input raster does not exist: {raster_path}")

    dst = np.full(
        grid.shape,
        np.nan,
        dtype=np.float32,
    )

    with rasterio.open(raster_path) as src:
        src_array = src.read(band).astype(np.float32)
        src_nodata = src.nodata

        if src_nodata is not None:
            src_array = np.where(src_array == src_nodata, np.nan, src_array)

        reproject(
            source=src_array,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            dst_nodata=np.nan,
            resampling=resampling,
        )

    if scale_factor != 1.0:
        dst = dst * float(scale_factor)

    dst = dst.astype(np.float32)
    dst[~np.isfinite(dst)] = np.nan

    return dst


def stack_rasters_to_grid(
    raster_paths: list[Path],
    grid: GridContext,
    resampling: Resampling,
    scale_factor: float = 1.0,
    band: int = 1,
) -> np.ndarray:
    """
    Read several rasters and return a stack aligned to the target grid.

    Output shape:
      (n_layers, height, width)
    """
    arrays = [
        read_raster_to_grid(
            raster_path=path,
            grid=grid,
            resampling=resampling,
            band=band,
            scale_factor=scale_factor,
        )
        for path in raster_paths
    ]

    if not arrays:
        raise ValueError("No raster paths provided for stack.")

    return np.stack(arrays, axis=0)