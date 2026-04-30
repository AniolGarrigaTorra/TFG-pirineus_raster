from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from src.pipeline.grid_context import GridContext
from src.pipeline.metadata import metadata_to_geotiff_tags, write_sidecar_json
from src.pipeline.validation import validate_raster_matches_grid


def build_output_profile(
    grid: GridContext,
    output_dtype: str = "float32",
    nodata: float = -9999.0,
    compression: str = "LZW",
) -> dict[str, Any]:
    """
    Build a GeoTIFF output profile matching the target grid.
    """
    profile = grid.profile.copy()

    profile.update(
        driver="GTiff",
        height=grid.height,
        width=grid.width,
        count=1,
        dtype=output_dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
        compress=compression,
    )

    return profile


def prepare_array_for_write(
    array: np.ndarray,
    nodata: float = -9999.0,
    output_dtype: str = "float32",
) -> np.ndarray:
    """
    Convert an internal float array with np.nan into a writable raster array.
    """
    if array.ndim != 2:
        raise ValueError(
            f"Expected 2D array for raster writing, got shape {array.shape}"
        )

    out = array.astype(np.float32, copy=True)
    out[~np.isfinite(out)] = nodata

    return out.astype(output_dtype)


def write_feature_raster(
    output_path: Path,
    array: np.ndarray,
    grid: GridContext,
    metadata: dict[str, Any],
    output_dtype: str = "float32",
    nodata: float = -9999.0,
    compression: str = "LZW",
    write_sidecar: bool = True,
    validate: bool = True,
) -> Path:
    """
    Write one feature raster aligned to the project grid.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_profile = build_output_profile(
        grid=grid,
        output_dtype=output_dtype,
        nodata=nodata,
        compression=compression,
    )

    writable = prepare_array_for_write(
        array=array,
        nodata=nodata,
        output_dtype=output_dtype,
    )

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(writable, 1)
        dst.update_tags(**metadata_to_geotiff_tags(metadata))

    if write_sidecar:
        write_sidecar_json(metadata, output_path)

    if validate:
        validate_raster_matches_grid(
            raster_path=output_path,
            grid_path=grid.path,
        )

    return output_path