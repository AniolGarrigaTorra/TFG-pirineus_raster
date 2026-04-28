from pathlib import Path

import rasterio


def validate_raster_matches_grid(
    raster_path: Path,
    grid_path: Path,
) -> None:
    """
    Ensure an output feature raster is perfectly aligned with the project grid.
    """
    with rasterio.open(raster_path) as raster, rasterio.open(grid_path) as grid:
        errors = []

        if raster.crs != grid.crs:
            errors.append(f"CRS mismatch: raster={raster.crs}, grid={grid.crs}")

        if raster.transform != grid.transform:
            errors.append(
                f"Transform mismatch:\n"
                f"  raster={raster.transform}\n"
                f"  grid={grid.transform}"
            )

        if raster.width != grid.width:
            errors.append(f"Width mismatch: raster={raster.width}, grid={grid.width}")

        if raster.height != grid.height:
            errors.append(f"Height mismatch: raster={raster.height}, grid={grid.height}")

        if errors:
            joined = "\n".join(errors)
            raise ValueError(
                f"Output raster does not match grid.\n"
                f"Raster: {raster_path}\n"
                f"Grid: {grid_path}\n"
                f"{joined}"
            )