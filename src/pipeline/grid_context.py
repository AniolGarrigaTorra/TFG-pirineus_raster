from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

from src.io.paths import get_grid_path


@dataclass(frozen=True, slots=True)
class GridContext:
    """
    Immutable description of the target project grid.

    All output feature rasters must match this grid exactly.
    """

    path: Path
    profile: dict[str, Any]
    transform: Affine
    crs: CRS
    height: int
    width: int
    resolution_m: int
    aoi_name: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


def load_grid_context(
    project_cfg: dict,
    aoi_cfg: dict,
    resolution_m: int,
) -> GridContext:
    """
    Load target grid metadata from the project grid raster.
    """
    aoi_name = aoi_cfg["name"]

    grid_path = get_grid_path(
        project_cfg=project_cfg,
        aoi_cfg=aoi_cfg,
        resolution_m=resolution_m,
    )

    if not grid_path.exists():
        raise FileNotFoundError(
            f"Target grid does not exist: {grid_path}\n"
            f"Create it first with:\n"
            f"  python -m src.make_grid "
            f"--project-config configs/project.yaml "
            f"--aoi-config <aoi_config> "
            f"--resolution {resolution_m}"
        )

    with rasterio.open(grid_path) as grid:
        profile = grid.profile.copy()

        return GridContext(
            path=grid_path,
            profile=profile,
            transform=grid.transform,
            crs=grid.crs,
            height=grid.height,
            width=grid.width,
            resolution_m=int(resolution_m),
            aoi_name=aoi_name,
        )


def print_grid_context(
    grid: GridContext,
    prefix: str = "[grid]",
) -> None:
    print(f"{prefix} AOI: {grid.aoi_name}")
    print(f"{prefix} Path: {grid.path}")
    print(f"{prefix} CRS: {grid.crs}")
    print(f"{prefix} Shape: {grid.width} x {grid.height}")
    print(f"{prefix} Resolution: {grid.resolution_m} m")