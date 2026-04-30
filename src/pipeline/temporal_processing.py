from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from rasterio.enums import Resampling

from src.pipeline.aggregation import aggregate_stack
from src.pipeline.grid_context import GridContext
from src.pipeline.raster_reading import read_raster_to_grid


@dataclass(frozen=True, slots=True)
class TemporalRasterSpec:
    """
    Generic description of one temporal raster slice.

    It can represent:
      - one monthly GeoTIFF
      - one year-month GeoTIFF
      - one extracted monthly band from a multiband file
    """

    path: Path
    month: int
    year: int | None = None
    band: int = 1


def read_temporal_stack_to_grid(
    specs: list[TemporalRasterSpec],
    grid: GridContext,
    resampling: Resampling,
    scale_factor: float = 1.0,
) -> np.ndarray:
    """
    Read a list of temporal raster specs into one aligned stack.

    Output shape:
      (n_slices, height, width)
    """
    if not specs:
        raise ValueError("No temporal raster specs provided.")

    arrays: list[np.ndarray] = []

    for spec in specs:
        if not spec.path.exists():
            raise FileNotFoundError(f"Missing temporal raster: {spec.path}")

        array = read_raster_to_grid(
            raster_path=spec.path,
            grid=grid,
            resampling=resampling,
            band=spec.band,
            scale_factor=scale_factor,
        )
        arrays.append(array)

    return np.stack(arrays, axis=0)


def aggregate_temporal_specs(
    specs: list[TemporalRasterSpec],
    grid: GridContext,
    resampling: Resampling,
    scale_factor: float,
    metric: str,
) -> np.ndarray:
    """
    Aggregate a simple list of temporal raster specs with one metric.

    Example:
      all months from 1991-2020 -> mean
      all months 5-9 -> sum
    """
    stack = read_temporal_stack_to_grid(
        specs=specs,
        grid=grid,
        resampling=resampling,
        scale_factor=scale_factor,
    )

    return aggregate_stack(
        stack=stack,
        metric=metric,
    ).astype(np.float32)


def aggregate_year_then_across_years(
    spec_factory: Callable[[int, int], TemporalRasterSpec],
    years: list[int],
    months: list[int],
    grid: GridContext,
    resampling: Resampling,
    scale_factor: float,
    within_year_metric: str,
    across_year_metric: str,
) -> np.ndarray:
    """
    Aggregate monthly rasters in two steps:

      1. For each year, aggregate selected months.
      2. Aggregate the yearly results across years.

    Example:
      annual precipitation:
        within_year_metric = sum
        across_year_metric = mean
    """
    if not years:
        raise ValueError("No years provided.")

    if not months:
        raise ValueError("No months provided.")

    yearly_arrays: list[np.ndarray] = []

    for year in years:
        specs = [
            spec_factory(year, month)
            for month in months
        ]

        year_array = aggregate_temporal_specs(
            specs=specs,
            grid=grid,
            resampling=resampling,
            scale_factor=scale_factor,
            metric=within_year_metric,
        )

        yearly_arrays.append(year_array)

    yearly_stack = np.stack(yearly_arrays, axis=0)

    return aggregate_stack(
        stack=yearly_stack,
        metric=across_year_metric,
    ).astype(np.float32)


def aggregate_time_series(
    spec_factory: Callable[[int, int], TemporalRasterSpec],
    years: list[int],
    months: list[int],
    grid: GridContext,
    resampling: Resampling,
    scale_factor: float,
    aggregation_cfg: dict,
) -> tuple[np.ndarray, str]:
    """
    Generic aggregation for year-month time series.

    Supports two modes:

    1. Direct aggregation:
       aggregation_cfg has "metric"

       All selected year-month rasters are stacked together and aggregated.

    2. Two-step aggregation:
       aggregation_cfg has "within_year_metric" and "across_year_metric"

       First aggregate months within each year, then aggregate yearly rasters.
    """
    if "metric" in aggregation_cfg:
        metric = aggregation_cfg["metric"]

        specs = [
            spec_factory(year, month)
            for year in years
            for month in months
        ]

        array = aggregate_temporal_specs(
            specs=specs,
            grid=grid,
            resampling=resampling,
            scale_factor=scale_factor,
            metric=metric,
        )

        return array, metric

    if "within_year_metric" in aggregation_cfg and "across_year_metric" in aggregation_cfg:
        within_year_metric = aggregation_cfg["within_year_metric"]
        across_year_metric = aggregation_cfg["across_year_metric"]
        metric_name = aggregation_cfg.get(
            "output_metric_name",
            f"{across_year_metric}_annual_{within_year_metric}",
        )

        array = aggregate_year_then_across_years(
            spec_factory=spec_factory,
            years=years,
            months=months,
            grid=grid,
            resampling=resampling,
            scale_factor=scale_factor,
            within_year_metric=within_year_metric,
            across_year_metric=across_year_metric,
        )

        return array, metric_name

    raise ValueError(
        "Invalid temporal aggregation. Expected either 'metric' or "
        "'within_year_metric' + 'across_year_metric'. "
        f"Got: {aggregation_cfg}"
    )


def aggregate_monthly_bands(
    raster_path: Path,
    months: list[int],
    grid: GridContext,
    resampling: Resampling,
    scale_factor: float,
    metric: str,
) -> np.ndarray:
    """
    Aggregate selected monthly bands from one multiband raster.

    This is useful for sources where each month is stored as one band:
      band 1  -> January
      band 12 -> December
    """
    specs = [
        TemporalRasterSpec(
            path=raster_path,
            month=month,
            band=month,
        )
        for month in months
    ]

    return aggregate_temporal_specs(
        specs=specs,
        grid=grid,
        resampling=resampling,
        scale_factor=scale_factor,
        metric=metric,
    )