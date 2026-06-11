from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from rasterio.enums import Resampling

from src.pipeline.raster_ops import read_raster_to_grid
from src.pipeline.memory_optimizer import stack_rasters_memory_aware, free_memory


# =============================================================================
# Basic temporal aggregation
# =============================================================================


def aggregate_stack(
    stack: np.ndarray,
    metric: str,
) -> np.ndarray:
    """
    Aggregate a stack of rasters along the first axis.

    Parameters
    ----------
    stack:
        Array with shape (time, height, width).

    metric:
        Aggregation metric. Supported:
          - mean
          - sum
          - std
          - min
          - max

    Notes
    -----
    The stack is expected to use np.nan for nodata.
    Aggregations ignore np.nan values.
    Empty slices (all NaN) are expected and produce NaN output.
    """
    if stack.ndim != 3:
        raise ValueError(
            f"Expected stack with shape (time, height, width), got {stack.shape}"
        )

    # Suppress warnings for empty slices (all NaN values are expected)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if metric == "mean":
            return np.nanmean(stack, axis=0)

        if metric == "sum":
            return np.nansum(stack, axis=0)

        if metric == "std":
            return np.nanstd(stack, axis=0)

        if metric == "min":
            return np.nanmin(stack, axis=0)

        if metric == "max":
            return np.nanmax(stack, axis=0)

    raise ValueError(f"Unsupported aggregation metric: {metric}")


def months_from_range(month_range: list[int]) -> list[int]:
    """
    Convert a month range [start, end] into a list of months.

    Example
    -------
    [5, 9] -> [5, 6, 7, 8, 9]
    """
    if len(month_range) != 2:
        raise ValueError(f"Month range must have two values: {month_range}")

    start_month, end_month = int(month_range[0]), int(month_range[1])

    if start_month < 1 or end_month > 12 or start_month > end_month:
        raise ValueError(f"Invalid month range: {month_range}")

    return list(range(start_month, end_month + 1))


# =============================================================================
# Generic temporal raster specs
# =============================================================================


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
    grid,
    resampling: Resampling,
    scale_factor: float = 1.0,
    resampling_method_name: str | None = None,
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
            resampling_method_name=resampling_method_name,
        )
        arrays.append(array)
        # Free memory between loads if many rasters
        if len(arrays) % 10 == 0:
            gc.collect()

    return np.stack(arrays, axis=0)


def aggregate_temporal_specs(
    specs: list[TemporalRasterSpec],
    grid,
    resampling: Resampling,
    scale_factor: float,
    metric: str,
    resampling_method_name: str | None = None,
) -> np.ndarray:
    """
    Aggregate a simple list of temporal raster specs with one metric.

    Example:
      all months from 1991-2020 -> mean
      selected months 5-9 -> sum
    """
    stack = read_temporal_stack_to_grid(
        specs=specs,
        grid=grid,
        resampling=resampling,
        scale_factor=scale_factor,
        resampling_method_name=resampling_method_name,
    )

    return aggregate_stack(
        stack=stack,
        metric=metric,
    ).astype(np.float32)


def aggregate_year_then_across_years(
    spec_factory: Callable[[int, int], TemporalRasterSpec],
    years: list[int],
    months: list[int],
    grid,
    resampling: Resampling,
    scale_factor: float,
    within_year_metric: str,
    across_year_metric: str,
    resampling_method_name: str | None = None,
) -> np.ndarray:
    """
    Aggregate monthly rasters in two steps:

      1. For each year, aggregate selected months.
      2. Aggregate the yearly results across years.

    Example
    -------
    Mean annual precipitation:
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
            resampling_method_name=resampling_method_name,
        )

        yearly_arrays.append(year_array)

    yearly_stack = np.stack(yearly_arrays, axis=0)
    result = aggregate_stack(
        stack=yearly_stack,
        metric=across_year_metric,
    ).astype(np.float32)
    
    # Free memory after aggregation
    del yearly_arrays, yearly_stack
    gc.collect()
    
    return result


def aggregate_time_series(
    spec_factory: Callable[[int, int], TemporalRasterSpec],
    years: list[int],
    months: list[int],
    grid,
    resampling: Resampling,
    scale_factor: float,
    aggregation_cfg: dict,
    resampling_method_name: str | None = None,
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
            resampling_method_name=resampling_method_name,
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
            resampling_method_name=resampling_method_name,
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
    grid,
    resampling: Resampling,
    scale_factor: float,
    metric: str,
    resampling_method_name: str | None = None,
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
        resampling_method_name=resampling_method_name,
    )
